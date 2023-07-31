import ast, ctypes, datetime, flask, functools, glob, gzip, hashlib, html, imageio, io, json, math, os
import psycopg2, resource, sqlalchemy, struct, subprocess, sys, threading, time, traceback

from dateutil import tz
from flask import after_this_request, request, g
from werkzeug.exceptions import abort
from sqlitedict import SqliteDict
from tileserver_utils import compile_and_load, to_ctype_reference, read_csv_url
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from psql_utils import epsql
from typing import Literal, Tuple, Union

engine = epsql.Engine()

def cputime_ms():
    resources = resource.getrusage(resource.RUSAGE_SELF)
    return 1000.0 * (resources.ru_utime + resources.ru_stime)

def vmsize_gb():
    return float([l for l in open('/proc/%d/status' % os.getpid()).readlines() if l.startswith('VmSize:')][0].split()[1])/1e6


def log(msg):
    date = datetime.datetime.now(tz.tzlocal()).strftime('%Y-%m-%d %H:%M:%S%z')
    req = '%5d' % os.getpid()
    try:
        if g.requestno:
            req += ':%05d' % g.requestno
    except:
        pass
    mem =  '%.3fGB' % vmsize_gb()
    logfile.write(f'{date} {req} {mem}: {msg}\n')
    logfile.flush()

# Choose logfile by running uwsgi with --logto PATH
logfile = sys.stderr

print('Python version: ', sys.version)

if '__file__' in globals():
    log('Starting, path ' + os.path.abspath(__file__))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

containing_dir = os.path.basename(os.path.dirname(os.path.abspath(__file__))) 
is_production = containing_dir == 'dotmaptiles-production-server'
if is_production:
    dotmaptiles_table = "dotmaptiles_production"
else:
    dotmaptiles_table = "dotmaptiles_staging"
log(f'Containing dir {containing_dir}, is_production={is_production}, dotmaptiles_table={dotmaptiles_table}')

engine.execute(f"""
    CREATE TABLE IF NOT EXISTS {dotmaptiles_table} (
        layer_id text primary key,
        layerdef text,
        drawopts jsonb,
        geography_year int
    );""")
engine.execute(f"ALTER TABLE {dotmaptiles_table} ADD COLUMN IF NOT EXISTS geography_year int")

app = flask.Flask(__name__)

requestno_lock = threading.Lock()
requestno = 0

@app.before_request
def handle_before_request_log():
    with requestno_lock:
        global requestno
        requestno += 1
        g.requestno = requestno
    log(f'START REQUEST {request.url}') 

@app.teardown_request
def handle_teardown_request_log(exception):
    msg = 'END REQUEST'
    if exception:
        msg += f', exception={exception}'
    log(msg)

def gzipped(f):
    @functools.wraps(f)
    def view_func(*args, **kwargs):
        @after_this_request
        def zipper(response):
            accept_encoding = request.headers.get('Accept-Encoding', '')
            
            if 'gzip' not in accept_encoding.lower():
                return response
            
            response.direct_passthrough = False
            
            if (response.status_code < 200 or
                response.status_code >= 300 or
                'Content-Encoding' in response.headers):
                return response
            nbytes = len(response.data)
            start_time = time.time()
            gzip_buffer = io.BytesIO()
            gzip_file = gzip.GzipFile(mode='wb',
                                      fileobj=gzip_buffer,
                                      compresslevel=1)
            gzip_file.write(response.data)
            gzip_file.close()
            
            response.data = gzip_buffer.getvalue()
            duration = int(1000 * (time.time() - start_time))
            print(f'{duration}ms to gzip {nbytes} bytes')

            response.headers['Content-Encoding'] = 'gzip'
            response.headers['Vary'] = 'Accept-Encoding'
            response.headers['Content-Length'] = len(response.data)
            
            return response
        zipper
        return f(*args, **kwargs)
    
    return view_func

# msg will be html-escaped, or use html_msg= to send raw html
def abort400(msg=None, html_msg=None):
    if msg and html_msg:
        raise Exception('abort400 should specify eithier msg or html_msg, not both')
    if msg:
        html_msg = html.escape(msg)
    response = flask.Response('<h2>400: Invalid request</h2>%s' % html_msg, status=400)
    response.headers['Access-Control-Allow-Origin'] = '*'
    abort(response)

# packs to the range 0 ... 256^3-1
def unpack_color(f):
    b = math.floor(f / 256.0 / 256.0)
    g = math.floor((f - b * 256.0 * 256.0) / 256.0)
    r = math.floor(f - b * 256.0 * 256.0 - g * 256.0)
    return {'r':r,'g':g,'b':b}

def pack_color(color, encoding=np.float32):
    if encoding == np.float32:
        return color['r'] + color['g'] * 256.0 + color['b'] * 256.0 * 256.0
    else:
        # Return with alpha = 255
        # Correct for PNG
        return 0xff000000 + color['b'] * 0x10000 + color['g'] * 0x100 + color['r']
        # Trying for MP4
        #return 0xff000000 + color['r'] * 0x10000 + color['g'] * 0x100 + color['b']

def parse_color(color, encoding=np.float32):
    color = color.strip()
    c = color
    try:
        if c[0] == '#':
            c = c[1:]
        if len(c) == 3:
            return pack_color({'r': 17 * int(c[0:1], 16),
                               'g': 17 * int(c[1:2], 16),
                               'b': 17 * int(c[2:3], 16)},
                               encoding)
        if len(c) == 6:
            return pack_color({'r': int(c[0:2], 16),
                               'g': int(c[2:4], 16),
                               'b': int(c[4:6], 16)},
                               encoding)
    except:
        pass
    abort400(html_msg='Cannot parse color <code><b>%s</b></code> from spreadsheet.<br><br>Color must be in standard web form, <code><b>#RRGGBB</b></code>, where RR, GG, and BB are each two-digit hexadecimal numbers between 00 and FF.<br><br>See <a href="https://www.w3schools.com/colors/colors_picker.asp">HTML Color Picker</a>' % color)

def parse_colors(colors, encoding=np.float32):
    packed = [parse_color(color, encoding) for color in colors]
    return np.array(packed, dtype = encoding) 

color3dark1 = parse_colors(['#1b9e77','#d95f02','#7570b3'])
color3dark2 = parse_colors(['#66c2a5','#fc8d62','#8da0cb'])

color4dark1 = parse_colors(['#1b9e77','#d95f02','#7570b3','#e7298a'])
color4dark2 = parse_colors(['#b3e2cd','#fdcdac','#cbd5e8','#f4cae4'])
color4dark3 = parse_colors(['#e41a1c','#377eb8','#4daf4a','#984ea3'])
color4dark4 = parse_colors(['#66c2a5','#fc8d62','#8da0cb','#e78ac3'])
color4dark5 = parse_colors(['#e41a1c','#4daf4a','#984ea3','#ff7f00'])

color5dark1 = parse_colors(['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99'])
color5dark2 = parse_colors(['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00'])
color5dark3 = parse_colors(['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854'])

color3light1 = parse_colors(['#7fc97f','#beaed4','#fdc086'])
color3light2 = parse_colors(['#1b9e77','#d95f02','#7570b3'])
color3light3 = parse_colors(['#66c2a5','#fc8d62','#8da0cb'])

color4light1 = parse_colors(['#1b9e77','#d95f02','#7570b3','#e7298a'])
color4light2 = parse_colors(['#b3e2cd','#fdcdac','#cbd5e8','#f4cae4'])
color4light3 = parse_colors(['#e41a1c','#377eb8','#4daf4a','#984ea3'])

color5light1 = parse_colors(['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e'])
color5light2 = parse_colors(['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00'])
color5light3 = parse_colors(['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854'])

prototile_record_format = '<ffii'  # x, y, block id, seq within block
prototile_record_len = struct.calcsize(prototile_record_format)

tile_record_format = '<fff'  # x, y, color
tile_record_len = struct.calcsize(tile_record_format)

default_psql_database = 'census2010'

def query_psql(query, quiet=False, database=None):
    database = database or default_psql_database
    conn = psycopg2.connect(dbname=database, host='/var/run/postgresql')
    before = time.time()
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    elapsed = time.time() - before
    if not quiet:
        sys.stdout.write('Execution of %s\ntook %g seconds and returned %d rows\n' % (query.strip(), elapsed, len(rows)))
    return rows

cache_dir = 'columncache'

def list_datasets():
    return [x for x in sorted(os.listdir(cache_dir)) if not '_hidden' in x]

# Compute relative path from request to /data
def dataroot():
    return '../' * len(request.path.split('/')) + 'data'

def list_columns(dataset):
    dir = f'{cache_dir}/{dataset}'
    if not os.path.exists(dir):
        abort400(html_msg = f'Dataset named "{html.escape(dataset)}" not found.<br><br><a href="{html.escape(dataroot())}">List valid datasets</a>')
    return sorted([os.path.basename(os.path.splitext(c)[0]) for c in (glob.glob(dir + '/*.float32') + glob.glob(dir + '/*.numpy'))])

# Removing the least recent takes O(N) time;  could be make more efficient if needed for larger dicts

class LruDict:
    def __init__(self, max_entries):
        self.max_entries = max_entries
        self.entries = {}
        self.usecount = 0
    
    def has(self, key):
        return key in self.entries
    
    def get(self, key):
        self.use(key)
        return self.entries[key]['data']
    
    def use(self, key):
        self.usecount += 1
        self.entries[key]['lastuse'] = self.usecount

    def insert(self, key, val):
        self.entries[key] = {'data':val}
        self.use(key)
        if len(self.entries) > self.max_entries:
            lru_key, lru_val = None, {}
            for key, val in self.entries.items():
                if not lru_val or val['lastuse'] < lru_val['lastuse']:
                    lru_key, lru_val = key, val
            if lru_val:
                del self.entries[lru_key]

column_cache = LruDict(100) # max entries

def map_as_array(path):
    return np.memmap(path, dtype=np.float32, mode='r')


# def load_column_old(dataset, column):
#     cache_key = f'{dataset}.{column}'
#     if column_cache.has(cache_key):
#         return column_cache.get(cache_key)
#     dir = f'{cache_dir}/{dataset}'
#     if not os.path.exists(dir):
#         abort400(html_msg=f'Dataset named "{html.escape(dataset)}" not found.<br><br><a href="{html.escape(dataroot())}">List valid datasets</a>')
#     cache_filename_prefix = dir + '/' + column
#     cache_filename = cache_filename_prefix + '.float32'
#     if not os.path.exists(cache_filename):
#         if not os.path.exists(cache_filename_prefix + '.numpy'):
#             abort400(html_msg=f'''Column named "{html.escape(column)}" in dataset "{html.escape(dataset)}" not found.<br><br>
#                               <a href="{html.escape(dataroot())}/{html.escape(dataset)}">List valid columns from {html.escape(dataset)}</a>''')
#         data = np.load(open(cache_filename_prefix + '.numpy', 'rb')).astype(np.float32)
#         tmpfile = cache_filename + '.tmp.%d.%d' % (os.getpid(), threading.current_thread().ident)
#         data.tofile(tmpfile)
#         os.rename(tmpfile, cache_filename)

#     data = map_as_array(cache_filename)
#     column_cache.insert(cache_key, data)
#     return data

# Loading columns

GeographyYear = Literal[2010, 2020]

def load_column(dataset: str, column: str, year: GeographyYear) -> Union[np.memmap, str]:
    """Gets the block data of a dataset interpreted for the given year

    Parameters:
        dataset: the name of the dataset
        column: the name of the column
        year: the geography year

    Returns:
        Memory mapped vector of the column

    Raises:
        Exception if the dataset or column does not exist
    """

    load_method = try_load_block_data_2010 if year == 2010 else try_load_block_data_2020
    (success, data) = load_method(dataset, column)

    if not success:
        raise Exception(data)

    return data   

def try_load_block_data_2010(dataset: str, column: str) -> Tuple[bool, Union[np.memmap, str]]:
    """Tries to map the 2010 block data of a dataset

    Parameters:
        dataset: the name of the dataset
        column: the name of the column
    
    Returns:
        A pair where the first element indicates success, and the second is 
        a string with the reason for failure, or the memory mapped vector.
    """
    (cached, cache_key, filename_prefix, filename) = _try_load_from_cache(dataset, column, 2010)

    if isinstance(cached, str):
        return False, cached
    
    if cached is not None:
        return True, cached

    if not os.path.exists(filename):
        if not os.path.exists(f"{filename_prefix}.numpy"):
            return False, f"No column named {column} in dataset {dataset}."

        data = np.load(f"{filename_prefix}.numpy").astype(np.float32)
        tmp_filename = f"{filename}.tmp.{os.getpid()}.{threading.current_thread().ident}"

        data.tofile(tmp_filename)

        os.rename(tmp_filename, filename)

    data = map_as_array(filename)
    column_cache.insert(cache_key, data)

    return True, data

def try_load_block_data_2020(dataset: str, column: str) -> Tuple[bool, Union[np.memmap, str]]:
    """Tries to map the 2020 block data of a dataset

    Parameters:
        dataset: the name of the dataset
        column: the name of the column
    
    Returns:
        A pair where the first element indicates success, and the second is 
        a string with the reason for failure, or the memory mapped vector.
    """
    (cached, cache_key, _, filename) = _try_load_from_cache(dataset, column, 2020)

    if isinstance(cached, str):
        return False, cached
    
    if cached is not None:
        return True, cached

    if not os.path.exists(filename):
        (success, data_2010) = try_load_block_data_2010(dataset, column)

        if not success:
            return success, data_2010

        print(f"Interpolating 2010 column {dataset}.{column} to 2020")

        crosswalk_matrix = load_npz("./crosswalk_matrix_2010_2020.npz")
        data_2020 = crosswalk_matrix.dot(data_2010)
        tmp_filename = f"{filename}.2020.tmp.{os.getpid()}.{threading.current_thread().ident}"

        data_2020.tofile(tmp_filename)

        os.rename(tmp_filename, filename)

    data = map_as_array(filename)
    column_cache.insert(cache_key, data)
    return True, data

def _try_load_from_cache(dataset: str, column: str, year: GeographyYear) -> Tuple[Union[None, np.memmap, str], str, str, str]:
    cache_key = f'{dataset}.{column}{"" if year == 2010 else f".{year}"}'
    
    if column_cache.has(cache_key):
        return column_cache.get(cache_key), "", "", ""
    
    cache_dir = "columncache"
    dataset_dir = f"{cache_dir}/{dataset}"

    if not os.path.exists(dataset_dir):
        return f"No such dataset: {dataset}", "", "", ""

    filename_prefix = f"{dataset_dir}/{column}"
    filename = f"{filename_prefix}{'.2020' if year == 2020 else ''}.float32"

    return [None, cache_key, filename_prefix, filename]


binary_operators = {
    ast.Add:  np.add,
    ast.Sub:  np.subtract,
    ast.Mult: np.multiply,
    ast.Div:  np.divide,
}

unary_operators = {
    ast.USub: np.negative, # negation (unary subtraction)
}

functions = {
    'max': np.maximum,
    'min': np.minimum,
}

def eval_(node, geography_year):
    if isinstance(node, ast.Num): # <number>
        return node.n
    elif isinstance(node, ast.BinOp): # <left> <operator> <right>
        return binary_operators[type(node.op)](eval_(node.left, geography_year), eval_(node.right, geography_year))
    elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
        return unary_operators[type(node.op)](eval_(node.operand, geography_year))
    elif isinstance(node, ast.Call):
        func_name = node.func.id
        if not func_name in functions:
            abort400(f'Function {func_name} does not exist.  Valid functions are ' +
                     ', '.join(sorted(functions.keys())))
        return functions[func_name](*[eval_(arg, geography_year) for arg in node.args])
    elif isinstance(node, ast.Attribute):
        return load_column(node.value.id, node.attr, geography_year)
    abort400('cannot parse %s' % ast.dump(node))

expression_cache = LruDict(50) # 

def eval_layer_column(expr, year: GeographyYear):
    cache_key = hashlib.sha256(f"{expr}{'' if year == 2010 else ';2020'}".encode('utf-8')).hexdigest()
    if expression_cache.has(cache_key):
        return expression_cache.get(cache_key)

    cache_filename = f'expression_cache/{cache_key}.float32'
    
    if not os.path.exists(cache_filename):
        try:
            expr = expr.replace(' DIV ', '/')
            body = ast.parse(expr, mode='eval').body
            evalled = eval_(body, year)
            data = evalled.astype(np.float32)
        except SyntaxError:
            abort400(html_msg = '<pre>' + html.escape(traceback.format_exc(0)) + '</pre>')
        
        try:
            os.mkdir('expression_cache')
        except:
            pass
        
        tmpfile = cache_filename + f'.{year}.tmp.{os.getpid()}.{threading.current_thread().ident}'
        data.tofile(tmpfile)
        os.rename(tmpfile, cache_filename)
    
    data = map_as_array(cache_filename)
    expression_cache.insert(cache_key, data)
    return data    

#def assemble_cols(cols):
#    return np.hstack([c.reshape(len(c), 1) for c in cols]).astype(np.float32)

populations = {}
colors = {}

compute_tile_data_ext = compile_and_load("""
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdio.h>
#include <errno.h>

typedef struct {
  float x;
  float y;
  uint32_t blockIdx;
  uint32_t seq;
} __attribute__ ((packed)) PrototileRecord;

typedef struct {
  float x;
  float y;
  float color;
} __attribute__ ((packed)) TileRecord;

typedef struct {
  unsigned char r, g, b, a;
} __attribute__ ((packed)) RGBA8;

RGBA8 black = {0,0,0,0};

int compute_tile_data(
    const char *prototile_path,
    int incount,
    TileRecord *tile_data,
    int tile_data_length,
    float **populations,
    unsigned int pop_rows,
    unsigned int pop_cols,
    float *colors)
{
    if (incount == 0) return 0;
    if (incount * sizeof(TileRecord) != tile_data_length) {
        return -10;
    }

    int fd = open(prototile_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open prototile_path %s for reading\\n", prototile_path);
        return -1;
    }

    PrototileRecord *p = mmap (0, incount*sizeof(PrototileRecord),
                               PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) return -2000 + errno;

    unsigned outcount = 0;
    for (unsigned i = 0; i < incount; i++) {
        PrototileRecord rec = p[i];
        double seq = rec.seq;
        seq += 0.5;
        for (unsigned c = 0; c < pop_cols; c++) {
            seq -= populations[c][rec.blockIdx];
            if (seq < 0) {
                if (outcount >= incount) return -4;
                tile_data[outcount].x = rec.x;
                tile_data[outcount].y = rec.y;
                tile_data[outcount].color = colors[c];
                outcount++;
                break;
            }
        }
    }
    munmap(p, incount*sizeof(PrototileRecord));
    close(fd);
    return outcount;
}

inline int min(int x, int y) { return x < y ? x : y; }
inline int max(int x, int y) { return x > y ? x : y; }

#include <math.h>

float sumsq(float a, float b) { return a*a + b*b; }


// Negative on fail
int compute_tile_data_box(
    const char *prototile_path,
    int incount,
    unsigned char *tile_box_pops, // [x * y * colors]
    int tile_width_in_boxes, // width=height
    float **populations,
    unsigned int pop_rows,
    unsigned int pop_cols,
    float min_x, float min_y, float max_x, float max_y,
    float level,
    float *block_areas,
    float prototile_subsample)
{
    unsigned char *tile_box_sums = NULL;
    int fd = open(prototile_path, O_RDONLY);
    if (fd < 0) return -1;

    PrototileRecord *p = mmap (0, incount*sizeof(PrototileRecord),
                               PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) return -2000 + errno;

    unsigned outcount = 0;
    int first = 1;

    tile_box_sums = calloc(tile_width_in_boxes * tile_width_in_boxes, 1);
    for (unsigned i = 0; i < incount; i++) {
        // rec is a single prototile dot.  We need to figure out which population it belongs to, if any, and
        // assign color accordingly
        PrototileRecord rec = p[i];

        double block_population = 0.0; // Total population in this block, across all colors/columns
        for (unsigned c = 0; c < pop_cols; c++) {
            block_population += populations[c][rec.blockIdx];
        }
        //// This is the total # of pixels to draw, across all prototiles at this Z level
        //double block_uncorrected_pixels_to_draw = block_population * prototile_subsample * (radius * 2) * (radius * 2);
        //
        //// level 0 pixels are 156543 meters across
        //double level_0_pixel_size = 156543; // meters
        //double size_of_block_in_pixels = block_areas[rec.blockIdx] * pow(4.0, level) / (level_0_pixel_size * level_0_pixel_size) * 40;
        //
        //// blockSubsample of 0.5 means show half the points
        ////double blockSubsample = 1.0 / (1.0 + block_uncorrected_pixels_to_draw / size_of_block_in_pixels);
        //double blockSubsample = 1.0 / (1.0 + sqrt(block_uncorrected_pixels_to_draw / size_of_block_in_pixels));
        //
        ////blockSubsample = 1.0;
        //
        //if (/*rec.blockIdx == 9501143 &&*/ first) {
        //    fprintf(stderr, "ctdp z=%f, seq=%d, prototile_subsample=%f, blockSubsample=%lf, b_uncorrected_pix_to_draw=%lf, size_of_block_pixels=%lf, radius=%f\\n", 
        //            level, rec.blockIdx, prototile_subsample, blockSubsample, 
        //            block_uncorrected_pixels_to_draw, size_of_block_in_pixels, radius);
        //    first = 0;
        //}
        //double seq = rec.seq / blockSubsample + 0.5;
        
        double seq = rec.seq + 0.5;

        for (unsigned c = 0; c < pop_cols; c++) {
            // Loop until we find the right population column for rec (if any)
            seq -= populations[c][rec.blockIdx];
            if (seq < 0) {
                // Prototile dot belongs in population column "c".  Draw with appropriate color
                if (outcount >= incount) {
                  free(tile_box_sums);
                  return -4;  // Illegal
                }

                // x and y are in original projection space (0-256 defines the world)
                // row and col are in box space for the returned tile (0 to tile_width_in_boxes-1)

                // Transform from prototile x,y which are 0.0-255.999999... web mercator coords
                // Transform to box coords for this tile which are row, col

                int col = (rec.x - min_x) / (max_x - min_x) * tile_width_in_boxes;
                int row = (rec.y - min_y) / (max_y - min_y) * tile_width_in_boxes;
                if (0 <= col && col < tile_width_in_boxes && 
                    0 <= row && row < tile_width_in_boxes) {
                    int sums_idx = row * tile_width_in_boxes + col;
                    // Only increment until the overall population in a box sums to 255
                    if (tile_box_sums[sums_idx] < 255) {
                        tile_box_sums[sums_idx]++;
                        int idx = (c * tile_width_in_boxes * tile_width_in_boxes) + sums_idx;
                        tile_box_pops[idx]++;
                    }
                }
                break;
            }
        }
    }
    munmap(p, incount*sizeof(PrototileRecord));
    close(fd);
    free(tile_box_sums);
    return 0;
}

// Negative on fail
int compute_tile_data_png(
    const char *prototile_path,
    int incount,
    RGBA8 *tile_pixels,
    int tile_width_in_pixels, // width=height
    float **populations,
    unsigned int pop_rows,
    unsigned int pop_cols,
    RGBA8 *colors,
    float min_x, float min_y, float max_x, float max_y,
    float radius,
    float level,
    float *block_areas,
    float prototile_subsample)
{
    if (incount == 0) return 0;
    int fd = open(prototile_path, O_RDONLY);
    if (fd < 0) return -1;

    PrototileRecord *p = mmap (0, incount*sizeof(PrototileRecord),
                               PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) return -2000 + errno;

    unsigned outcount = 0;
    int first = 1;

    for (unsigned i = 0; i < incount; i++) {
        // rec is a single prototile dot.  We need to figure out which population it belongs to, if any, and
        // assign color accordingly
        PrototileRecord rec = p[i];

        double block_population = 0.0; // Total population in this block, across all colors/columns
        for (unsigned c = 0; c < pop_cols; c++) {
            block_population += populations[c][rec.blockIdx];
        }
        // This is the total # of pixels to draw, across all prototiles at this Z level
        double block_uncorrected_pixels_to_draw = block_population * prototile_subsample * (radius * 2) * (radius * 2);

        // level 0 pixels are 156543 meters across
        double level_0_pixel_size = 156543; // meters
        double size_of_block_in_pixels = block_areas[rec.blockIdx] * pow(4.0, level) / (level_0_pixel_size * level_0_pixel_size) * 40;

        // blockSubsample of 0.5 means show half the points
        //double blockSubsample = 1.0 / (1.0 + block_uncorrected_pixels_to_draw / size_of_block_in_pixels);
        double blockSubsample = 1.0 / (1.0 + sqrt(block_uncorrected_pixels_to_draw / size_of_block_in_pixels));

        //blockSubsample = 1.0;

        if (/*rec.blockIdx == 9501143 &&*/ first) {
            fprintf(stderr, "ctdp z=%f, seq=%d, prototile_subsample=%f, blockSubsample=%lf, b_uncorrected_pix_to_draw=%lf, size_of_block_pixels=%lf, radius=%f\\n", 
                    level, rec.blockIdx, prototile_subsample, blockSubsample, 
                    block_uncorrected_pixels_to_draw, size_of_block_in_pixels, radius);
            first = 0;
        }

        //double seq = rec.seq / blockSubsample + 0.5;
        double seq = rec.seq / blockSubsample + 0.999;

        for (unsigned c = 0; c < pop_cols; c++) {
            // Loop until we find the right population column for rec (if any)
            seq -= populations[c][rec.blockIdx];
            if (seq < 0) {
                // Prototile dot belongs in population column "c".  Draw with appropriate color
                if (outcount >= incount) return -4;  // Illegal

                // render the point so long as we're within the pixel range of the tile
                // x and y are in original projection space (0-256 defines the world)
                // row and col are in pixel space for the returned tile (0 to tile_width_in_pixels-1)

                // Transform from prototile x,y which are 0.0-255.999999... web mercator coords
                // Transform to pixel coords for this tile which are row, col

                // center_row, col are increased by 0.5 so that (int) behaves like round instead of floor
                float center_col = (rec.x - min_x) / (max_x - min_x) * tile_width_in_pixels + 0.5;
                float center_row = (rec.y - min_y) / (max_y - min_y) * tile_width_in_pixels + 0.5;

                // For more details, see below

                int min_col = max((int) (center_col - radius), 0);
                int min_row = max((int) (center_row - radius), 0);  
                int max_col = min((int) (center_col + radius), tile_width_in_pixels);
                int max_row = min((int) (center_row + radius), tile_width_in_pixels);
                for (int col = min_col; col < max_col; col++) {
                    for (int row = min_row; row < max_row; row++) {
                        tile_pixels[row * tile_width_in_pixels + col] = colors[c];
                    }
                }
                break;
            }
        }
    }
    munmap(p, incount*sizeof(PrototileRecord));
    close(fd);
    return 0;
}

                // Leftmost pixel stretches from 0 <= col < 1
                // Center of the leftmost pixel is col=0.5

                // Rightmost pixel stretch from tile_width_in_pixels-1 <= col < tile_width_in_pixels
                // Center of the rightmost pixel is col=tile_width_in_pixels-0.5

                // Test case, generating tile 0/0/0:  rec.x = 0, min_x = 0, max_x = 512
                // center_col = (0 - 0) / (256 - 0) * 512 + 0.5 = 0.5
                // min_col = int(0.5-0.5) = 0;  max_col = int(0.5+0.5) = 1
                // colors pixel 0, good

                // Test case, generating tile 0/0/0:  rec.x = .499, min_x = 0, max_x = 512
                // center_col = (.499 - 0) / (256 - 0) * 512 + 0.5 = 1.498
                // min_col = int(1.498 - 0.5) = 0;  max_col = int(1.498 + 0.5) = 1
                // colors pixel 0, good

                // Test case, generating tile 0/0/0:  rec.x = 255.99, min_x = 0, max_x = 512
                // center_col = (255.99 - 0) / (256 - 0) * 512 + 0.5 = 512.48
                // min_col = int(512.48 - 0.5) = 511
                // max_col = int(512.48 + 0.5) = 512
                // colors pixel 511, good
""")

def compute_tile_data_c(prototile_path, incount, tile, populations, colors):
    assert(populations[0].dtype == np.float32)
    assert(colors.dtype == np.float32)

    ret = compute_tile_data_ext.compute_tile_data(
        prototile_path.encode('utf-8'),
        int(incount),
        to_ctype_reference(tile),
        len(tile),
        to_ctype_reference(populations),
        populations[0].size, len(populations),
        to_ctype_reference(colors))
    if ret < 0:
        raise Exception(f"compute_tile_data_c failed with {ret}, prototile_path={prototile_path}")
    return ret

def compute_tile_data_png(prototile_path, incount, tile_pixels, tile_width_in_pixels, populations, colors_rgba8,
                          min_x, min_y, max_x, max_y, radius, level, block_areas, prototile_subsample):
    assert(populations[0].dtype == np.float32)
    assert(colors_rgba8.dtype == np.uint32)
    return compute_tile_data_ext.compute_tile_data_png(
        prototile_path.encode('utf-8'),
        int(incount),
        to_ctype_reference(tile_pixels),
        tile_width_in_pixels,
        to_ctype_reference(populations),
        populations[0].size, len(populations),
        to_ctype_reference(colors_rgba8),
        ctypes.c_float(min_x),
        ctypes.c_float(min_y),
        ctypes.c_float(max_x),
        ctypes.c_float(max_y),
        ctypes.c_float(radius),
        ctypes.c_float(level),
        to_ctype_reference(block_areas),
        ctypes.c_float(prototile_subsample))

def compute_tile_data_box(prototile_path, incount, tile_box_pops, tile_width_in_boxes, populations,
                          min_x, min_y, max_x, max_y, level, block_areas, prototile_subsample):
    assert(populations[0].dtype == np.float32)
    return compute_tile_data_ext.compute_tile_data_box(
        prototile_path.encode('utf-8'),
        int(incount),
        to_ctype_reference(tile_box_pops),
        tile_width_in_boxes,
        to_ctype_reference(populations),
        populations[0].size, len(populations),
        ctypes.c_float(min_x),
        ctypes.c_float(min_y),
        ctypes.c_float(max_x),
        ctypes.c_float(max_y),
        ctypes.c_float(level),
        to_ctype_reference(block_areas),
        ctypes.c_float(prototile_subsample))

def generate_tile_data_pixmap(layer, z, x, y, tile_width_in_pixels, format, geography_year, draw_options={}):
    # Load block area column
    block_areas = load_column('geometry_block2010', 'area_web_mercator_sqm', geography_year)

    z = int(z)
    x = int(x)
    y = int(y)
    start_time = time.time()
    max_prototile_level = 10
    if z <= max_prototile_level:
        pz = z
        px = x
        py = y
    else:
        pz = max_prototile_level
        px = int(x / (2 ** (z - max_prototile_level)))
        py = int(y / (2 ** (z - max_prototile_level)))
        
    # remove block # and seq #, add color

    levelSubsample = draw_options.get('levelSubsample', None)
    prototile_dir = compute_prototile_dir(geography_year, levelSubsample)

    print(f"prototiles dir= {prototile_dir}")
    

    prototile_path = compute_prototile_path(prototile_dir, pz=pz, px=px, py=py)
    incount = os.path.getsize(prototile_path) / prototile_record_len

    if levelSubsample:
        prototile_subsamples = [0] * 11
        for (level_str, subsample) in json.load(open('%s/subsamples.json' % prototile_dir)).items():
            prototile_subsamples[int(level_str)] = subsample

        log('****************** prototile_subsamples %s' % prototile_subsamples)
        log('****************** levelSubsample %s' % levelSubsample)
        actual_subsample = prototile_subsamples[z] if z < len(prototile_subsamples) else 1
        desired_subsample = levelSubsample[z] if z < len(levelSubsample) else 1
        prototile_subsample = desired_subsample / actual_subsample
        log('****************** z %d, desired_subsample %g, prototile_subsample %g' % (z, desired_subsample, prototile_subsample))

    else:
        # subsampling factors already baked into prototiles, from C02 Generate prototiles.ipynb
        prototile_subsamples = [
            0.001, # level 0
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
            0.004,
            0.016,
            0.064,
            0.256,
            1.0    # level 10
        ]
        prototile_subsample = 1
        if z < len(prototile_subsamples):
            prototile_subsample = prototile_subsamples[z]

        if z < 5:
            # Further subsample the points
            subsample = 2.0 ** ((5.0 - z) / 2.0)  # z=4, subsample=2;  z=3, subsample=4 ...
            # We're further subsampling the prototile
            prototile_subsample /= subsample
            incount = int(incount / subsample)
    
    local_tile_width = 256.0 / 2 ** int(z)
    min_x = int(x) * local_tile_width
    max_x = min_x + local_tile_width
    min_y = int(y) * local_tile_width
    max_y = min_y + local_tile_width

    # z=10, r=0.5, area=1
    # z=11, r=sqrt(2)/2, area=2
    # z=12, 1, area=4
    # z=13, sqrt(2), area=8
    
    if z <= 10:
        radius = 0.5
    else:
        #radius = 2 ** (z-11)  # radius 1 for z=11.  radius 2 for z=12.  radius 4 for z=13
        radius = 2 ** ((z-12)/2.0)  # radius 1 for z=11.  radius 2 for z=12.  radius 4 for z=13

    if format == 'box':
        tile_data = np.zeros((len(layer['colors_rgba8']), tile_width_in_pixels, tile_width_in_pixels), dtype=np.uint8)
        if incount > 0:
            status = compute_tile_data_box(prototile_path, incount, tile_data, tile_width_in_pixels,
                                           layer['populations'],
                                           min_x, min_y, max_x, max_y, 
                                           z, block_areas, prototile_subsample)
        else:
            status = 0
    else:
        bytes_per_pixel = 4 # RGBA x 8-bit
        tile_data = np.zeros((tile_width_in_pixels, tile_width_in_pixels, bytes_per_pixel), dtype=np.uint8)
        if incount > 0:
            status = compute_tile_data_png(prototile_path, incount, tile_data, tile_width_in_pixels,
                                           layer['populations'], layer['colors_rgba8'],
                                           min_x, min_y, max_x, max_y, radius,
                                           z, block_areas, prototile_subsample)
        else:
            status = 0
    if status < 0:
        raise Exception('compute_tile_data returned error %d.  path %s %d' % (status, prototile_path, incount))

    duration = int(1000 * (time.time() - start_time))
    log(f'{z}/{x}/{y}: {duration}ms to create pixmap tile from prototile')

    return tile_data

def compute_prototile_path(prototile_dir, pz, px, py):
    prototile_path = "%s/%d/%d/%d.bin" % (prototile_dir, pz, px, py)
    return prototile_path

def compute_prototile_dir(geography_year, levelSubsample):
    if levelSubsample:
        log('draw_options.levelSubsample exists, using prototiles003')
        prototile_dir = "prototiles003"
    else:
        log('draw_options.levelSubsample does not exist, using prototiles')
        log(f'using prototiles for geography {geography_year}')
        prototile_dir = f"prototiles{'' if geography_year == 2010 else '.2020'}"
    return prototile_dir

def gzip_buffer(buf, compresslevel=1):
    str = io.BytesIO()
    out = gzip.GzipFile(fileobj=str, mode='wb', compresslevel=compresslevel)
    out.write(buf)
    out.flush()
    return str.getvalue()

def generate_tile_data_png_or_box(layer, z, x, y, tile_width_in_pixels, format, geography_year, draw_options={}):
    tile_data = generate_tile_data_pixmap(layer, z, x, y, tile_width_in_pixels, format, geography_year, draw_options)
    if format == 'png':
        out = io.BytesIO()
        #scipy.misc.imsave(out, tile_data, format='png')
        imageio.imwrite(out, tile_data, format='png')
        png = out.getvalue()
        return png
    else:
        return tile_data.tobytes()
    
def generate_tile_data_mp4(layers, z, x, y, tile_width_in_pixels, geography_year):
    # make ffmpeg

    cmd = ['/usr/bin/ffmpeg']
    # input from stdin, rgb24
    cmd += ['-pix_fmt', 'rgba', '-f', 'rawvideo', '-s', '%dx%d' % (tile_width_in_pixels, tile_width_in_pixels), '-i', 'pipe:0', '-r', '10']
    # output encoding
    #cmd += ['-vcodec', 'libx264', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-crf', '20', '-g', '10', '-bf', '0']
    cmd += ['-vcodec', 'libx264', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-crf', '20']
    #cmd += ['-vcodec', 'libx264', '-preset', 'slow', '-pix_fmt', 'yuv444p', '-crf', '20']

    #cmd += ['-c:v', 'libvpx-vp9', '-crf', '35', '-threads', '8', '-b:v', '10000k', '-pix_fmt', 'yuv444p']

    
    cmd += ['-movflags',  'faststart'] # move TOC to beginning for fast streaming

    video_path = '/tmp/tile.%d.%s.mp4' % (os.getpid(), threading.current_thread().name)
    #video_path = '/tmp/tile.%d.%s.webm' % (os.getpid(), threading.current_thread().name)
    
    cmd += ['-y', video_path]

    log('about to start ffmpeg')
    before_time = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=sys.stderr, stderr=sys.stderr)
    log('started ffmpeg')

    for frameno in range(0, len(layers)):
        layer = layers[frameno]
        tile_pixels = generate_tile_data_pixmap(layer, z, x, y, tile_width_in_pixels, geography_year, 'mp4')
        log('about to spoot frame %d of len %d' % (frameno, len(tile_pixels)))
        p.stdin.write(tile_pixels)
        log('done')
        #(out, err) = p.communicate(tile_pixels)
        #log('saw out:%s err:%s' % (out, err))

    p.stdin.flush()
    p.stdin.close()
    ret = p.wait()
    ret
    encoding_time = time.time() - before_time
    video_contents = open(video_path).read()
    os.unlink(video_path)
    log('%s/%s/%s VIDSIZE %dKB TIME %.1fs CMD %s' % (z, x, y, int(len(video_contents)/1024), encoding_time, ' '.join(cmd)))
    
    return video_contents
    
def generate_tile_data(layer, z, x, y, geography_year, use_c=False):
    start_time = time.time()
    # remove block # and seq #, add color
    
    prototile_path = f'prototiles{"" if geography_year == 2010 else ".2020"}/{z}/{x}/{y}.bin'
    incount = int(os.path.getsize(prototile_path) / prototile_record_len)
    
    # Preallocate output array for returned tile
    tile = bytearray(tile_record_len * incount)

    if incount > 0:
        assert use_c
        outcount = compute_tile_data_c(prototile_path, incount, tile, layer['populations'], layer['colors'])
    else:
        outcount = 0

    if outcount < 0:
        raise Exception('compute_tile_data returned error %d' % outcount)

    duration = int(1000 * (time.time() - start_time))
    log(f'{z}/{x}/{y}: {duration}ms to create tile from prototile')

    return tile[0 : outcount * tile_record_len]

layer_cache = {
    2010: LruDict(50), # max entries
    2020: LruDict(50)
}

def find_or_generate_layer(layerdef, geography_year):
    if layer_cache[geography_year].has(layerdef):
        print(f'Using cached {layerdef}')
        return layer_cache[geography_year].get(layerdef)

    start_time = time.time()
    start_cputime_ms = cputime_ms()
    
    layerdef_hash = hashlib.md5(f"{layerdef}{'' if geography_year == 2010 else ';2020'}".encode('utf-8')).hexdigest()
    log(f'{layerdef_hash}: computing from {layerdef} with {geography_year} geography')
    colors = []
    populations = []
    for (color, expression) in [x.split(';') for x in layerdef.split(';;')]:
        colors.append(color)
        populations.append(eval_layer_column(expression, geography_year))

    layer = {'populations': populations,
             'colors': parse_colors(colors, encoding=np.float32),
             'colors_rgba8': parse_colors(colors, encoding=np.uint32),
             'year': geography_year}
    layer_cache[geography_year].insert(layerdef, layer)
    duration = int(1000 * (time.time() - start_time))
    cpu = cputime_ms() - start_cputime_ms
    log(f'{layerdef_hash}: {duration}ms ({cpu}ms CPU) to create')
    return layer



@app.route('/tilesv1/<layersdef>/512x512/<z>/<x>/<y>.mp4')
def serve_video_tile_v1_mp4(layersdef, z, x, y):
    geography_year = 2010
    x = int(int(x) / 4)
    y = int(int(y) / 4)
    (x,y)=(y,x)
    try:
        layers = [find_or_generate_layer(layer, geography_year) for layer in layersdef.split(';;;')]
        tile_width_in_pixels = 1024
        tile = generate_tile_data_mp4(layers, z, x, y, tile_width_in_pixels)
        
        #response = flask.Response(tile, mimetype='video/mp4')
        response = flask.Response(tile, mimetype='video/webm')
    except:
        print(traceback.format_exc())
        raise
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/tilesv1/<layerdef>/<z>/<x>/<y>.png')
def serve_tile_v1_png(layerdef, z, x, y):
    geography_year=2010
    try:
        layer = find_or_generate_layer(layerdef, geography_year)
        tile_width_in_pixels = 512
        tile = generate_tile_data_png_or_box(layer, z, x, y, tile_width_in_pixels, 'png', geography_year)
        #outcount = len(tile) / tile_record_len
        
        response = flask.Response(tile, mimetype='image/png')
    except:
        print(traceback.format_exc())
        raise
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

def generate_tile_data_tbox(layers, z, x, y, tile_width_in_pixels, geography_year, draw_options={}):
    frames = [generate_tile_data_png_or_box(layer, z, x, y, tile_width_in_pixels, 'box', geography_year, draw_options) for layer in layers]
    return b''.join(frames)

# HTML description of .box tile
def describe_tile_box(tile, tile_width_in_pixels):
    n_pixels = tile_width_in_pixels ** 2
    assert len(tile) % n_pixels == 0
    num_colors = len(tile) // n_pixels
    ret = f'<pre>Number of colors: {num_colors}\n'
    for i in range(num_colors):
        color_array = np.frombuffer(tile, dtype=np.uint8, count=n_pixels, offset=i * n_pixels)
        nonzero = np.count_nonzero(color_array)
        ret += f'   Color {i}: {nonzero} nonzero values'
        if nonzero:
            ret += f', with average {color_array.sum() / nonzero:.5f}'
        ret += '\n'
    ret += '</pre>\n'
    return ret

# Serves v1 and v2 "box" (single-frame) tiles 
def serve_tile_box(layerdef, z, x, y, geography_year, draw_options=None):
    log(f'serve tile box draw options= {draw_options}')
    log(f'serve tile box geography year= {geography_year}')
    try:
        layer = find_or_generate_layer(layerdef, geography_year)
        tile_width_in_pixels = 256
        tile = generate_tile_data_png_or_box(layer, z, x, y, tile_width_in_pixels, 'box', geography_year, draw_options)
        if 'debug' in request.args:
            # decode the tile instead
            response = flask.Response(describe_tile_box(tile, tile_width_in_pixels))
        else:
            response = flask.Response(tile, mimetype='application/octet-stream')
            
    except:
        print(traceback.format_exc())
        raise
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


# .box V1 (single frame), using full definition in URL
# This is deprecated in favor of .box v2, because complex definitions won't fit in the URL
@app.route('/tilesv1/<layerdef>/<z>/<x>/<y>.box')
@gzipped
def serve_tile_v1_box(layerdef, z, x, y):
    log(f'DEPRECATED URL {request.url};  switch to /tilesv2/layername instead')
    geography_year = 2010
    return serve_tile_box(layerdef, z, x, y, geography_year)

# .box V2 (single frame), using layer name
# Looks up layer defintion -- column expressions and colors -- from database, which itself is loaded
#  from google sheet
@app.route('/tilesv2/<layername>/<z>/<x>/<y>.box')
@gzipped
def serve_tile_v2_box(layername, z, x, y):
    #dotmap_dict = SqliteDict(dotmap_layerdef_path, flag='c',
    #                         tablename=dotmap_layerdef_table_name,
    #                         autocommit=False)
    return serve_tile_box(get_layerdef(layername), z, x, y, get_geography_year(layername), get_draw_options(layername))

# .tbox V1 (animated)
@app.route('/tilesv1/<layerdefs>/<z>/<x>/<y>.tbox')
@gzipped
def serve_tile_v1_tbox(layerdefs, z, x, y):
    geography_year=2010
    try:
        if isinstance(layerdefs, list):
            layers = layerdefs
        else:
            layers = [find_or_generate_layer(layerdef, geography_year) for layerdef in layerdefs.split(';;;')]
        tile_width_in_pixels = 256
        tile = generate_tile_data_tbox(layers, z, x, y, tile_width_in_pixels, geography_year)
        
        response = flask.Response(tile, mimetype='application/octet-stream')
    except:
        print(traceback.format_exc())
        raise
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


def get_layer_from_db(layername):
    log(layername)
    dicts = engine.execute_returning_dicts(f'SELECT * FROM {dotmaptiles_table} WHERE layer_id = %(layer_id)s;', {'layer_id': layername})
    if not dicts:
        abort400('Cannot find layer named "%s" in local database (consider running reload-layers.)' % layername)
    return dicts[0]

def get_layerdef(layername):
    return get_layer_from_db(layername)['layerdef']

def get_draw_options(layername):
    return get_layer_from_db(layername)['drawopts']
    
def get_geography_year(layername: str) -> Literal[2010, 2020]:
    return get_layer_from_db(layername)['geography_year']

# .tbox V2 (animated)
@app.route('/tilesv2/<layername>/<z>/<x>/<y>.tbox')
@gzipped
def serve_tile_v2_tbox(layername, z, x, y):
    frame_layernames = get_layerdef(layername).split('|')
    geography_years = sorted({get_geography_year(frame_layername) for frame_layername in frame_layernames})

    assert len(geography_years) == 1, f"All layers must have identical geography year for {layername} but instead have years {geography_years}."

    geography_year = geography_years[0]

    log(f'serving tbox: layer {layername} expands to frames {frame_layernames}')

    layers = [find_or_generate_layer(get_layerdef(frame_layername), geography_year) for frame_layername in frame_layernames]

    draw_options = get_draw_options(layername)
    
    try:
        tile_width_in_pixels = 256
        tile = generate_tile_data_tbox(layers, z, x, y, tile_width_in_pixels, geography_year, draw_options)
        
        response = flask.Response(tile, mimetype='application/octet-stream')
    except:
        print(traceback.format_exc())
        raise
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response
    #return serve_tile_v1_tbox(layers, z, x, y)

# .bin is the first tile format, with every point enumerated, vector-style
@app.route('/tilesv1/<layerdef>/<z>/<x>/<y>.<suffix>')
@gzipped
def serve_tile_v1(layerdef, z, x, y, suffix):
    geography_year=2010
    assert(suffix != 'box')

    if geography_year == 2020:
        log("Using 2020 geographies")

    try:
        layer = find_or_generate_layer(layerdef, geography_year)
        tile = generate_tile_data(layer, z, x, y, geography_year, use_c=True)
        outcount = len(tile) / tile_record_len
        
        if suffix == 'debug':
            html = '<html><head></head><body>'
            html += f'tile {layer}/{z}/{y}/{x}  has {outcount} points<br>'
            for i in range(0, min(outcount, 10)):
                html += f'Point {i}: '
                html += ', '.join([str(x) for x in struct.unpack_from(tile_record_format, tile, i * tile_record_len)])
                html += '<br>\n'
            if outcount > 10:
                html += '...<br>'
                html += '</body></html>'
                    
            return flask.Response(html, mimetype='text/html')
        elif suffix == 'bin':
            response = flask.Response(tile[0 : outcount * tile_record_len], mimetype='application/octet-stream')
        else:
            abort400(f'Invalid suffix {suffix}')
    except:
        print(traceback.format_exc())
        if suffix == 'debug':
            html = '<html><head></head><body><pre>\n'
            html += traceback.format_exc()
            html += '\n</pre></body></html>'
            return html
        else:
            raise
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/data')
def show_datasets():
    html = '<html><head></head><body><h1>Available datasets:</h1>\n'
    for ds in list_datasets():
        html += f'<a href="data/{ds}">{ds}</a><br>\n'
    html += '</body></html>'
    return html

def compute_json_path(dataset):
    return f'{cache_dir}/{dataset}/description.json'

@app.route('/data/<dataset>.json')
def dataset_json(dataset):
    ret = flask.Response(open(compute_json_path(dataset)).read(), mimetype="application/json")
    ret.headers['Access-Control-Allow-Origin'] = '*'
    return ret

@app.route('/data/<dataset>')
def show_dataset_columns(dataset):
    if os.path.exists(compute_json_path(dataset)):
        return open('description.html').read()
    try:
        html = []
        columns = list_columns(dataset)
        if dataset == 'census2000_block2010':
            columns = [c for c in columns if c == c.upper()]
        html.append('<a href="../data">Back to all datasets</a><br>')
        html.append(f'<h1>Columns in dataset {dataset}:</h1>')
        for col in columns:
            html.append(f'{col}<br>')
        html.append('</body></html>')
        return '\n'.join(html)
    except:
        print(traceback.format_exc())
        raise

@app.route('/data/<dataset>/<column>.float32')
def serve_column_float32(dataset, column):
    data = load_column(dataset, column, 2020 if column.endswith("2020") else 2010)
    ret = flask.Response(data.tobytes(), mimetype="application/octet-stream")
    ret.headers['Access-Control-Allow-Origin'] = '*'
    return ret

@app.route('/')
def hello():
    return """
<html><head></head><body>
Test tiles:<br>
<a href="/tilesv1/%230000ff;min(census2000_block2010.p001001%2Ccensus2010_block2010.p001001);;%23ff0000;max(0%2Ccensus2000_block2010.p001001-census2010_block2010.p001001);;%2300ff00;max(0%2Ccensus2010_block2010.p001001-census2000_block2010.p001001)/0/0/0.debug">Pop change 2000-2010 0/0/0</a>
"""
 
@app.route('/test')
def test_1990():
    return open('test-1990-hierarchy.html').read()

@app.route('/assets/<filename>')
def get_asset(filename):
    return open('assets/' + filename).read()

#app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    with app.test_request_context(sys.argv[1]):
        response = app.full_dispatch_request()
        print(response.status_code)
        print(response.get_data())
        
@app.route('/reload-layers')
def reload_layers():

    # # Will mutate records by setting mirroredTime, baseKey, tablename, view
    # def upsert(self, con, records, mirroredTime):
    #     if not records:
    #         return
    #     for record in records:
    #         record.update(self.db_indices()) # set baseKey, tablename, view
    #         record['mirroredTime'] = mirroredTime

    #     before = con.scalar(mirror_table.count())
    #     ins = psql_insert(mirror_table)
    #     upsert = ins.on_conflict_do_update(
    #         constraint=mirror_table.primary_key,
    #         set_=dict(
    #             mirroredTime=ins.excluded.mirroredTime,
    #             createdTime=ins.excluded.createdTime,
    #             fields=ins.excluded.fields))
    #     con.execute(upsert, records)
    #     after = con.scalar(mirror_table.count())
    #     n_inserted = after - before
    #     print(f'Upserted {len(records)} records ({len(records)-n_inserted} updated, {n_inserted} inserted): {self.compositeName()}')

    # This points to the default version of the layer sheet.  Be aware that anyone
    # using non-standard dotmap sheet, or anyone modifying the default sheet without
    # calling reload-layers will not get what they want.
    dotmap_url = 'https://docs.google.com/spreadsheets/d/1rCiksJv4aXi1usI0_9zdl4v5vuOfiHgMRidiDPt1WfE/edit#gid=358696896'

    dotmap_df = read_csv_url(dotmap_url)

    # Calculate column names
    col_count = 10

    color_colnames = ['Color%d'%(i) for i in range(1, col_count+1)]
    def_colnames = ['Definition%d'%(i) for i in range(1, col_count+1)]
    for i in range(0,col_count):
        if (not color_colnames[i] in dotmap_df.columns or
            not def_colnames[i] in dotmap_df.columns):
            col_count = i
            break
    
    records = []
    errors = []

    # For each row in dotmap_df, extract all the non-empty colors into a string
    for idx in dotmap_df.index:
        layer_id = dotmap_df.at[idx,'Share link identifier']
        draw_opts_str = dotmap_df.at[idx, 'Draw Options']
        anim_str = dotmap_df.at[idx,'AnimationLayers']
        try:
            geography_year = int(dotmap_df.at[idx,'GeographyYear'])
        except:
            geography_year = 2010

        record = {'layer_id': layer_id, 'drawopts': '{}', 'geography_year': geography_year}

        if not pd.isna(draw_opts_str) and not draw_opts_str=='':
            try:
                record['drawopts'] = json.dumps(json.loads(draw_opts_str))
            except:
                errors.append(['Error parsing Draw Options JSON for {layer_id}'])

        if not pd.isna(anim_str) and not anim_str=='':
            # This is an animation layer, store anim_str
            # in dotmap_dict
            record['layerdef'] = anim_str
        else:
            # Not an animiation layer, put together the color and definition fields
            ldef_arr = []
            for i in range(0, col_count):
                color_str = dotmap_df.at[idx,color_colnames[i]]
                def_str =   dotmap_df.at[idx,def_colnames[i]]
                if pd.isna(color_str) or pd.isna(def_str) or color_str == '' or def_str == '':
                    # We're done here
                    break

                # Extend ldef_arr
                ldef_arr.append('%s;%s'%(color_str,def_str))

            # Join the ldef elements from each set of color/definitions
            # for this layer into a single string
            record['layerdef'] = ';;'.join(ldef_arr)

        records.append(record)

    with engine.connect() as con:
        for record in records:
            con.execute(f"""
                INSERT INTO {dotmaptiles_table} (layer_id, layerdef, drawopts, geography_year)
                    VALUES (%(layer_id)s, %(layerdef)s, %(drawopts)s, %(geography_year)s)
                ON CONFLICT (layer_id) DO
                    UPDATE SET layerdef=excluded.layerdef, drawopts=excluded.drawopts, geography_year=excluded.geography_year;""", 
                record)


    count = engine.execute_count(f'SELECT COUNT(*) FROM {dotmaptiles_table};')

    response = f'''
<pre>
{len(records)} layers loaded with {len(errors)} errors
{chr(10).join(errors)}
There are now {count} records in table
</pre>'''

    return(response)

