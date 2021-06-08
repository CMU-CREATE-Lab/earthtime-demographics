import ctypes, distutils, distutils.sysconfig, os, tempfile
from utils.utils import subprocess_check
import numpy as np
import pandas as pd

def numpy_atomic_save(dest_filename, array):
    dir = os.path.dirname(os.path.abspath(dest_filename))
    try:
        os.makedirs(dir)
    except:
        pass
    tmp_file = tempfile.NamedTemporaryFile(dir=dir, delete=False)
    np.save(tmp_file, array)
    #nrecs = len(array)
    tmp_file.close()
    os.rename(tmp_file.name, dest_filename)
    
    #nbytes = os.stat(dest_filename).st_size
    #sys.stdout.write('Wrote {nbytes} bytes to {dest_filename}\n'.format(**locals()))

# this gives mutable references, never copies
def to_ctype_reference(x):
    if type(x)==bytearray:
        return (ctypes.c_ubyte * len(x)).from_buffer(x)
    if type(x)==np.ndarray:
        return np.ctypeslib.as_ctypes(x)
    if type(x)==np.core.memmap and x.dtype == np.float32:
        return x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    if type(x) is list:
        # ASSUMES LIST OF ARRAYS;  construct a small disposable ctypes array of pointers
        # pointing to the underlying arrays
        colptrs = [to_ctype_reference(c) for c in x]
        # Make sure all columns have same type
        coltype = colptrs[0]._type_
        for colptr in colptrs[1:]:
            assert(colptr._type_ == coltype)
        return (ctypes.POINTER(coltype) * len(x))(*colptrs)
    raise Exception('Unknown type %s in to_ctype' % type(x))

compile_and_load_seq = 0

def compile_and_load(src):
    # python won't reload unless we change the name of the so; make
    # certain we always use a new name by incrementing sequence
    global compile_and_load_seq
    try:
        compile_and_load_seq += 1
    except:
        compile_and_load_seq = 0
    so_suffix = suffix='-%06d.so' % compile_and_load_seq
    with tempfile.NamedTemporaryFile(suffix='.c') as srcfile, tempfile.NamedTemporaryFile(suffix=so_suffix) as sofile:
        srcfile.write(src.encode('utf-8'))
        srcfile.flush()
        cmd = 'gcc -pthread -shared -rdynamic -fno-strict-aliasing'
        cmd += ' -g -DNDEBUG -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC'
        
        cmd += ' -I' + distutils.sysconfig.get_python_inc()
        
        cmd += ' -L' + distutils.sysconfig.get_python_lib()
        
        # libpython seems to be up 2 directories on Anaconda on hal15
        if distutils.sysconfig.get_python_lib().endswith('site-packages'):
            cmd += ' -L' + distutils.sysconfig.get_python_lib() + '/../..'

        #cmd += ' -lpython2.7'
        cmd += ' -lpython3.8'
        
        srcfile_name = srcfile.name
        sofile_name = sofile.name
        cmd += ' {srcfile_name} -o {sofile_name}'.format(**locals())
        print(cmd)
        subprocess_check(cmd)
        return ctypes.cdll.LoadLibrary(os.path.abspath(sofile_name))

##############

def read_csv_url(url):
    # This is a convenience function for use in testing.  It will automatically 
    # process a Google Sheets URL into the appropriate type of result with 
    # @include expansion
    csv_df = read_raw_csv_url(url)
    data_type = header_to_type(csv_df.columns)
    return process_df(csv_df, data_type)

def process_df(df, data_type):
    if(data_type=='waypoints_sheet'):
        return process_waypoints_sheet(df)
    elif(data_type=='csv_or_dotmap_sheet'):
        return process_csv_or_dotmap_sheet(df)
    else:
        return df

def header_to_type(colname_arr):
    # Randy says that column names have to have a particular capitalization to work, so skip normalizing
    # column names for case
    s_colname_arr=[cn.strip() for cn in colname_arr]
    if('Waypoint Title' in s_colname_arr):
        return 'waypoints_sheet'
    elif('Share link identifier' in s_colname_arr):
        return 'csv_or_dotmap_sheet'
    else:
        return 'other'

def process_csv_or_dotmap_sheet(cod_df,prefix=""):
    start_idx, url_arr = get_include_url_arr(cod_df, 'Share link identifier')
    
    inc_df_arr=[]
    if(len(url_arr)>0):
        #print("%sProcessing CSV or Dotmaps include URLS"%(prefix))
        for url in url_arr:
            #print("%s  Loading %r"%(prefix,url))
            inc_df = read_raw_csv_url(url)
            if len(inc_df)<1:
                #print "%s    Empty or null, skipping"%(prefix)
                continue
                
            # inc_df is something, check if it's a waypoints URL
            data_type = header_to_type(inc_df.columns)
            if(not data_type=='csv_or_dotmap_sheet'):
                print("%s    ERROR: included %r is not a CSV or Dotmaps sheet, skipping "%(prefix,url))
                continue
                
            # inc_df is a waypoints sheet, process it.  Potentially it could contain other includes, 
            # so do recursive call
            #print "%s    Processing embedded CSV or Dotmaps sheet %r, init len=%d"%(prefix,url, len(inc_df))
            inc_df = process_csv_or_dotmap_sheet(inc_df, prefix+"    ")
            
            if len(inc_df)<1:
                #print "%s    Empty or null, skipping"%(prefix)
                continue
            
            # inc_df is done and non-empty
            #print "%s    Done processing %r, len=%d"%(prefix, url, len(inc_df))
            inc_df_arr.append(inc_df)
            
    # Add the trimmed initial cod_df to the end of inc_df_arr
    #print("%s  Including rows from original starting at %d"%(prefix,start_idx))
    inc_df_arr.append(cod_df[start_idx:].reset_index(drop=True))

    # process potentially overriden layers.  Later layers override former.
    # i loops over earlier layers, j loops over later layers 
    for i in range(0,len(inc_df_arr)-1):
        for j in range(i+1, len(inc_df_arr)):
            # Check if there are any share link identifiers in common between i and j.
            # If so, remove the overriden row(s) from i
            s1=inc_df_arr[i]['Share link identifier']
            s2=inc_df_arr[j]['Share link identifier']
            s_int= s1[s1.isin(s2)]
            if(len(s_int)>0):
                # Remove overridden ids from s1
                #print "%sRemoving overriden link id(s) %r from inc_df_arr[%d]"%(prefix,list(s_int),i)
                inc_df_arr[i] = inc_df_arr[i][~s1.isin(s_int)]
    # Concatenate all the data frames in order, ignore the index, don't sort the columns
    merge_df = pd.concat(inc_df_arr, ignore_index=True, sort=False)
    
    # Replace nan's with empty strings
    merge_df.fillna('', inplace=True)
    
    # Drop any 'Unnamed' columns
    merge_df = merge_df.loc[:, ~merge_df.columns.str.match('Unnamed')]
    #merge_df = merge_df.rename(columns={cn:'' for cn in merge_df.columns if 'Unnamed' in cn})
    
    #print "%sCSV or Dotmaps sheet sheet: merged %d dfs, result=%d rows"%(prefix, len(inc_df_arr), len(merge_df))
    return merge_df

def process_waypoints_sheet(wpts_df, prefix=""):
    start_idx, url_arr = get_include_url_arr(wpts_df, 'Waypoint Title')
    
    inc_df_arr=[]
    if(len(url_arr)>0):
        #print "%sProcessing waypoints include URLS"%(prefix)
        for url in url_arr:
            #print "%s  Loading %r"%(prefix,url)
            inc_df = read_raw_csv_url(url)
            if len(inc_df)<1:
                #print "%s    Empty or null, skipping"%(prefix)
                continue
                
            # inc_df is something, check if it's a waypoints URL
            data_type = header_to_type(inc_df.columns)
            if(not data_type=='waypoints_sheet'):
                print("%s    ERROR: included %r is not a waypoints sheet, skipping "%(prefix,url))
                continue
                
            # inc_df is a waypoints sheet, process it.  Potentially it could contain other includes, 
            # so do recursive call
            #print "%s    Processing embedded waypoints sheet %r, init len=%d"%(prefix,url, len(inc_df))
            inc_df = process_waypoints_sheet(inc_df, prefix+"    ")
            
            if len(inc_df)<1:
                #print "%s    Empty or null, skipping"%(prefix)
                continue
            
            # inc_df is done and non-empty
            #print "%s    Done processing %r, len=%d"%(prefix, url, len(inc_df))
            inc_df_arr.append(inc_df)
            
    # Add the trimmed initial wpts_df to the end of inc_df_arr
    inc_df_arr.append(wpts_df[start_idx:].reset_index(drop=True))

    # Concatenate all the data frames in order, ignore the index, don't sort the columns
    merge_df = pd.concat(inc_df_arr, ignore_index=True, sort=False)
    
    # Replace nan's with empty strings
    merge_df.fillna('', inplace=True)

    # Drop any 'Unnamed' columns
    merge_df = merge_df.loc[:, ~merge_df.columns.str.match('Unnamed')]

    # Rename any 'Unnamed' columns to empty column names
    #merge_df = merge_df.rename(columns={cn:'' for cn in merge_df.columns if 'Unnamed' in cn})

    #print "%sWaypoints sheet: merged %d dfs, result=%d rows"%(prefix, len(inc_df_arr), len(merge_df))
    return merge_df

# Don't process for type, just return csv df verbatim
def read_raw_csv_url(url):
    # How can first read the text from the URL, then only switch to pandas if it's recognized
    # as one of the special sheet types?
    csv_df = pd.read_csv(sheetUrl2CsvUrl(url),keep_default_na=False,dtype={'Enabled':str,'Share link identifier':str})

    # Get rid of any carriage returns or tabs (which google would have already done if we'd requested tsv)
    csv_df = csv_df.replace({'\n|\r|\t':''},regex=True)
    return csv_df

# Checks column colname in df, extracts any @include URL lines, and returns 
# a tuple containing a start index for the non-include rows and an array of include URLS (which may be empty)
def get_include_url_arr(df, colname):
    # Randy says that column names have to have a particular capitalization to work, so skip normalizing
    # column names for case
    cname_arr = df.columns
    if not colname in cname_arr:
        print("ERROR: %r missing from columm set %r"%(colname,df.columns))
        return (0,[])
    
    url_arr = []
    found_more_at_end=False
    for i in range(0,len(df)):
        colval = df.iloc[i][colname]
        #print "Checking column %r row %d: value=%r"%(colname, i,colval)
        if pd.isnull(colval) or str(colval).strip()=='':
            # Empty value in colname, keep going
            continue
        if '@include' in colval:
            tokens = colval.split()
            if(len(tokens)!=2 or len(tokens)==2 and tokens[0]!='@include'):
                print("ERROR: Malformed @include line in %r: %r"%(colname,colval))
                continue
            # If we got to here, tokens [1] should be a URL.  Strip any quotes that might be around the URL
            url = tokens[1].strip("'\"")
            if(len(url)>0):
                #print "Found @include for %r"%(url)
                # Check if this @include is disabled
                if('Enabled' in cname_arr):
                    enval = df.iloc[i]['Enabled']
                    if(enval.lower()=='false'):
                        #print "  Disabled, not including"
                        continue
                url_arr.append(url)
            else:
                print("ERROR: Malformed @include line in %r: %r"%(colname,colval))
        else:
            # This value is neither empty nor @include, exit and return i
            found_more_at_end=True
            break
            
    # In the case that there are no rows other than @include rows, found_more_at_end will be failse.  In that
    # case increment i past the end and check for that condition in the calling funcition.  Either that or 
    # the last @include row will be preserved, which we don't want
    if(not found_more_at_end):
        i+=1
        
    return (i,url_arr)

def sheetUrl2CsvUrl(sheetUrl):
    tokens = sheetUrl.split('/')
    assert(len(tokens) == 7)
    assert(tokens[4] == 'd')
    docHash = tokens[5]
    assert(len(docHash) > 20)
    edit = tokens[6]
    assert edit[0:9] == 'edit#gid='
    gid = edit[9:]
    return 'https://docs-proxy.cmucreatelab.org/spreadsheets/d/' + docHash + '/export?format=csv&gid=' + gid