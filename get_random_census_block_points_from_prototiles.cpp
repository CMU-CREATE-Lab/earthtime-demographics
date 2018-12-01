#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <assert.h>
#include <malloc.h>

struct dot {
  float x;
  float y;
  int blockIdx;
  int subIdx;
};

int compare(const dot *a, const dot *b) {
  if (a->blockIdx != b->blockIdx) {
    return a->blockIdx - b->blockIdx;
  } else {
    return a->subIdx - b->subIdx;
  }
}


int main() {
  const char *filename = "prototiles/master.bin";
  const char *out_filename = "prototiles/master-sorted-by-block.bin";
  struct stat statbuf;
  stat(filename, &statbuf);
  int ndots = statbuf.st_size / sizeof(dot);
  printf("There are %d dots in %s\n", ndots, filename);
  //ndots = 10000000;
  FILE *in = fopen(filename, "r");
  assert(in);
  struct dot *dots = new dot[ndots];
  int nread = fread(dots, ndots * sizeof(dot), 1, in);
  printf("nread %d\n", nread);
  assert(nread == 1);
  printf("%g %g %d %d\n", dots[0].x, dots[0].y, dots[0].blockIdx, dots[0].subIdx);
  qsort(dots, ndots, sizeof(dot), (int (*)(const void*, const void*))compare);
  printf("%g %g %d %d\n", dots[0].x, dots[0].y, dots[0].blockIdx, dots[0].subIdx);

  FILE *out = fopen(out_filename, "w");
  int nwrote = fwrite((void*)dots, ndots * sizeof(dot), 1, out);
  assert(nwrote == 1);
  printf("Wrote %d sorted dots to %s\n", ndots, out_filename);
  

//(dot*)mmap(NULL, statbuf.st_size, PROT_READ, MAP_SHARED, fd, 0);
//  int count = 0;
//  for (int i = 0; i < ndots; i++) {
//    if (dots[i].blockIdx == 6109119) count++;
//  }
//  printf("count is %d\n", count);
  return 0;
}
