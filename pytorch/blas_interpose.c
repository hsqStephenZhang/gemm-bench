/* dyld interposer: route the float BLAS that libtorch_cpu imports from Accelerate
 * (AMX on Apple Silicon) to OpenBLAS (NEON only). Loaded via DYLD_INSERT_LIBRARIES. */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef OPENBLAS_PATH
#error OPENBLAS_PATH required
#endif
static void *blas;
static long routed;
static void *lookup(const char *name) {
  if (!blas && !(blas = dlopen(OPENBLAS_PATH, RTLD_NOW | RTLD_LOCAL))) {fprintf(stderr, "blas_interpose: %s\n", dlerror()); abort();}
  void *f = dlsym(blas, name);
  if (!f) {fprintf(stderr, "blas_interpose: missing %s\n", name); abort();}
  return f;
}
#define INTERPOSE(name, ret, params, args) \
  extern ret name params; \
  static ret openblas_##name params {static ret (*f) params; if (!f) f = (ret (*) params)lookup(#name); ++routed; return f args;} \
  __attribute__((used, section("__DATA,__interpose"))) static const void *interpose_##name[2] = {(const void *)openblas_##name, (const void *)name};
typedef const int ci; typedef const float cf; typedef const char cc;
INTERPOSE(sgemm_, void, (cc *ta, cc *tb, ci *m, ci *n, ci *k, cf *al, cf *a, ci *lda, cf *b, ci *ldb, cf *be, float *c, ci *ldc), (ta, tb, m, n, k, al, a, lda, b, ldb, be, c, ldc))
INTERPOSE(sgemv_, void, (cc *t, ci *m, ci *n, cf *al, cf *a, ci *lda, cf *x, ci *incx, cf *be, float *y, ci *incy), (t, m, n, al, a, lda, x, incx, be, y, incy))
INTERPOSE(saxpy_, void, (ci *n, cf *al, cf *x, ci *incx, float *y, ci *incy), (n, al, x, incx, y, incy))
INTERPOSE(sscal_, void, (ci *n, cf *al, float *x, ci *incx), (n, al, x, incx))
INTERPOSE(scopy_, void, (ci *n, cf *x, ci *incx, float *y, ci *incy), (n, x, incx, y, incy))
INTERPOSE(cblas_sdot, float, (const int n, cf *x, const int incx, cf *y, const int incy), (n, x, incx, y, incy))
/* Number of routed BLAS calls; lets the harness prove the interposer is active. */
long gemm_bench_blas_routed(void) {return routed;}
const char *gemm_bench_blas_library(void) {return OPENBLAS_PATH;}
