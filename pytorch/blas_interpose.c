/* dyld interposer: route the float BLAS that libtorch_cpu imports from
 * Accelerate (AMX on Apple Silicon) to OpenBLAS (NEON only). Loaded via
 * DYLD_INSERT_LIBRARIES. */
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef OPENBLAS_PATH
#error OPENBLAS_PATH required
#endif
static void *blas;
static long routed;
static void *lookup(const char *name) {
  if (!blas && !(blas = dlopen(OPENBLAS_PATH, RTLD_NOW | RTLD_LOCAL))) {
    fprintf(stderr, "blas_interpose: %s\n", dlerror());
    abort();
  }
  void *f = dlsym(blas, name);
  if (!f) {
    fprintf(stderr, "blas_interpose: missing %s\n", name);
    abort();
  }
  return f;
}
/* OpenBLAS (an OpenMP build) spins forever when called from inside an OpenMP
 * parallel region with more than one thread (e.g. ATen's parallel conv2d
 * calling sgemm_): its nested team gets one thread. Calls made inside a region
 * therefore run with OpenBLAS limited to one thread; the limit is set by the
 * first such call and lifted by the last one. */
static pthread_mutex_t nesting = PTHREAD_MUTEX_INITIALIZER;
static int nested, outer_threads;
static int enter_nested(void) {
  static int (*in_parallel)(void);
  static void (*set_threads)(int);
  static int (*get_threads)(void);
  if (!in_parallel) {
    in_parallel = (int (*)(void))dlsym(RTLD_DEFAULT, "omp_in_parallel");
    set_threads = (void (*)(int))lookup("openblas_set_num_threads");
    get_threads = (int (*)(void))lookup("openblas_get_num_threads");
  }
  if (!in_parallel || !in_parallel())
    return 0;
  pthread_mutex_lock(&nesting);
  if (nested++ == 0) {
    outer_threads = get_threads();
    set_threads(1);
  }
  pthread_mutex_unlock(&nesting);
  return 1;
}
static void leave_nested(int entered) {
  static void (*set_threads)(int);
  if (!entered)
    return;
  if (!set_threads)
    set_threads = (void (*)(int))lookup("openblas_set_num_threads");
  pthread_mutex_lock(&nesting);
  if (--nested == 0)
    set_threads(outer_threads);
  pthread_mutex_unlock(&nesting);
}
#define WRAP(name, ret, params, body)                                          \
  extern ret name params;                                                      \
  static ret openblas_##name params {                                          \
    static ret(*f) params;                                                     \
    if (!f)                                                                    \
      f = (ret(*) params)lookup(#name);                                        \
    ++routed;                                                                  \
    int entered = enter_nested();                                              \
    body                                                                       \
  }                                                                            \
  __attribute__((used, section("__DATA,__interpose"))) static const void       \
      *interpose_##name[2] = {(const void *)openblas_##name,                   \
                              (const void *)name};
#define INTERPOSE(name, params, args)                                          \
  WRAP(name, void, params, f args; leave_nested(entered);)
#define INTERPOSE_RET(name, ret, params, args)                                 \
  WRAP(name, ret, params, ret result = f args; leave_nested(entered);          \
       return result;)
typedef const int ci;
typedef const float cf;
typedef const char cc;
INTERPOSE(sgemm_,
          (cc * ta, cc *tb, ci *m, ci *n, ci *k, cf *al, cf *a, ci *lda, cf *b,
           ci *ldb, cf *be, float *c, ci *ldc),
          (ta, tb, m, n, k, al, a, lda, b, ldb, be, c, ldc))
INTERPOSE(sgemv_,
          (cc * t, ci *m, ci *n, cf *al, cf *a, ci *lda, cf *x, ci *incx,
           cf *be, float *y, ci *incy),
          (t, m, n, al, a, lda, x, incx, be, y, incy))
INTERPOSE(saxpy_, (ci * n, cf *al, cf *x, ci *incx, float *y, ci *incy),
          (n, al, x, incx, y, incy))
INTERPOSE(sscal_, (ci * n, cf *al, float *x, ci *incx), (n, al, x, incx))
INTERPOSE(scopy_, (ci * n, cf *x, ci *incx, float *y, ci *incy),
          (n, x, incx, y, incy))
INTERPOSE_RET(cblas_sdot, float,
              (const int n, cf *x, const int incx, cf *y, const int incy),
              (n, x, incx, y, incy))
/* Number of routed BLAS calls; lets the harness prove the interposer is active.
 */
long gemm_bench_blas_routed(void) { return routed; }
const char *gemm_bench_blas_library(void) { return OPENBLAS_PATH; }
