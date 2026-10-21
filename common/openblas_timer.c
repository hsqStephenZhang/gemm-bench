#include <cblas.h>
#include <stddef.h>
#include <time.h>
static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
double measure(int m, int n, int k, const float *a, const float *b, float *c,
               int count) {
  double start = now();
  for (int i = 0; i < count; ++i)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, k, b,
                n, 0, c, n);
  return (now() - start) / count;
}
double measure_gemv(int m, int k, const float *a, const float *b, float *c,
                    int count) {
  double start = now();
  for (int i = 0; i < count; ++i)
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, k, 1, a, k, b, 1, 0, c, 1);
  return (now() - start) / count;
}
double measure_conv2d(int n, int ci, int h, int w, int co, int kh, int kw,
                      int stride, int pad, const float *a, const float *b,
                      float *c, float *col, int count) {
  int oh = (h + 2 * pad - kh) / stride + 1,
      ow = (w + 2 * pad - kw) / stride + 1, spatial = oh * ow, k = ci * kh * kw;
  double start = now();
  for (int iter = 0; iter < count; ++iter)
    for (int bn = 0; bn < n; ++bn) {
      for (int ch = 0; ch < ci; ++ch)
        for (int ry = 0; ry < kh; ++ry)
          for (int rx = 0; rx < kw; ++rx)
            for (int y = 0; y < oh; ++y)
              for (int x = 0; x < ow; ++x) {
                int iy = y * stride + ry - pad, ix = x * stride + rx - pad;
                col[(size_t)((ch * kh + ry) * kw + rx) * spatial + y * ow + x] =
                    (iy >= 0 && iy < h && ix >= 0 && ix < w)
                        ? a[((size_t)bn * ci + ch) * h * w + iy * w + ix]
                        : 0;
              }
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, co, spatial, k, 1,
                  b, k, col, spatial, 0, c + (size_t)bn * co * spatial,
                  spatial);
    }
  return (now() - start) / count;
}
