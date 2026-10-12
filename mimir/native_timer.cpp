// MimIR's tensor extern returns a malloc-owned output, as in MimIR PR #1.
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cstddef>
extern "C" float* mim_kernel(float*,float*);
extern "C" int run_kernel(float* a,float* b,float* output,size_t elements) {
  float* result=mim_kernel(a,b);
  if(!result)return -1;
  std::memcpy(output,result,elements*sizeof(float));
  std::free(result);
  return 0;
}
extern "C" double measure(float* a,float* b,int count) {
  auto start=std::chrono::steady_clock::now();
  for(int i=0;i<count;++i) {
    float* result=mim_kernel(a,b);
    if(!result)return -1;
    std::free(result);
  }
  auto stop=std::chrono::steady_clock::now();
  return std::chrono::duration<double>(stop-start).count()/count;
}
