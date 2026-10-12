#include <ATen/ops/from_blob.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/mv.h>
#include <ATen/ops/convolution.h>
#include <c10/core/InferenceMode.h>
#include <chrono>
#include <exception>
#include <string>

namespace {
struct Kernel {
  int op, stride=1, pad=0;
  at::Tensor a,b,c,result;
  Kernel(int kind, const int* d, float* ap, float* bp, float* cp):op(kind) {
    if(op==0) {
      a=at::from_blob(ap,{d[0],d[2]},at::kFloat); b=at::from_blob(bp,{d[2],d[1]},at::kFloat);
      c=at::from_blob(cp,{d[0],d[1]},at::kFloat);
    } else if(op==1) {
      a=at::from_blob(ap,{d[0],d[1]},at::kFloat); b=at::from_blob(bp,{d[1]},at::kFloat);
      c=at::from_blob(cp,{d[0]},at::kFloat);
    } else {
      stride=d[7];pad=d[8];
      a=at::from_blob(ap,{d[0],d[1],d[2],d[3]},at::kFloat);
      b=at::from_blob(bp,{d[4],d[1],d[5],d[6]},at::kFloat);
      c=at::from_blob(cp,{d[0],d[4],(d[2]+2*pad-d[5])/stride+1,(d[3]+2*pad-d[6])/stride+1},at::kFloat);
    }
  }
};
thread_local std::string error;
}
extern "C" const char* last_error() {return error.c_str();}
extern "C" void* create_kernel(int op,const int* d,float* a,float* b,float* c) {
  try {return new Kernel(op,d,a,b,c);} catch(const std::exception& e) {error=e.what();return nullptr;}
}
extern "C" void destroy_kernel(void* ptr) {delete static_cast<Kernel*>(ptr);}
extern "C" double measure(void* ptr,int count) {
  try {
    auto* k=static_cast<Kernel*>(ptr);c10::InferenceMode guard;
    auto start=std::chrono::steady_clock::now();
    for(int i=0;i<count;++i) {
      if(k->op==0) at::mm_out(k->c,k->a,k->b);
      else if(k->op==1) at::mv_out(k->c,k->a,k->b);
      else k->result=at::convolution(k->a,k->b,std::nullopt,{k->stride,k->stride},{k->pad,k->pad},{1,1},false,{0,0},1);
    }
    auto stop=std::chrono::steady_clock::now();
    // Extracting the result for correctness checking is outside timing.
    if(k->op==2) k->c.copy_(k->result);
    return std::chrono::duration<double>(stop-start).count()/count;
  } catch(const std::exception& e) {error=e.what();return -1;}
}
