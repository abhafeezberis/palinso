/* Copyright (C) 2013 by Mickeal Verschoor

   Permission is hereby granted, free of charge, to any person
   obtaining a copy of this software and associated documentation
   files (the "Software"), to deal in the Software without
   restriction, including without limitation the rights to use, copy,
   modify, merge, publish, distribute, sublicense, and/or sell copies
   of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE. */

#include "core/cgfdefs.hpp"
#include "util/cuda_util.hpp"

using namespace CGF;

namespace CGF{
  template<class T>
  __global__ void measure_mach_eps(T* res){
    T mach_eps = 1.0f;
    do{
      mach_eps /= 2.0f;
    }while((T)(1.0f + (mach_eps/2.0f)) != 1.0f);
    res[0] = mach_eps;
  }

  template<class T>
  void measure_precision(){
    dim3 grid(1);
    dim3 threads(1);
    T* cuda_res;
    T res;
    
    cudaSafeCall(cudaMalloc((void**)&cuda_res, sizeof(T)));
    
    measure_mach_eps<<<grid, threads>>>(cuda_res);
    
    cudaSafeCall(cudaMemcpy(&res, cuda_res, sizeof(T),
			    cudaMemcpyDeviceToHost));
    message("GPU precision = %10.10e", res);
    cudaSafeCall(cudaFree(cuda_res));
  }

  void display_cuda_info(int dev){
    cudaDeviceProp prop;
    cudaSafeCall(cudaGetDeviceProperties(&prop, dev));
    
    message("[%d] Name:  %s", dev, prop.name);
    message("[%d] TotalGlobalMem:  %d", dev, (int)prop.totalGlobalMem);
    message("[%d] SharedMemPerBlock:  %d", dev, (int)prop.sharedMemPerBlock);
    message("[%d] regsPerBlock:  %d", dev, (int)prop.regsPerBlock);
    message("[%d] warpSize:  %d", dev, (int)prop.warpSize);
    message("[%d] memPitch:  %d", dev, (int)prop.memPitch);
    message("[%d] maxThreadsPerBlock:  %d", dev, (int)prop.maxThreadsPerBlock);
    message("[%d] maxThreadsDim:  (%d,%d,%d)", dev, (int)prop.maxThreadsDim[0],
	    (int)prop.maxThreadsDim[1],(int)prop.maxThreadsDim[2]);
    message("[%d] maxGridSize:  (%d,%d,%d)", dev, (int)prop.maxGridSize[0],
	    (int)prop.maxGridSize[1],(int)prop.maxGridSize[2]);
    message("[%d] totalConstMem:  %d", dev, (int)prop.totalConstMem);
    message("[%d] major:  %d", dev, (int)prop.major);
    message("[%d] minor:  %d", dev, (int)prop.minor);
    message("[%d] clockRate:  %d", dev, (int)prop.clockRate);
    message("[%d] textureAlignment:  %d", dev, (int)prop.textureAlignment);   
    message("[%d] deviceOverlap:  %d", dev, (int)prop.deviceOverlap);
    message("[%d] multiProcessorCount:  %d", dev, (int)prop.multiProcessorCount);
    message("[%d] kernelExecTimeoutEnabled:  %d", dev, (int)prop.kernelExecTimeoutEnabled);
    message("[%d] integrated:  %d", dev, (int)prop.integrated);
    message("[%d] computeMode:  %d", dev, (int)prop.computeMode);    
    message("[%d] canMapHostMemory: %d", dev, (int)prop.canMapHostMemory);

    measure_precision<float>();
    measure_precision<double>();
  }
  
  void init_cuda_host_thread(){
    //cudaSafeCall(cudaDeviceReset());
    cudaSafeCall(cudaSetDeviceFlags(cudaDeviceMapHost));
  }
  
  void exit_cuda_thread(){
    //cudaSafeCall(cudaDeviceReset());
  }
  
  void init_cuda_thread(uint i){
    //message("init_cuda_thread %d", i);
    int deviceCount;
    
    //cudaSafeCall(cudaDeviceReset());
    
    cudaSafeCall(cudaGetDeviceCount(&deviceCount));
    
    //message("%d cuda devices found", deviceCount);
    
    cgfassert(deviceCount > i);

    cudaSafeCall(cudaSetDeviceFlags(cudaDeviceMapHost));
    
    cudaSafeCall(cudaSetDevice(i));
    
    //display_cuda_info(i);
  }
}
