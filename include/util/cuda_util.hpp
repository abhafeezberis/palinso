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

#ifndef UTIL_HPP
#define UTIL_HPP

#include "core/cgfdefs.hpp"
#include <iostream>

#ifdef CUDA
#include <cuda_runtime.h>
#endif

namespace CGF{

#ifdef CUDA

#if 0
#ifdef BENCHMARK
#define START_BENCHMARK                             \
  cudaEvent_t start_event, stop_event;              \
  float elapsed;                                    \
  cudaSafeCall(cudaEventCreate(&start_event));		\
  cudaSafeCall(cudaEventCreate(&stop_event));		\
  cudaSafeCall(cudaEventRecord(start_event, 0));

#define STOP_BENCHMARK                                                  \
  cudaSafeCall(cudaEventRecord(stop_event, 0));                         \
  cudaSafeCall(cudaEventSynchronize(stop_event));                       \
  cudaSafeCall(cudaEventElapsedTime(&elapsed, start_event, stop_event)); \
  cudaSafeCall(cudaEventDestroy(start_event));                          \
  cudaSafeCall(cudaEventDestroy(stop_event));
#else
#define START_BENCHMARK
#define STOP_BENCHMARK
#endif

#else

#ifdef BENCHMARK
#define START_BENCHMARK                         \
  cudaSafeCall(cudaThreadSynchronize());
#define STOP_BENCHMARK                          \
  cudaSafeCall(cudaThreadSynchronize());
#else
#define START_BENCHMARK
#define STOP_BENCHMARK
#endif
#endif



#ifdef _DEBUG
#include <stdio.h>
#include <assert.h>
#endif

#ifdef _DEBUG
#define cudaCheckError(errorMessage) do {                           \
    cudaThreadSynchronize();                                        \
    cudaError_t err = cudaGetLastError();                           \
    if( cudaSuccess != err) {                                       \
      error("Cuda error: %s (%d) in file '%s' in line %i : %s.\n",	\
            errorMessage, (int)err, __FILE__, __LINE__,             \
            cudaGetErrorString( err) );                             \
      cgfassert(0);                                                 \
    } } while (0)

#else
#define cudaCheckError(errormessage)
#endif

#ifdef _DEBUG
#define cudaSafeCall(call)	do{                                 \
    cudaError_t err = (call);                                   \
    if(err != cudaSuccess){                                     \
      message("Cuda driver error %x in file '%s', line %i. \n", \
              err, __FILE__, __LINE__);                         \
      message("Call: %s failed\n", #call);                      \
      error("%s\n", cudaGetErrorString(err));                   \
      assert(0);                                                \
    }                                                           \
  }while(0)
#else
#define cudaSafeCall(call) (call)
#endif



  /*Wrapper around cudaMalloc in order to detect out of memory
    excpetions and throw them to the caller. The caller should terminate
    the execution and release the memory*/
#define cudaSafeMalloc(ptr, size) do {                                  \
    if((*ptr)!= 0){                                                     \
      message("%s contains %p", #ptr, *ptr);                            \
      throw CUDAException(__LINE__, __FILE__, "device pointer not null"); \
    }                                                                   \
    cudaMalloc((ptr),(size));                                           \
    cudaThreadSynchronize();                                            \
    cudaError_t err = cudaGetLastError();                               \
    if(err != cudaSuccess){                                             \
      throw CUDAException(__LINE__, __FILE__, cudaGetErrorString(err));	\
    }                                                                   \
  }while(0)

#define cudaSafeFree(ptr) do {                                          \
    if((ptr) != 0){                                                     \
      cudaFree((ptr));                                                  \
      cudaError_t err = cudaGetLastError();                             \
      if(err != cudaSuccess){                                           \
        ptr = 0;                                                        \
        throw CUDAException(__LINE__, __FILE__, cudaGetErrorString(err)); \
      }else{                                                            \
        ptr = 0;                                                        \
      }                                                                 \
    }else{                                                              \
      throw CUDAException(__LINE__, __FILE__, "nullpointer");           \
    }                                                                   \
  }while(0)

#if 0
#define cudaDisplayFloatArray(d_array, size, request) do{           \
    float* array = new float[size];                                 \
    cudaSafeCall(cudaMemcpy(array, d_array, sizeof(float)*size,		\
                            cudaMemcpyDeviceToHost));               \
    message("Dumping cuda device array %s, %p", #d_array, d_array);	\
    std::ios_base::fmt_flags ff;                                    \
    ff = std::cout.flags();                                         \
    std::cout << std::scientific;                                   \
    for(int i = 0;i<request;i++){                                   \
      std::cout << i << "\t" << array[i] << "\n";                   \
    }                                                               \
    std::cout.flags(ff);                                            \
    delete[] array;                                                 \
  } while(0)

#endif

  template<class T>
  inline void cudaDisplayFloatArray(T* d_array, int size, int request){
    T* array = new T[size];
    cudaSafeCall(cudaMemcpy(array, d_array, sizeof(T)*(uint)size,
                            cudaMemcpyDeviceToHost));
    message("Dumping cuda device array, %p", d_array);
    std::ios_base::fmtflags ff;
    ff = std::cout.flags();
    std::cout.precision(10);
    std::cout << std::scientific;
    for(int i = 0;i<request;i++){
      std::cout << i << "\t" << array[i] << "\n";
    }
    std::cout.flags(ff);
    delete[] array;
  }


  void display_cuda_info(int dev);
  void init_cuda_thread(int id);
  void exit_cuda_thread();
  void init_cuda_host_thread();

#if 0
  //namespace CGF{
  void bindTexture(float* d_array, int size);
  void bindUIntTexture(int* d_array, int size);
  void bindFloatBlockTexture(float* d_array, int size);

  void unbindTexture();
  //};
#endif
#endif
}
#endif/*UTIL_HPP*/
