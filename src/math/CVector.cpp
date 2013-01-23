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

#ifdef CUDA
#include "math/CVector.hpp"
#include "math/Vector.hpp"
#include "core/Exception.hpp"
#include "math/CUDACGOp.hpp"
#include "math/CUDASpmv.hpp"
#include "core/ThreadPool.hpp"
#include <stdio.h>

#define TID (caller?caller->getId():0)
#define NTHREADS 256

#define PINNED

namespace CGF{

  template<class T>
  CVector<T>::CVector(const Vector<T>* vector, const ThreadPool* p, bool _copy):
    CObject(p), vec(vector), copy(_copy){

    d_data   = new T*[n_devices];
    d_mapped = new T*[n_devices];
    d_mapped_reductions = new T*[n_devices];
    streams  = new cudaStream_t[n_devices];
    partial_reductions = 0;

    for(uint i=0;i<n_devices;i++){
      d_data[i]   = 0;
      d_mapped[i] = 0;
      d_mapped_reductions[i] = 0;
    }

    pinned_memory = 0;

    size = vector->size;
    origSize = vector->origSize;

    setDefaultRange();

    if(pool == 0){
      allocateDevice(0);
    }

    if(copy && pool){
      /*Allocate host proxy vector*/
      allocateHostMemory();
    }

    reductionBuffer = 0;
  }

  template<class T>
  CVector<T>::CVector(const CVector<T>* vector):
    CObject(vector->pool), vec(vector->vec), copy(vector->copy){

    d_data   = new T*[n_devices];
    d_mapped = new T*[n_devices];
    d_mapped_reductions = new T*[n_devices];
    streams  = new cudaStream_t[n_devices];
    partial_reductions = 0;

    for(uint i=0;i<n_devices;i++){
      d_data[i]   = 0;
      d_mapped[i] = 0;
      d_mapped_reductions[i] = 0;
    }

    pinned_memory = 0;

    size = vec->size;
    origSize = vec->origSize;

    setDefaultRange();

    if(pool == 0){
      allocateDevice(0);
    }

    if(copy && pool){
      /*Allocate host proxy vector*/
      allocateHostMemory();
    }
    reductionBuffer = 0;
  }

  template<class T>
  CVector<T>::CVector(ulong s, const ThreadPool* p, bool c):
    CObject(p), vec(0), copy(c){

    d_data   = new T*[n_devices];
    d_mapped = new T*[n_devices];
    d_mapped_reductions = new T*[n_devices];
    streams  = new cudaStream_t[n_devices];
    partial_reductions = 0;
    
    for(uint i=0;i<n_devices;i++){
      d_data[i]   = 0;
      d_mapped[i] = 0;
      d_mapped_reductions[i] = 0;
    }

    pinned_memory = 0;
    
    if(s%16==0){
      size = s;
    }else{
      size = ((s/16)+1)*16;
    }

    origSize = s;

    setDefaultRange();

    if(pool == 0){
      allocateDevice(0);
    }    

    if(copy && pool){
      /*Allocate host proxy vector*/
      allocateHostMemory();
    }
    reductionBuffer = 0;
  }

  template<class T>
  CVector<T>::~CVector(){
    if(pool == 0){
      deallocateDevice(0);
    }
    if(copy && pool){
      if(n_devices != 1){
	cudaSafeCall(cudaFreeHost(pinned_memory));
      }
    }
    if(partial_reductions){
      //cudaSafeCall(cudaFreeHost(partial_reductions));
      delete [] partial_reductions;
    }

    delete[] d_data;
    delete[] d_mapped;
    delete[] d_mapped_reductions;
    delete[] streams;
  }

  template<class T>
  CVector<T>& CVector<T>::operator=(const CVector<T>& v){
    if(this == &v){
      /*Self assignment*/
      return *this;
    }
    error("Not implemented yet");
    return *this;
  }

  template<class T>
  void CVector<T>::set(const Vector<T>* vec){
    cgfassert(origSize == vec->getSize());

    /*Copy data*/
    for(uint i=0;i<n_devices;i++){
      cudaSafeCall(cudaMemcpy(d_data[i], vec->data + vRange[i].startBlock,
			      sizeof(T)*vRange[i].range, 
			      cudaMemcpyHostToDevice));      
    }
  }

  template<class T>
  void CVector<T>::setReductionBuffer(CVector<T>* vec){
    reductionBuffer = vec;
    allocateReductionBuffers();
  }

  template<class T>
  void CVector<T>::allocateReductionBuffers(){
#if 0
    int flags = 
      cudaHostAllocDefault;
    /*      cudaHostAllocMapped |
      cudaHostAllocWriteCombined |
      cudaHostAllocPortable;*/
    cudaSafeCall(cudaHostAlloc((void**)&partial_reductions, 
    			       sizeof(T)*n_devices, 
			       flags));
#else
    partial_reductions = new T[n_devices];
#endif

    if(pool == 0){
      /*      cudaSafeCall(cudaHostGetDevicePointer((void**)&d_mapped_reductions[0],
	      (void*) &(partial_reductions[0]),0));*/
    }
  }

  template<class T>
  void CVector<T>::allocateHostMemory(){
    if(n_devices == 1)
      return;

#ifdef PINNED
    int flags = 
      cudaHostAllocMapped |
      cudaHostAllocWriteCombined |
      cudaHostAllocPortable;
        
    cudaSafeCall(cudaHostAlloc((void**)&pinned_memory, 
			       sizeof(T)*size, 
			       flags));
#else
    cudaSafeCall(cudaMallocHost((void**)&pinned_memory, 
				sizeof(T)*size));
#endif
  }

  template<class T>
  void CVector<T>::setDefaultRange(){
    /*Each device get the same ammount of data*/
    uint segments = size/16;
    uint segmentsPerDevice = segments/n_devices;

    uint total_segments = 0;
    for(uint i=0;i<n_devices;i++){
      if(i==n_devices-1){
	segmentsPerDevice += segments%n_devices;
      }
      vRange[i].range      = segmentsPerDevice * 16;
      vRange[i].startBlock = total_segments    * 16;
      vRange[i].endBlock   = total_segments    * 16 + segmentsPerDevice * 16;
      
      mRange[i].range    = segmentsPerDevice   * 16;
      mRange[i].startRow = total_segments      * 16;
      mRange[i].endRow   = total_segments      * 16 + segmentsPerDevice * 16;
      
      total_segments += segmentsPerDevice;
    }
  }

  template<class T>
  void CVector<T>::allocateDevice(const Thread* caller){
    uint tid = 0;
    if(caller){
      tid = caller->getId();
    }

    if(copy && pool){
      /*Each device has its own full copy of the vector*/
      uint lsize = 0;
      for(uint i=0;i<n_devices;i++){
	lsize += vRange[i].range;
      }

      lsize = size; /*Added for square matrices*/

      textureSize = lsize;

      cudaSafeMalloc((void**)&(d_data[tid]), 
		     sizeof(T)*lsize);

      if(vec != 0){
	cudaSafeCall(cudaMemcpy(d_data[tid], vec->data,
				sizeof(T)*size, 
				cudaMemcpyHostToDevice));
      }

      if(n_devices != 1){
#ifdef PINNED
	/*Create mapped device pointers*/
	cudaSafeCall(cudaHostGetDevicePointer((void**)&d_mapped[tid],
					      (void*)  pinned_memory,0));
#endif
      }
    }else{
      /*Each device has only a part of the complete vector*/
      cudaSafeMalloc((void**)&(d_data[tid]), 
		     sizeof(T)*vRange[tid].range);
      if(vec != 0){
	cudaSafeCall(cudaMemcpy(d_data[tid], vec->data + vRange[tid].startBlock,
				sizeof(T)*vRange[tid].range, 
				cudaMemcpyHostToDevice));
      }

      textureSize = vRange[tid].range;
    }
    
    if(partial_reductions){
      /*cudaSafeCall(cudaHostGetDevicePointer((void**)&d_mapped_reductions[tid],
	(void*) &(partial_reductions[tid]),0));*/
    }

    cudaStreamCreate(&streams[tid]);
  }

  template<class T>
  void CVector<T>::deallocateDevice(const Thread* caller){
    uint tid = 0;
    if(caller)
      tid = caller->getId();

    cudaSafeFree(d_data[tid]);
    cudaStreamDestroy(streams[tid]);
  }

  template<class T>
  void CVector<T>::bindToTexture(const Thread* caller){
    bindTexture2(d_data[caller->getId()], textureSize);
  }

  template<class T>
  void CVector<T>::print(const Thread* caller){
    uint tid = 0;
    if(caller)
      tid = caller->getId();

    if(copy){
      if(tid == 0){
	cudaDisplayFloatArray<T>(d_data[tid], size,size);
      }
      getchar();
      if(tid == 1){
	cudaDisplayFloatArray<T>(d_data[tid], size,size);
      }
    }else{
      uint sz = vRange[tid].range;
      
      if(tid == 0){
	cudaDisplayFloatArray<T>(d_data[tid], sz,sz);
      }
      getchar();
      if(tid == 1){
	cudaDisplayFloatArray<T>(d_data[tid], sz,sz);
      }
      getchar();
    }
  }

  template<class T>
  T CVector<T>::sum(const Thread* caller)const{
    cgfassert(this != reductionBuffer);
    cgfassert(reductionBuffer != 0);
    
    T* result;
    if(pool == 0){
      parallel_reduction<T>(this, reductionBuffer, &result, 
			    /*d_mapped_reductions[0]*/0, caller);
      
      cudaSafeCall(cudaMemcpy(&(partial_reductions[0]), 
			      reductionBuffer->d_data[0],
			      sizeof(T), 
			      cudaMemcpyDeviceToHost));

      return partial_reductions[0];
    }else{
      cgfassert(caller != 0);
      
      uint tid = caller->getId();
      parallel_reduction<T>(this, reductionBuffer, &result, 
			    /*d_mapped_reductions[tid]*/0, caller);
      
      cudaSafeCall(cudaMemcpy(&(partial_reductions[tid]), 
			      /*reductionBuffer->d_data[tid]*/result,
			      sizeof(T),
			      cudaMemcpyDeviceToHost));

      caller->sync();
      
      T total_sum = 0;
      for(uint i=0;i<n_devices;i++){
	total_sum += partial_reductions[i];
      }
      return total_sum;
    }
  }

  template<class T>
  void CVector<T>::gather(const Thread* caller){
    START_BENCHMARK;

    if(n_devices == 1 || copy == false)
      return;

    /*Copy vector associated with this thread to pinned_memory*/
    cudaSafeCall(cudaMemcpyAsync(pinned_memory + vRange[TID].startBlock,
				 d_data[TID] + vRange[TID].startBlock,
				 sizeof(T)*vRange[TID].range,
				 cudaMemcpyDeviceToHost, streams[TID]));
#ifdef BENCHMARK
    cudaThreadSynchronize();
#endif
    STOP_BENCHMARK;
  }

  template<class T>
  void CVector<T>::scatter(const Thread* caller){
    START_BENCHMARK;
    if(n_devices == 1 || copy == false)
      return;

    for(uint i=0;i<n_devices;i++){
      if(i==TID){
	continue;
      }
      /*Update vectors on other devices*/
      cudaSafeCall(cudaMemcpyAsync(d_data[TID] + vRange[i].startBlock,
				   pinned_memory + vRange[i].startBlock,
				   sizeof(T)*vRange[i].range,
				   cudaMemcpyHostToDevice, streams[TID]));
    }
    /*In order to measure the elapsed time correctly*/
#ifdef BENCHMARK
    cudaThreadSynchronize();
#endif
    STOP_BENCHMARK;
  }

  template class CVector<float>;
  template class CVector<double>;
}
#endif/*CUDA*/
