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

#ifndef CUDAVECTOR_HPP
#define CUDAVECTOR_HPP
#ifdef CUDA

#include "math/CObject.hpp"
#include "util/cuda_util.hpp"
#include <cuda_runtime.h>
#include "core/Thread.hpp"

namespace CGF{
  template<class T>
  class Vector;

  template<class T>
  void bindTexture(T* d_array, uint size);

  template<class T>
  void bindTexture2(T* d_array, uint size);

  void bindUIntTexture(uint* d_array, uint size);
  void bindUIntTexture2(uint* d_array, uint size);

  template<class T>
  void bindFloatBlockTexture(T* d_array, uint size);

  void unbindTexture();

  template<class T>
  class CVector : public CObject{
  public:
    CVector(const Vector<T>* vector, const ThreadPool* pool = 0, 
	    bool copy=false);
    CVector(const CVector<T>* vector);
    CVector(ulong size = 0, const ThreadPool* pool = 0, bool copy=false);

    virtual ~CVector();

    CVector<T>& operator=(const CVector<T>& v);
    virtual void allocateReductionBuffers();
    virtual void allocateDevice(const Thread* caller);
    virtual void deallocateDevice(const Thread* caller);
    
    void print(const Thread* caller);

    T* getData(uint i){
      return d_data[i];
    }

    const T* getData(uint i)const{
      return d_data[i];
    }

    T* getData(const Thread* caller){
      int tid = 0;
      if(caller){
	tid = caller->getId();
      }
      return d_data[tid];
    }

    const T* getData(const Thread* caller)const{
      int tid = 0;
      if(caller){
	tid = caller->getId();
      }
      return d_data[tid];
    }

    T* getMappedData(uint i){
      cgfassert(copy);
      return d_mapped[i];
    }

    const T* getMappedData(uint i)const{
      cgfassert(copy);
      return d_mapped[i];
    }

    T sum(const Thread* caller = 0)const;

    ulong getSize()const{return origSize;}
    ulong getPaddedSize()const{return size;}

    void set(const Vector<T>* vec);
    void setReductionBuffer(CVector<T>* vec);

    void scatter(const Thread* caller);
    void gather(const Thread* caller);

    void bindToTexture(const Thread* caller);

  protected:
    void setDefaultRange();
    void allocateHostMemory();
    const Vector<T>* const vec;
    ulong size;
    ulong origSize;
    bool  copy;
    T** d_data;
    T** d_mapped;
    T*  pinned_memory;
    T*  partial_reductions;
    T** d_mapped_reductions;
    CVector<T>* reductionBuffer;
    cudaStream_t* streams;
    template<int N, class U>
    friend class CSpMatrix;
    ulong textureSize;
  };
}

#endif/*CUDA*/
#endif/*CUDAVECTOR_HPP*/
