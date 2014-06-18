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

#ifndef CUDASPMATRIX_HPP
#define CUDASPMATRIX_HPP
#ifdef CUDA

#include "math/CObject.hpp"
#include "math/CVector.hpp"
#include "math/CUDASpmv.hpp"

#define CSPMATRIX_EXT
//#undef CSPMATRIX_EXT

#ifdef CSPMATRIX_EXT
#define NX 1
namespace CGF{
  inline int HNNX(int n, int nth){
    if(NX > nth/(n*n)){
      //message("N = %d, NTHREADS = %d, NX = %d, maxNX = %d", n, nth, NX, nth/(n*n));
      return nth/(n*n);
    }
    //message("N = %d, NTHREADS = %d, NX = %d, maxNX = %d", n, nth, NX, NX);
    return NX;
  }
}

#endif

namespace CGF{
  class Thread;
  class ThreadPool;


  template<int N, class T>
  class SpMatrix;

  template<int N, class T>
  void parallel_spmv_cuda(T* d_blocks, 
                          int* d_col_indices, 
                          int* d_row_lengths, int* d_row_indices,
                          const T* d_b, T* d_x, int dim, 
                          int n_blocks, int device);

  template<int N, class T>
  void ordered_spmv_cuda(T* d_blocks, 
                         int* d_col_indices, 
                         int* d_row_lengths, int* d_row_indices,
                         int* d_row_map,
                         const T* d_b, T* d_x, int dim, 
                         int n_blocks);



  template<int N, class T>
  class CSpMatrix : public CObject{
  public:
    CSpMatrix(const SpMatrix<N, T> * const matrix, 
              int n_th = 256, 
              TextureOperation tex = TexVector, const ThreadPool* p = 0);
    virtual ~CSpMatrix();
    void computeBlockDistribution();

    virtual void allocateDevice(const Thread* caller);
    virtual void deallocateDevice(const Thread* caller);

    /*Scatter vector b to all devices*/
    void fullScatterVector(const CVector<T>* const b, const Thread* caller);
    /*Copy vector part b to full vec in each device*/
    void preScatterVector(const CVector<T>* const b, const Thread* caller);
    /*Copy vector part b to full vec to other device(s)*/
    void postScatterVector(const CVector<T>* const b, const Thread* caller);
    void preSpmv(const Thread* caller);
    void spmv(CVector<T>* x, const CVector<T>* const b, const Thread* caller);

    int getWidth()const{
      return mat->getWidth();
    }

    int getHeight()const{
      return mat->getHeight();
    }

  protected:
    const SpMatrix<N, T>* const mat;
#ifdef CSPMATRIX_EXT    
    /*Extension for other methods*/
    T** d_ext_blocks;
    int** d_ext_col_indices;
    int** d_ext_row_lengths; /*Block row length per cuda block*/
    int** d_ext_row_indices;
    int** d_ext_row_map;

    int* n_ext_blocks;
#else
    T** d_blocks;
    int** d_col_indices;
    int** d_row_lengths;
    int** d_row_indices;
#endif
    
    TextureOperation texture;
    int n_threads;
    template<class U>
    friend class CVector;
  };
}

#endif/*CUDA*/
#endif/*CUDASPMATRIX_HPP*/
