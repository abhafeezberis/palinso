/* Copyright (C) 2012 by Mickeal Verschoor

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

#ifndef PARALLELSPMVCUDATASK_HPP
#define PARALLELSPMVCUDATASK_HPP
#ifdef CUDA

#include <cuda_runtime.h>
#include "core/Task.hpp"
#include "core/BenchmarkTimer.hpp"
#include "math/Math.hpp"
#include "math/CSpMatrix.hpp"

namespace CGF{

  template<class T>
  class CGFAPI Vector;

  template<int N, class T>
  class CGFAPI SpMatrix;

  template<int N, class T>
  class CGFAPI ParallelSPMVCudaTask : public Task{
  public:
    ParallelSPMVCudaTask(const ThreadPool* pool, Vector<T>* const x, 
			 const SpMatrix<N, T>* const mat, 
			 const Vector<T>* const b,
			 int n_thr=256, TextureOperation tex=TexVector, 
			 int d_offset=0);
    virtual ~ParallelSPMVCudaTask();

    virtual void execute(const Thread* caller);
    virtual void allocate(const Thread* caller);
    virtual void deallocate(const Thread* caller);
    virtual void copyResult(const Thread* caller);
    virtual void updateBlocks(const Thread* caller);
    virtual void solveSystem(const Thread* caller);
    //virtual void exportSummary();

  protected:
    Vector<T>* x;
    CVector<T>* cx;
    const SpMatrix<N, T>* mat;
    const Vector<T>* b;
    CVector<T>* cb;
    CVector<T>* tmp;

    CSpMatrix<N, T>* cmat;
    
#if 0
    T** d_blocks;
    uint**  d_col_indices;
    uint**  d_row_lengths;
    uint**  d_row_indices;
    T** d_full_vec;
#endif
    T** d_res;

#if 0
    uint* n_blocks;
    MatrixRange* mRange;
    VectorRange* vRange;
    uint* startBlock;
    T* reductions1;
    T* reductions2;
    T* reductions3;

    Vector<T>* r;

    Vector<T>* scratch1;
    Vector<T>* scratch2;

    Vector<T>* u;
    T*   pinned_memory;
    T**  mapped_memory;

    uint* k;
#endif
    BenchmarkTimer* timers;
    int n_cuda_threads;
    TextureOperation texture;
    int device_offset;
  };
}

#endif/*CUDA*/
#endif/*PARALLELSPMVCUDATASK_HPP*/
