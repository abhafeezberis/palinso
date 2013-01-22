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

#ifndef PARALLELCGCUDATASK_HPP
#define PARALLELCGCUDATASK_HPP
#ifndef NO_CUDA
#include "math/Math.hpp"
#include <cuda_runtime.h>
#include "core/Task.hpp"
#include "math/CUDASpmv.hpp"

namespace CGF{
  
  template<int N, class T>
  class CGFAPI CSpMatrix;
  template<class T>
  class CGFAPI CVector;

  class CGFAPI BenchmarkTimer;
  template<class T>
  class CGFAPI Vector;

  template<int N, class T>
  class CGFAPI SpMatrix;

  template<int N, class T>
  class CGFAPI ParallelCGCudaTask : public Task{
  public:
    ParallelCGCudaTask(const ThreadPool* pool, Vector<T>* const x, 
		       const SpMatrix<N, T>* const mat, 
		       const Vector<T>* const b, int n_thr=256, 
		       TextureOperation tex = TexVector);
    virtual ~ParallelCGCudaTask();

    virtual void execute(const Thread* caller);
    virtual void allocate(const Thread* caller);
    virtual void deallocate(const Thread* caller);
    virtual void copyResult(const Thread* caller);
    virtual void updateBlocks(const Thread* caller);
    virtual void solveSystem(const Thread* caller);
    virtual void setb(Vector<T>* vec);
    virtual void setx(Vector<T>* vec);
    virtual void setMatrix(SpMatrix<N, T>* m);
    void computePreconditioner();
    void computeDistribution();
    void prepareMatrix();
  protected:
    const ThreadPool* pool;
    Vector<T>* x;
    CVector<T>* cx;
    const SpMatrix<N, T>* mat;
    const Vector<T>* b;
    CVector<T>* cb;
    CSpMatrix<N, T>* cmat;
    CVector<T>* cv;
    CVector<T>* cres;
    CVector<T>* cu;
    CVector<T>* cw;
    CVector<T>* cC;
    CVector<T>* ctmp1;
    CVector<T>* ctmp2;
    CVector<T>* ctmp3;
    CVector<T>* crtmp1;
    CVector<T>* crtmp2;
    CVector<T>* crtmp3;
    CVector<T>* cfull_vec;

    Vector<T>* r;

    Vector<T>* C;

    BenchmarkTimer* timers;
    uint* k;

    int n_cuda_threads;
    TextureOperation texture;
  };
};

#endif/*NO_CUDA*/
#endif/*PARALLELCGCUDATASK_HPP*/
