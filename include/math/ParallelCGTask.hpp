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

#ifndef PARALLELCGTASK_HPP
#define PARALLELCGTASK_HPP

#include "core/Task.hpp"
#include "core/Thread.hpp"
#include "math/Math.hpp"

namespace CGF{
  class BenchmarkTimer;
  template<class T>
  class CGFAPI Vector;

  template<int N, class T>
  class CGFAPI SpMatrix;

  template<int N, class T>
  class CGFAPI ParallelCGTask : public Task{
  public:
    ParallelCGTask(const int n_threads, Vector<T>* const x,
                   const SpMatrix<N, T>* const mat,
                   const Vector<T>* const b);
    virtual ~ParallelCGTask();

    virtual void execute(const Thread* caller);
    virtual void setb(Vector<T>* vec);
    virtual void setx(Vector<T>* vec);
    virtual void setMatrix(SpMatrix<N, T>* m);
    void computePreconditioner();
    void computeDistribution();

    void setTolerance(T tol){
      tolerance = tol;
    }

    void setMaxIterations(int steps){
      maxIterations = steps;
    }

  protected:
    Vector<T>* r;
    const SpMatrix<N, T>* mat;
    Vector<T>* scratch1;
    Vector<T>* scratch2;
    Vector<T>* scratch3;
    Vector<T>* scratch4;
    Vector<T>* scratch5;
    Vector<T>* scratch6;
    Vector<T>* x;
    Vector<T>* u;
    Vector<T>* v;
    Vector<T>* w;
    Vector<T>* C;
    const Vector<T>* b;

    T* reductions1;
    T* reductions2;
    T* reductions3;

    //double alpha, beta;
    //double divider;
    //double residual;
    //double s;

    VectorRange* vRange;
    MatrixRange* mRange;
    int* n_blocks;
    BenchmarkTimer* timers;
    int* k;

    T tolerance;
    int maxIterations;
    T bnorm;
  };
}

#endif/*PARALLELCGTASK_HPP*/
