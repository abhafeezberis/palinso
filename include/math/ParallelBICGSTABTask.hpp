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

#ifndef PARALLELBICGSTABTASK_HPP
#define PARALLELBICGSTABTASK_HPP

#include "core/Task.hpp"
#include "core/Thread.hpp"
#include "math/Math.hpp"

namespace CGF{
  class BenchmarkTimer;
  template<class T>
  class CGFAPI Vector;

  template<int N, class T>
  class CGFAPI SpMatrix;

  enum BICGSTABSubTask{BICGSTABConfig, BICGSTABSolve};
  
  template<int N, class T>
  class CGFAPI ParallelBICGSTABTask : public Task{
  public:
    ParallelBICGSTABTask(const uint n_threads, Vector<T>* const x, 
			 const SpMatrix<N, T>* const mat, 
			 const Vector<T>* const b);

    virtual ~ParallelBICGSTABTask();

    virtual void execute(const Thread* caller);
    virtual void exportSummary();
  protected:
    const SpMatrix<N, T>* mat;
    Vector<T>* x;
    const Vector<T>* b;

    Vector<T>* r;
    Vector<T>* C;
    Vector<T>* C2;
    Vector<T>* r2;
    Vector<T>* p;
    Vector<T>* pp;
    Vector<T>* s;
    Vector<T>* sp;
    Vector<T>* t;
    Vector<T>* scratch1;
    Vector<T>* scratch2;
    Vector<T>* v;

    T* reductions1;
    T* reductions2;
    
    VectorRange* vRange;
    MatrixRange* mRange;
    uint* n_blocks;
    BenchmarkTimer* timers;
    uint* k;
  };
}

#endif/*PARALLELBICGTASK_HPP*/
