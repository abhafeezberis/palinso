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

#ifndef PARALLELSPMVTASK_HPP
#define PARALLELSPMVTASK_HPP

#include "core/Task.hpp"
#include "core/ThreadPool.hpp"
#include "core/BenchmarkTimer.hpp"
#include "math/Math.hpp"

namespace CGF{
  template<class T>
  class CGFAPI Vector;

  template<int N, class T>
  class CGFAPI SpMatrix;
  
  template<int N, class T>
  class CGFAPI ParallelSPMVTask : public Task{
  public:
    ParallelSPMVTask(const ThreadPool* pool, const SpMatrix<N, T>* const mat,
		     Vector<T>* const _r = 0, const Vector<T>* const _x = 0);
    virtual ~ParallelSPMVTask();
    void setVectors(Vector<T>* const _r, const Vector<T>* const _x){
      x = _x;
      r = _r;
    }

    virtual void execute(const Thread* caller);
    //virtual void exportSummary();
  protected:
    const SpMatrix<N, T>* mat;
    const Vector<T>* x;
    Vector<T>* r;
    
    MatrixRange* mRange;
    VectorRange* vRange;
    uint* n_blocks;
    BenchmarkTimer* timers;
  };
}

#endif/*PARALLELSPMVTASK_HPP*/
