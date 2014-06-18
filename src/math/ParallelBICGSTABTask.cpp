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

#include "math/ParallelBICGSTABTask.hpp"
#include "math/Vector.hpp"
//#include "util/CSVExporter.hpp"
#include "core/BenchmarkTimer.hpp"
#include "core/Exception.hpp"
#include "math/SpMatrix.hpp"

namespace CGF{

#define TOL 1e-16

  template<int N, class T>
  ParallelBICGSTABTask<N, T>::ParallelBICGSTABTask(const int _n, 
                                                   Vector<T>* const _x, 
                                                   const SpMatrix<N, T>* const _mat, 
                                                   const Vector<T>* const _b) : 
    Task(_n),mat(_mat),x(_x), b(_b){
    /*Allocate shared resources*/

    r = new Vector<T>(mat->getHeight());
    r->clear();
    
    C = new Vector<T>(*r);
    C2 = new Vector<T>(*r);
    r2 = new Vector<T>(*r);
    p = new Vector<T>(*r);
    pp = new Vector<T>(*r);
    s = new Vector<T>(*r);
    sp = new Vector<T>(*r);
    t = new Vector<T>(*r);
    v = new Vector<T>(*r);
    
    scratch1 = new Vector<T>(*r);
    scratch2 = new Vector<T>(*r);

    reductions1 = new T[n_threads];
    reductions2 = new T[n_threads];

    vRange = 0;
    mRange = 0;

    n_blocks = new int[n_threads];

    timers = new BenchmarkTimer[n_threads];
    k = new int[n_threads];
  }

  template<int N, class T>
  ParallelBICGSTABTask<N, T>::~ParallelBICGSTABTask(){
    delete r;
    delete C;
    delete C2;
    delete r2;
    delete p;
    delete pp;
    delete s;
    delete sp;
    delete t;
    delete v;
    
    delete scratch1;
    delete scratch2;

    delete[] reductions1;
    delete[] reductions2;

    if(vRange){
      delete[] vRange;
    }
    if(mRange){
      delete[] mRange;
    }

    delete[] n_blocks;

    delete[] timers;
    delete[] k;
  }

#if 1
  template<int N, class T>
  void ParallelBICGSTABTask<N, T>::exportSummary(){
    /*for(int i=0;i<n_threads;i++){
      timers[i].start("device_device_copy");
      timers[i].stop("device_device_copy");

      exporter->setValue("thread_id", i);
      exporter->setValue("blocks_per_thread", n_blocks[i]);
      exporter->setValue("partial_dim", vRange[i].range);
      exporter->setValue("partial_time", timers[i].getAccumulativeUSec());
      exporter->setValue("n_iterations", k[i]);

      exporter->saveRow();
      }*/
  }
#endif

#if 0
  template<int N, class T>
  void ParallelBICGSTABTask<N, T>::configure(){


  }
#endif

  template<int N, class T>
  void ParallelBICGSTABTask<N, T>::execute(const Thread* caller){
    switch(subTask){
    case BICGSTABConfig:
      if(vRange == 0){
        vRange   = new VectorRange[n_threads];
        mRange   = new MatrixRange[n_threads];
	
        mat->computeBlockDistribution(mRange, vRange, n_blocks, n_threads);

#if 0	
        for(int i=0;i<n_threads;i++){
          message("mRange.start = %d, end = %d, range = %d", 
                  mRange[i].startRow, mRange[i].endRow, mRange[i].range);
	  
          message("vRange.start = %d, end = %d, range = %d", 
                  vRange[i].startBlock, vRange[i].endBlock, mRange[i].range);
        }
#endif
      }

      /*Create preconditioner*/
      for(int i=0;i<mat->getWidth();i++){
        (*C)[i] = (T)1.0/Sqrt((*mat)[i][i]);
        (*C2)[i] = Sqrt((*mat)[i][i]);
        cgfassert((*mat)[i][i] > 0);
      }
      return;
    case BICGSTABSolve:
      /*r = Ax*/
      spmv_partial(*r, *mat, *x, mRange[TID]);
      
      /*r = b - r*/
      Vector<T>::subp(*r, *b, *r, vRange[TID]);

      /*r2 = r*/
      Vector<T>::mfaddp(*r2, 0, *r, *r, vRange[TID]);      
      
      T w0, rho0, alpha;
      w0 = rho0 = alpha = 1;
      k[TID] = 1;
      T rho1, beta;
      rho1 = 1;
      
      while(k[TID] < 10000){
        /*res = |R|^2*/
        Vector<T>::mulp(*scratch1, *r, *r, vRange[TID]);
        T res = scratch1->sum(reductions1, caller, vRange);
	
        if(res < TOL){
          if(TID == 0){
            message("Success in %d iterations", k[TID]);
          }
          return;
        }
	
        /*rho0 = rho1*/
        /*rho1 = r2.r*/
        rho0 = rho1;
        Vector<T>::mulp(*scratch2, *r2, *r, vRange[TID]);
        rho1 = scratch2->sum(reductions2, caller, vRange);
	
        /*beta = (rho1/rho0)*(alpha/w0)*/
        beta = (rho1/rho0)*(alpha/w0);
	
        /*p = r + p -beta * w0 * v*/
        Vector<T>::mfaddp(*scratch1, -w0, *v, *p, vRange[TID]);
        Vector<T>::mulfp(*scratch1, *scratch1, beta, vRange[TID]);
        Vector<T>::addp(*p, *r, *scratch1, vRange[TID]);
	
        Vector<T>::mulp(*pp, *C, *p, vRange[TID]);
	
        /*pp must be completely computed*/
        caller->sync();
	
        spmv_partial(*v, *mat, *pp, mRange[TID]);      
	
        Vector<T>::mulp(*scratch1, *r2, *v, vRange[TID]);
	
        reductions1[TID] = scratch1->sump(vRange[TID]);
        T divider = scratch1->sum(reductions1, caller, vRange);
	
        alpha = rho1/divider;
	
        Vector<T>::mfaddp(*s, -alpha, *v, *r, vRange[TID]);
        Vector<T>::mulp(*sp, *C, *s, vRange[TID]);
	
        /*sp must be completely computed*/
        caller->sync();
	
        spmv_partial(*t, *mat, *sp, mRange[TID]);
	
        Vector<T>::mulp(*scratch1, *C2, *t, vRange[TID]);
        Vector<T>::mulp(*scratch2, *C2, *s, vRange[TID]);
	
        Vector<T>::mulp(*scratch1, *scratch1, *scratch2, vRange[TID]);
	
        w0 = scratch1->sum(reductions1, caller, vRange);
	
        Vector<T>::mulp(*scratch2, *C2, *t, vRange[TID]);
        Vector<T>::mulp(*scratch2, *scratch2, *scratch2, vRange[TID]);

        divider = scratch2->sum(reductions2, caller, vRange);
	
        w0/=divider;
	
        Vector<T>::mulfp(*scratch1, *pp, alpha, vRange[TID]);
        Vector<T>::mulfp(*scratch2, *sp, w0, vRange[TID]);
        Vector<T>::addp(*x, *x, *scratch1, vRange[TID]);
        Vector<T>::addp(*x, *x, *scratch2, vRange[TID]);
	
        Vector<T>::mfaddp(*r, -w0, *t, *s, vRange[TID]);
	
        k[TID]++;
        //caller->sync();
      }
      throw new SolutionNotFoundException(__LINE__, __FILE__);
      return;
    }
  }

  template class CGFAPI ParallelBICGSTABTask<1, float>;
  template class CGFAPI ParallelBICGSTABTask<2, float>;
  template class CGFAPI ParallelBICGSTABTask<4, float>;
  template class CGFAPI ParallelBICGSTABTask<8, float>;
  //template class CGFAPI ParallelBICGSTABTask<16, float>;

  template class CGFAPI ParallelBICGSTABTask<1, double>;
  template class CGFAPI ParallelBICGSTABTask<2, double>;
  template class CGFAPI ParallelBICGSTABTask<4, double>;
  template class CGFAPI ParallelBICGSTABTask<8, double>;
  //template class CGFAPI ParallelBICGSTABTask<16, double>;
}
