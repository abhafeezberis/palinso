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

#include "math/ParallelSPMVTask.hpp"
#include "math/Vector.hpp"
//#include "util/CSVExporter.hpp"

namespace CGF{

  template<int N, class T>
  ParallelSPMVTask<N, T>::ParallelSPMVTask(const ThreadPool* pool,
                                           const SpMatrix<N, T>* const _mat,
                                           Vector<T>* const _r, 
                                           const Vector<T>* const _x):
    Task(pool->getSize()),mat(_mat),x(_x), r(_r){
    
    mRange   = new MatrixRange[n_threads];
    vRange   = new VectorRange[n_threads];
    n_blocks = new int[n_threads];

    mat->computeBlockDistribution(mRange, vRange, n_blocks,  n_threads);

    timers = new BenchmarkTimer[n_threads];
  }

  template<int N, class T>
  ParallelSPMVTask<N, T>::~ParallelSPMVTask(){
    delete [] timers;
    delete [] mRange;
    delete [] vRange;
    delete [] n_blocks;
  }

  template<int N, class T>
  void ParallelSPMVTask<N, T>::execute(const Thread* caller){
    cgfassert(r != 0);
    cgfassert(x != 0);

    timers[TID].start("spmv_partial");
    for(uint i=0;i<1000;i++){
      spmv_partial(*r, *mat, *x, mRange[TID]);
    }
    timers[TID].stop("spmv_partial");
  }


#if 0
  template<int N, class T>
  void ParallelSPMVTask<N, T>::exportSummary(){
    for(uint i=0;i<n_threads;i++){
      exporter->setValue("thread_id", i);
      exporter->setValue("blocks_per_thread", n_blocks[i]);
      exporter->setValue("partial_dim", vRange[i].range);
      exporter->setValue("average_time", timers[i].getAverageTimeUSec("spmv_partial"));

      exporter->saveRow();
    }
  }
#endif

  template class CGFAPI ParallelSPMVTask<1, float>;
  template class CGFAPI ParallelSPMVTask<2, float>;
  template class CGFAPI ParallelSPMVTask<4, float>;
  template class CGFAPI ParallelSPMVTask<8, float>;
  //template class CGFAPI ParallelSPMVTask<16, float>;

  template class CGFAPI ParallelSPMVTask<1, double>;
  template class CGFAPI ParallelSPMVTask<2, double>;
  template class CGFAPI ParallelSPMVTask<4, double>;
  template class CGFAPI ParallelSPMVTask<8, double>;
  //template class CGFAPI ParallelSPMVTask<16, double>;
}
