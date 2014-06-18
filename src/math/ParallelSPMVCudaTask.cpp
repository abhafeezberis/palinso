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

#ifdef CUDA
#include "math/ParallelSPMVCudaTask.hpp"
#include "core/Thread.hpp"
#include "util/cuda_util.hpp"
#include "math/SpMatrix.hpp"
#include "math/Vector.hpp"
#include "core/Exception.hpp"
#include "math/CSpMatrix.hpp"
//#include "util/CSVExporter.hpp"
#include "core/ThreadPool.hpp"
#include <stdio.h>
#define MAPPED

namespace CGF{

#define TOL 1e-8
  template<int N, class T>
  ParallelSPMVCudaTask<N, T>::ParallelSPMVCudaTask(const ThreadPool* pool,
                                                   Vector<T>* const _x,
                                                   const SpMatrix<N, T>* const _mat,
                                                   const Vector<T>* const _b,
                                                   int n_thr,
                                                   TextureOperation tex,
                                                   int d_offset)
    : Task(pool->getSize()), x(_x), mat(_mat), b(_b), n_cuda_threads(n_thr), texture(tex), device_offset(d_offset){

    //init_cuda_host_thread();

    cmat = new CSpMatrix<N, T>(_mat, n_cuda_threads, texture, pool);
    cmat->computeBlockDistribution();

    cx = new CVector<T>(x, pool, true);
    cb = new CVector<T>(b, pool, true);

    tmp = new CVector<T>(b, pool);

    cx->copyRanges(cmat);
    cb->copyRanges(cmat);
    tmp->copyRanges(cmat);

    cb->setReductionBuffer(tmp);

    timers = new BenchmarkTimer[n_threads];
  }

  template<int N, class T>
  ParallelSPMVCudaTask<N, T>::~ParallelSPMVCudaTask(){
    delete   cmat;
    delete   cx;
    delete   cb;
    delete   tmp;
    delete[] timers;
  }

#if 0
  template<int N, class T>
  void ParallelSPMVCudaTask<N, T>::exportSummary(){
    for(int i=0;i<n_threads;i++){
      exporter->setValue("thread_id", i);
      exporter->setValue("blocks_per_thread", cmat->getNBlocks()[i]);
      exporter->setValue("partial_dim", cmat->getVRange(i)->range);
      exporter->setValue("average_time", timers[i].getAverageTimeUSec("cuda_spmv"));

      exporter->saveRow();
    }
  }
#endif

  template<int N, class T>
  void ParallelSPMVCudaTask<N, T>::execute(const Thread* caller){
    /*If this task is invalidated due to exceptions, stop execution of
      this task*/
    if(!valid)
      return;

    /*Zero range, skip allocation / computation*/

    if(cmat->getMRange(TID)->range == 0)
      return;

    switch(subTask){
    case Allocate:
#if 1
      try{
        /*Try to allocate all the memory needed. If this fails,
          deallocate and throw an exception*/
        allocate(caller);
      }catch(CUDAException& e){
        std::cerr << e.getError();
        valid = false;
        deallocate(caller);
        throw;
      }catch(Exception& e){
        std::cerr << e.getError();
        throw;
      }
#else
      allocate(caller);
#endif
      break;
    case Deallocate:
      deallocate(caller);
      break;
    case CopyResult:
      copyResult(caller);
      break;
    case UpdateBlocks:
      updateBlocks(caller);
      break;
    case SolveSystem:
      solveSystem(caller);
      break;
    }
  }

  template<int N, class T>
  void ParallelSPMVCudaTask<N, T>::copyResult(const Thread* caller){
#if 1
    message("%d", x->getSize());
    cudaSafeCall(cudaMemcpy(x->data + cx->getVRange(TID)->startBlock,
                            cx->getData(TID),//d_res[TID],
                            sizeof(T)*(uint)x->getSize(),
                            cudaMemcpyDeviceToHost));
#endif
  }

  template<int N, class T>
  void ParallelSPMVCudaTask<N, T>::updateBlocks(const Thread* caller){

  }

  template<int N, class T>
  void ParallelSPMVCudaTask<N, T>::allocate(const Thread* caller){
    init_cuda_thread(TID + device_offset);
    //init_cuda_thread(0);

    /*Allocate data*/
    cmat->allocateDevice(caller);
    cx->allocateDevice(caller);
    cb->allocateDevice(caller);
    tmp->allocateDevice(caller);
  }

  template<int N, class T>
  void ParallelSPMVCudaTask<N, T>::deallocate(const Thread* caller){
    cmat->deallocateDevice(caller);
    cx->deallocateDevice(caller);
    cb->deallocateDevice(caller);

    tmp->deallocateDevice(caller);

    exit_cuda_thread();
  }

  template<int N, class T>
  void ParallelSPMVCudaTask<N, T>::solveSystem(const Thread* caller){
    cudaSafeCall(cudaThreadSynchronize());
    timers[TID].start("cuda_spmv");

    cmat->preSpmv(caller);
    cb->bindToTexture(caller);

    int nn = 1000;
    for(int i=0;i<nn;i++){
      cmat->spmv(cx, cb, caller);
    }

    cudaThreadSynchronize();
    timers[TID].stop("cuda_spmv");

    //cx->print(caller);


    printf("Average elapsed time = %lu usec, after %d SpMVs\n", timers[TID].getAverageTimeUSec("cuda_spmv")/nn, nn);

    /*cx->gather(caller);
      caller->sync();
      cx->scatter(caller);

      cx->print(caller);
      getchar();
    */
  }

  template class ParallelSPMVCudaTask<1, float>;
  template class ParallelSPMVCudaTask<2, float>;
  template class ParallelSPMVCudaTask<4, float>;
  template class ParallelSPMVCudaTask<8, float>;
  //template class ParallelSPMVCudaTask<16, float>;

  template class ParallelSPMVCudaTask<1, double>;
  template class ParallelSPMVCudaTask<2, double>;
  template class ParallelSPMVCudaTask<4, double>;
  template class ParallelSPMVCudaTask<8, double>;
  //template class ParallelSPMVCudaTask<16, double>;
}

#endif/*CUDA*/
