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

#ifndef NO_CUDA
#include <cuda_runtime.h>
#include "math/ParallelCGCudaTask.hpp"
#include "core/Thread.hpp"
#include "util/cuda_util.hpp"
#include "math/SpMatrix.hpp"
#include "math/Vector.hpp"
#include "math/CUDACGOp.hpp"
#include "math/CSpMatrix.hpp"
#include "core/Exception.hpp"
#include "math/CSpMatrix.hpp"
#include "math/CVector.hpp"
#include "core/ThreadPool.hpp"

#define MAPPED

namespace CGF{

#define TOL 1e-8
  template<int N, class T>
  ParallelCGCudaTask<N, T>::ParallelCGCudaTask(const ThreadPool* _pool, 
					       Vector<T>* const _x, 
					       const SpMatrix<N, T>* const _mat, 
					       const Vector<T>* const _b,
					       int n_thr, TextureOperation tex)
    : Task(_pool->getSize()), pool(_pool), x(_x), mat(_mat), b(_b), 
      n_cuda_threads(n_thr), texture(tex){

    //init_cuda_host_thread();

    cmat = 0;

    r = new Vector<T>(*x);
    C = new Vector<T>(*x);
    /*Clear x vector*/
    for(uint i=0;i<x->getSize();i++){
      (*r)[i] = 0;
      (*C)[i] = 1;
    }

    cx            = new CVector<T>(x, pool);
    cb            = new CVector<T>(b, pool);
    cu            = new CVector<T>(x, pool);
    cv            = new CVector<T>(x, pool);
    cw            = new CVector<T>(x, pool);
    cC            = new CVector<T>(C, pool);
    cres          = new CVector<T>(r, pool);
    ctmp1         = new CVector<T>(x, pool);
    ctmp2         = new CVector<T>(x, pool);
    ctmp3         = new CVector<T>(x, pool);
    uint red_size = x->getSize();
    crtmp1        = new CVector<T>(red_size/*minTempReductionSize(x->getSize(), 
					     n_threads)*/, pool);
    crtmp2        = new CVector<T>(red_size/*minTempReductionSize(x->getSize(), 
					     n_threads)*/, pool);
    crtmp3        = new CVector<T>(red_size/*minTempReductionSize(x->getSize(), 
					     n_threads)*/, pool);
    cfull_vec     = new CVector<T>(x, pool, true);

    ctmp1->setReductionBuffer(crtmp1);
    ctmp2->setReductionBuffer(crtmp2);
    ctmp3->setReductionBuffer(crtmp3);
    
    k             = new uint[n_threads];
  }

  template<int N, class T>
  ParallelCGCudaTask<N, T>::~ParallelCGCudaTask(){
    delete r;
    delete C;

    delete cx;
    delete cb;
    delete cu;
    delete cv;
    delete cw;
    delete cres;
    delete cC;
    delete ctmp1;
    delete ctmp2;
    delete ctmp3;
    delete crtmp1;
    delete crtmp2;
    delete crtmp3;
    delete cfull_vec;

    delete cmat;
    
    delete[] k;
  }

  template<int N, class T>
  void ParallelCGCudaTask<N, T>::setb(Vector<T>* vec){
    cgfassert(vec->getSize() == b->getSize());
    b = vec;
    delete cb;
    cb = new CVector<T>(b, pool);
  }

  template<int N, class T>
  void ParallelCGCudaTask<N, T>::setx(Vector<T>* vec){
    cgfassert(vec->getSize() == x->getSize());
    x = vec;
    delete cx;
    cx = new CVector<T>(x, pool);    
  }

  template<int N, class T>
  void ParallelCGCudaTask<N, T>::setMatrix(SpMatrix<N, T>* m){
    cgfassert(m->getWidth() == mat->getWidth());
    cgfassert(m->getHeight() == mat->getHeight());
    mat = m;
  }

  template<int N, class T>
  void ParallelCGCudaTask<N, T>::execute(const Thread* caller){
    /*Zero range, skip allocation/computation*/
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
  void ParallelCGCudaTask<N, T>::allocate(const Thread* caller){
    /*Initialize a cuda context by selecting the desired device and
      allocate memory on device*/
    caller->setCuda();

    cmat->     allocateDevice(caller);
 
    cb->       allocateDevice(caller);
    cx->       allocateDevice(caller);
    cu->       allocateDevice(caller);
    cv->       allocateDevice(caller);
    cw->       allocateDevice(caller);
    cC->       allocateDevice(caller);
    ctmp1->    allocateDevice(caller);
    ctmp2->    allocateDevice(caller);
    ctmp3->    allocateDevice(caller);
    crtmp1->   allocateDevice(caller);
    crtmp2->   allocateDevice(caller);
    crtmp3->   allocateDevice(caller);
    cres->     allocateDevice(caller);
    cfull_vec->allocateDevice(caller);

    //message("crtmp1[%d] size = %d", TID, crtmp1->getVRange(TID)->range );
  }

  template<int N, class T>
  void ParallelCGCudaTask<N, T>::deallocate(const Thread* caller){
    /*De-allocate all used device resources*/
    cmat->     deallocateDevice(caller);

    cb->       deallocateDevice(caller);
    cx->       deallocateDevice(caller);
    cu->       deallocateDevice(caller);
    cv->       deallocateDevice(caller);
    cw->       deallocateDevice(caller);
    cC->       deallocateDevice(caller);
    ctmp1->    deallocateDevice(caller);
    ctmp2->    deallocateDevice(caller);
    ctmp3->    deallocateDevice(caller);
    crtmp1->   deallocateDevice(caller);
    crtmp2->   deallocateDevice(caller);
    crtmp3->   deallocateDevice(caller);
    cres->     deallocateDevice(caller);
    cfull_vec->deallocateDevice(caller);

    //exit_cuda_thread();
  }

  template<int N, class T>
  void ParallelCGCudaTask<N, T>::copyResult(const Thread* caller){
    cudaSafeCall(cudaMemcpy(x->data + cx->getVRange(TID)->startBlock, 
			    cx->getData(TID), 
			    sizeof(T)*cx->getVRange(TID)->range, 
			    cudaMemcpyDeviceToHost));
  }

  template<int N, class T>
  void ParallelCGCudaTask<N, T>::computePreconditioner(){
    for(uint i=0;i<x->getSize();i++){
      /*In the first CG kernel, the preconditioner is computed using
	C, which must be the diagonal of the matrix.*/
      (*C)[i] = (*mat)[i][i];
    }
  }

  template<int N, class T>
  void ParallelCGCudaTask<N, T>::computeDistribution(){
    cmat->computeBlockDistribution();
  }

  template<int N, class T>
  void ParallelCGCudaTask<N, T>::prepareMatrix(){
    if(cmat){
      delete cmat;
    }
    
    /*Allocate memory for device pointers*/
    cmat = new CSpMatrix<N, T>(mat, n_cuda_threads, texture, pool);
    cmat->computeBlockDistribution();

    /*Copy ranges*/
    cx->   copyRanges(cmat);
    cb->   copyRanges(cmat);
    cu->   copyRanges(cmat);
    cv->   copyRanges(cmat);
    cw->   copyRanges(cmat);
    cC->   copyRanges(cmat);
    cres-> copyRanges(cmat);
    ctmp1->copyRanges(cmat);
    ctmp2->copyRanges(cmat);
    ctmp3->copyRanges(cmat);

    cfull_vec->copyRanges(cmat);
  }

  template<int N, class T>
  void ParallelCGCudaTask<N, T>::updateBlocks(const Thread* caller){
  }

  template<int N, class T>
  void ParallelCGCudaTask<N, T>::solveSystem(const Thread* caller){
    double lalpha = 0;
    double lbeta = 0;
    double lresidual = 0;
    double ldivider = 0;
    double lr = 0;
    double ls = 0;
    
    cmat->preSpmv(caller);
    //bindTexture(cfull_vec->getData(TID), cmat->getWidth());
    cfull_vec->bindToTexture(caller);

    /*r = Ax*/
    cmat->spmv(cres, cfull_vec, caller);
    //cres->print(caller);

    /*r = b - r*/
    /*w = C * r*/
    /*v = C * w*/
    /*tmp1 = w * w*/
    /*tmp2 = v * v*/
    /*Copy part of v to mapped memory and full_vec*/
    parallel_cg_step_1<T>(cb, cC, cres, cv, cw,ctmp1, ctmp2, cfull_vec, 
			  cfull_vec->getMappedData(TID), n_threads, 
			  caller);
    //cx->print(caller);
      
    lalpha = ctmp1->sum(caller);

    //message("alpha = %10.10e", lalpha);
    //getchar();

    lresidual = sqrt(fabs(ctmp2->sum(caller)));
    
    k[TID] = 0;
    
    while(k[TID]<mat->getHeight()*1000){
      /*Copy v to host, preparing for multiplication*/
      if(lresidual<TOL){
	if(TID == 0){
	  message("Succesfull in %d iterations", k[TID]);
	}
	return;
      }
#if 0
      /*Only if memory mapping is not used*/
      cfull_vec->gather(caller);
#endif

      /*Distribute changes to other devices*/
      cfull_vec->scatter(caller);
 
      /*After scatter is performed, full_vec for that device is up to date*/

      /*Scatter is asynchronous, but the spmv kernel will only start
	if the copy has been finished.*/
      //caller->sync();

      /*u = Av*/
      cmat->spmv(cu, cfull_vec, caller);

      //cfull_vec->print(caller);
      
      //getchar();


      /*tmp1 = u*v*/
      parallel_cg_step_3<T>(cv, cu, ctmp1, caller);
      
      ldivider = ctmp1->sum(caller);

      /*t    = alpha / (v*u) */
      T t = lalpha/ldivider;
      /*x    = x + tv*/
      /*r    = r - tu*/
      /*w    = C * r*/
      /*tmp2 = w * w*/

      parallel_cg_step_4<T>(cv, cu, cC, t, cw, cx, cres, ctmp2, caller);
      
      lbeta = ctmp2->sum(caller);

      ls = (double)lbeta/(double)lalpha;
      lalpha = lbeta;
#if 1
      if(lbeta < TOL){
	parallel_cg_step_2<T>(cres, ctmp1, caller);
	
	lr = ctmp1->sum(caller);
	
	if(lr< TOL){
	  if(TID == 0){
	    message("Succesfull in %d iterations", k[TID]);
	  }
	  return;
	}
      }
#endif

      /*v    = C*w + sv */
      /*tmp3 = v * v*/
      /*Copy v to full_vec and mapped memory*/
      parallel_cg_step_5<T>(cw, cC, ls, cv, ctmp3, cfull_vec, 
			    cfull_vec->getMappedData(TID), 
			    n_threads, caller);
       
      lresidual = sqrt(fabs(ctmp3->sum(caller)));

      k[TID]++;
    }
    if(TID == 0){
      message("Not succesfull in %d iterations", k[TID]);
    }
  }


  template class ParallelCGCudaTask<1, float>;
  template class ParallelCGCudaTask<2, float>;
  template class ParallelCGCudaTask<4, float>;
  template class ParallelCGCudaTask<8, float>;
  //template class ParallelCGCudaTask<16, float>;

  template class ParallelCGCudaTask<1, double>;
  template class ParallelCGCudaTask<2, double>;
  template class ParallelCGCudaTask<4, double>;
  template class ParallelCGCudaTask<8, double>;
  //template class ParallelCGCudaTask<16, double>;
};
#endif/*NO_CUDA*/
