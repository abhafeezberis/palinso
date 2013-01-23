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

#include "math/ParallelCGTask.hpp"
#include "math/Vector.hpp"

namespace CGF{

  template<int N, class T>
  ParallelCGTask<N, T>::ParallelCGTask(const uint _n, 
				       Vector<T>* const _x, 
				       const SpMatrix<N, T>* const _mat, 
				       const Vector<T>* const _b) : 
    Task(_n),mat(_mat),x(_x), b(_b), tolerance(1e-6), maxIterations(100000){
    /*Allocate shared resources*/

    r = new Vector<T>(mat->getHeight());
    r->clear();

    scratch1 = new Vector<T>(*r);
    scratch2 = new Vector<T>(*r);
    scratch3 = new Vector<T>(*r);
    scratch4 = new Vector<T>(*r);
    scratch5 = new Vector<T>(*r);
    scratch6 = new Vector<T>(*r);

    u = new Vector<T>(*r);
    v = new Vector<T>(*r);
    w = new Vector<T>(*r);
    C = new Vector<T>(*r);

    /*Initialize (initial) preconditionervector and result vector*/
    Vector<T>::adds(*C, *x, 1); //C = 1

    reductions1 = new T[n_threads];
    reductions2 = new T[n_threads];
    reductions3 = new T[n_threads];

    vRange   = new VectorRange[n_threads];
    mRange   = new MatrixRange[n_threads];

    n_blocks = new uint[n_threads];

    k = new uint[n_threads];
  }

  template<int N, class T>
  ParallelCGTask<N, T>::~ParallelCGTask(){
    delete scratch1;
    delete scratch2;
    delete scratch3;
    delete scratch4;
    delete scratch5;
    delete scratch6;
    delete r;
    delete u;
    delete v;
    delete w;
    delete C;
    delete [] reductions1;
    delete [] reductions2;
    delete [] reductions3;
    delete [] vRange;
    delete [] mRange;
    delete [] n_blocks;
    delete [] k;
  }

  template<int N, class T>
  void ParallelCGTask<N, T>::setb(Vector<T>* vec){
    cgfassert(vec->getSize() == b->getSize());
    b = vec;    
  }

  template<int N, class T>
  void ParallelCGTask<N, T>::setx(Vector<T>* vec){
    cgfassert(vec->getSize() == x->getSize());
    x = vec;    
  }

  template<int N, class T>
  void ParallelCGTask<N, T>::setMatrix(SpMatrix<N, T>* m){
    cgfassert(m->getWidth() == mat->getWidth());
    cgfassert(m->getHeight() == mat->getHeight());
    mat = m;
  }

  template<int N, class T>
  void ParallelCGTask<N, T>::execute(const Thread* caller){
     /*r = b - Ax*/
    spmv_partial(*r, *mat, *x, mRange[TID]);

    Vector<T>::subp(*r, *b, *r, vRange[TID]);
    
    /*w = C * r*/
    /*v = C * w*/
    Vector<T>::mulp(*w, *C, *r, vRange[TID]);
    Vector<T>::mulp(*v, *C, *w, vRange[TID]);
    
    /*alpha = w*w */
    Vector<T>::mulp(*scratch1, *w, *w, vRange[TID]);    

    reductions2[TID] = scratch1->sump(vRange[TID]);
    caller->sync();

    T lalpha = 0;
    for(uint i=0;i<n_threads;i++){
      lalpha += reductions2[i];  
    }

    caller->sync();

    /*res = v * v*/
    Vector<T>::mulp(*scratch1, *v, *v, vRange[TID]);

    reductions2[TID] = scratch1->sump(vRange[TID]);

    caller->sync();

    double lresidual = 0;
    for(uint i=0;i<n_threads;i++){
      lresidual += reductions2[i];  
    }

    lresidual = sqrt(lresidual);

    k[TID] = 0;

    while(k[TID]<maxIterations){
    //while(k[TID] < 103){
      /*Compute |V| */
      Vector<T>::mulp(*scratch1, *v, *v, vRange[TID]);
      reductions1[TID] = scratch1->sump(vRange[TID]);
      caller->sync();

      T lresidual = 0;
      for(uint i=0;i<n_threads;i++){
	lresidual += reductions1[i];
      }

      //lresidual = sqrt(lresidual);
      if(sqrt(fabs(lresidual))<tolerance){
	if(TID == 0){
	  message("Succesfull in %d iterations, %10.10e, %10.10e", k[TID],
		  lresidual, sqrt(fabs(lresidual)));
	}
	return;
      }

      /*u = Av*/
      spmv_partial(*u, *mat, *v, mRange[TID]);

      /*divider = v*u */
      Vector<T>::mulp(*scratch3, *v, *u, vRange[TID]);
      reductions2[TID] = scratch3->sump(vRange[TID]);
      caller->sync();

      T ldivider = 0;
      for(uint i=0;i<n_threads;i++){
	ldivider += reductions2[i];
      }

      /*t = alpha / (v*u) */
      T t = lalpha/ldivider;

      /*x = x + tv*/
      /*r = r - tu*/
      /*w = C * r*/
      Vector<T>::mfaddp(*x,  t, *v, *x, vRange[TID]);
      Vector<T>::mfaddp(*r, -t, *u, *r, vRange[TID]);
      Vector<T>::mulp  (*w, *C, *r, vRange[TID]);

      /*beta = w*w */
      Vector<T>::mulp  (*scratch5, *w, *w, vRange[TID]);
      reductions3[TID] = scratch5->sump(vRange[TID]);
      caller->sync();

      double lbeta = 0;
      double ls = 0;
      for(uint i=0;i<n_threads;i++){
	lbeta += reductions3[i];
      }
      
#if 0
      if(k[TID]%100 == 0){
	message("beta[%d] = %10.10e", TID, lbeta);
	message("residual[%d] = %10.10e", TID, lresidual);
      }
#endif
      
      ls = lbeta/lalpha;
      lalpha = lbeta;

#if 1
      if(lbeta < tolerance){
#if 1
	Vector<T>::mulp(*scratch1, *r, *r, vRange[TID]);
	reductions2[TID] = scratch1->sump(vRange[TID]);
	caller->sync();
	
	double rl = 0;
	for(uint i=0;i<n_threads;i++){
	  rl += reductions2[i];
	}

	//rl = sqrt(rl);
	if(rl< tolerance){
#endif
	  if(TID == 0){
	    message("Succesfull in %d iterations, %10.10e, %10.10e", k[TID],
		    rl, sqrt(fabs(rl)));
	  }
	  //message("Succesfull in %d iterations", k[TID]);
	  return;
	}
      }

#endif


      /*v = C*w + sv */
      Vector<T>::mulp(*scratch3, *C, *w, vRange[TID]);
      Vector<T>::mfaddp(*v, ls, *v, *scratch3, vRange[TID]);
      k[TID]++;
    }
    if(TID == 0){
      message("Unsuccesfull");
    }    
  }
  
  template<int N, class T>
  void ParallelCGTask<N, T>::computePreconditioner(){
    /*Create preconditioner*/
    for(uint i=0;i<mat->getHeight();i++){
      (*x)[i] = 1;
      (*r)[i] = 0;
#if 0
      (*C)[i] = 1.0/((*mat)[i][i]);//1.0/sqrt((*mat)[i][i]);
      //cgfassert((*mat)[i][i] > 0);
#else
      (*C)[i] = 1.0/sqrt((*mat)[i][i]);
      cgfassert((*mat)[i][i] > 0);
#endif
    }    
  }

  template<int N, class T>
  void ParallelCGTask<N, T>::computeDistribution(){
    mat->computeBlockDistribution(mRange, vRange, n_blocks, n_threads);

#if 0    
    for(uint i=0;i<n_threads;i++){
      message("mRange.start = %d, end = %d, range = %d", 
	      mRange[i].startRow, mRange[i].endRow, mRange[i].range);
    }
#endif
  }

  template class CGFAPI ParallelCGTask<1, float>;
  template class CGFAPI ParallelCGTask<2, float>;
  template class CGFAPI ParallelCGTask<4, float>;
  template class CGFAPI ParallelCGTask<8, float>;
  //template class CGFAPI ParallelCGTask<16, float>;

  template class CGFAPI ParallelCGTask<1, double>;
  template class CGFAPI ParallelCGTask<2, double>;
  template class CGFAPI ParallelCGTask<4, double>;
  template class CGFAPI ParallelCGTask<8, double>;
  //template class CGFAPI ParallelCGTask<16, double>;
}
