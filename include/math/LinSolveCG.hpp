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

#ifndef LINSOLVECG_HPP
#define LINSOLVECG_HPP

#include "math/LinSolve.hpp"
#include "core/Exception.hpp"
#include "math/ParallelCGTask.hpp"
#include "math/ParallelCGCudaTask.hpp"
#include "core/ThreadPool.hpp"

namespace CGF{
  template<int N, class T>
  class LinSolveCG : public LinSolve<N, T>{
  public:
    LinSolveCG(uint d):LinSolve<N, T>(d){
      r = new Vector<T>(this->dim);
      C = new Vector<T>(this->dim);
      scratch1 = new Vector<T>(this->dim);
      scratch2 = new Vector<T>(this->dim);
      scratch1->enableReduction();
      scratch2->enableReduction();
      w = new Vector<T>(this->dim);
      v = new Vector<T>(this->dim);
      u = new Vector<T>(this->dim);

      Vector<T>::mulf(*r, *r, 0); //r = 0
    }

    virtual ~LinSolveCG(){
      delete r;
      delete C;
      delete scratch1;
      delete scratch2;
      delete w;
      delete v;
      delete u;
    }

    virtual void preSolve(){
      this->mat->finalize();
    }

    virtual void solve(uint steps = 100000, T tolerance = 1e-6){
      cgfassert(this->mat->getWidth() == this->b->getSize());
      cgfassert(this->mat->getWidth() == this->mat->getHeight());
      cgfassert(this->mat->getWidth() == this->x->getSize());

      for(uint i=0;i<this->mat->getHeight();i++){
#if 1
	//(*this->x)[i] = 0;
	if((*this->mat)[i][i] > 0)
	  (*C)[i] = 1.0/sqrt((*this->mat)[i][i]);
	else
	  message("%d, %d", i, C->getSize());
	//(*C)[i] = 1;
	if((*this->mat)[i][i] <= 0){
	  message("[%d, %d] = %f", i, i, (*this->mat)[i][i]);
	}
	cgfassert((*this->mat)[i][i] > 0);
#else
	(*C)[i] = 1;
#endif
      }

      //std::cout << *this->C;

      /*r = Ax*/
      spmv(*r, *(this->mat), *(this->x));
      
      //*r = *b - *r;
      Vector<T>::sub(*r, *(this->b), *r);
      
      /*w = C * r*/
      Vector<T>::mul(*w, *C, *r);
      
      /*v = C * w*/
      Vector<T>::mul(*v, *C, *w);
      
      /*s1 = w * w*/
      Vector<T>::mul(*scratch1, *w, *w);
      T alpha;
      
      /*alpha = sum(s1)*/
      alpha = scratch1->sum();
      
      uint k=0;
      
      while(k<steps){
	/*s1 = v * v*/
	Vector<T>::mul(*scratch1, *v, *v);
	T residual;
	
	/*res = sqrt(sum(s1))*/
	residual = scratch1->sum();
	
	if(sqrt(fabs(residual)) < tolerance){
	  message("Success in %d iterations, %10.10e, %10.10e", k, residual, sqrt(residual));
	  return;
	}
	
	/*u = A*v*/
	spmv(*u, *(this->mat), *v);
	
	/*s1 = v * u*/
	Vector<T>::mul(*scratch1, *v, *u);
	T divider;
	
	/*divider = sum(s1)*/
	divider = scratch1->sum();
	
	T t = alpha/divider;
	
	/*x = x + t*v*/
	/*r = r - t*u*/
	/*w = C * r*/
	Vector<T>::mfadd(*(this->x),  t, *v, *(this->x));
	Vector<T>::mfadd(*r, -t, *u, *r);
	Vector<T>::mul  (*w, *C, *r);
	
	/*s1 = w*w*/
	Vector<T>::mul(*scratch1, *w, *w);
	/*beta = sum(s1)*/
	T beta = scratch1->sum();
	
#if 1
	if(beta< tolerance){
	  float rl = r->length2(); 
	  if(rl<tolerance){
	    message("Success in %d iterations, %10.10e, %10.10e", k, rl, 
		    sqrt(fabs(rl)));
	    return;
	  }
	}
#endif
	
	T s = beta/alpha;
	
	/*s1 = C * w*/
	Vector<T>::mul(*scratch1, *C, *w);
	/*v = s1 + s * v*/
	Vector<T>::mfadd(*v, s, *v, *scratch1);
	alpha = beta;
	k++;
      }
      message("Unsuccesfull");
      throw new SolutionNotFoundException(__LINE__, __FILE__,
					  "Number of iterations exceeded.");
      
    }
  protected:
    Vector<T>* r;
    Vector<T>* C;
    Vector<T>* scratch1;
    Vector<T>* scratch2;
    Vector<T>* w;
    Vector<T>* v;
    Vector<T>* u;    
  };

#ifdef CUDA
  template<int N, class T>
  class LinSolveCGGPU : public LinSolve<N, T>{
  public:
    LinSolveCGGPU(uint d, int t, int n_thr=256, 
		  TextureOperation tex = TexVector):LinSolve<N, T>(d){
      pool = CGF::getThreadPool(t);
      task = new ParallelCGCudaTask<N, T>(pool, this->x, this->mat, this->b,
					  n_thr, tex);
      allocated = false;
    }

    virtual ~LinSolveCGGPU(){
      /*Deallocate GPU memory*/
      pool->assignTask(task, Deallocate);
      pool->sync();

      delete task;
    }

    /*Update vector and/or matrix in ParallelCGTask*/
    virtual void setb(Vector<T>* vec){
      task->setb(vec);
      LinSolve<N, T>::setb(vec);
    }

    virtual void setx(Vector<T>* vec){
      task->setx(vec);
      LinSolve<N, T>::setx(vec);
    }

    virtual void setMatrix(SpMatrix<N, T>* m){
      task->setMatrix(m);
      LinSolve<N, T>::setMatrix(m);
    }

    virtual void preSolve(){
      this->mat->finalize();

      if(allocated){
	/*If memory was allocated, deallocate and upload modified matrix*/
	pool->assignTask(task, Deallocate);
	pool->sync();
	allocated = false;
      }

      task->computePreconditioner();
      task->prepareMatrix();

      /*Allocation contains conversion from host to optimized CUDA
	representation, which is not optimized at all, and the
	creation of temporary vectors.*/
      pool->assignTask(task, Allocate);
      pool->sync();
      allocated = true;
    }

    virtual void solve(uint steps = 100000, T tolerance = 1e-6){
      cgfassert(this->mat->getWidth() == this->b->getSize());
      cgfassert(this->mat->getWidth() == this->mat->getHeight());

      task->setTolerance(tolerance);
      task->setMaxIterations(steps);

      /*Start the algorithm and measure the elapsed time. The measured
	time is very close to the time measured using cudaEvents.*/
      pool->assignTask(task, SolveSystem);
      pool->sync();
      
      /*Copy back the result*/
      pool->assignTask(task, CopyResult);
    }
    
  protected:
    ParallelCGCudaTask<N,T>* task;
    ThreadPool* pool;
    bool allocated;
  private:
    LinSolveCGGPU();
    LinSolveCGGPU(const LinSolveCGGPU&);
    LinSolveCGGPU& operator=(const LinSolveCGGPU&);
  };
#endif

  template<int N, class T>
  class LinSolveCGParallel : public LinSolve<N, T>{
  public:
    LinSolveCGParallel(uint d, uint t):LinSolve<N, T>(d){
      pool = CGF::getThreadPool(t);
      task = new ParallelCGTask<N,T>(pool->getSize(), this->x, this->mat,
				     this->b);
    }

    virtual ~LinSolveCGParallel(){
      delete task;
    }

    virtual void preSolve(){
      /*Compute distribution among the threads and compute the
	preconditioner*/
      task->computePreconditioner();
      task->computeDistribution();
    }

    virtual void solve(uint steps = 100000, T tolerance = 1e-6){
      cgfassert(this->mat->getWidth() == this->b->getSize());
      cgfassert(this->mat->getWidth() == this->mat->getHeight());
      
      this->mat->finalize();

      task->setTolerance(tolerance);
      task->setMaxIterations(steps);

      pool->assignTask(task);
      pool->sync();
    }

    
    /*Update vector and/or matrix in ParallelCGTask*/
    virtual void setb(Vector<T>* vec){
      task->setb(vec);
      LinSolve<N, T>::setb(vec);
    }

    virtual void setx(Vector<T>* vec){
      task->setx(vec);
      LinSolve<N, T>::setx(vec);
    }

    virtual void setMatrix(SpMatrix<N, T>* m){
      task->setMatrix(m);
      LinSolve<N, T>::setMatrix(m);
    }

  protected:
    ParallelCGTask<N,T>* task;
    ThreadPool* pool;
  private:
    LinSolveCGParallel();
    LinSolveCGParallel(const LinSolveCGParallel&);
    LinSolveCGParallel& operator=(const LinSolveCGParallel&);
  };

#ifndef CUDA
  /*Substitute Parallel solver*/
  template<int N, class T>
  class LinSolveCGGPU : public LinSolveCGParallel<N, T>{

  };
#endif
}

#endif/*LINSOLVECG*/
