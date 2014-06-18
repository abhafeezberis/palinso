#ifndef LINSOLVECGGPU_HPP
#define LINSOLVECGGPU_HPP

#include "math/LinSolveCGParallel.hpp"
#include "core/ThreadPool.hpp"

#ifdef CUDA
#include "math/CUDASpmv.hpp"
#include "math/ParallelCGCudaTask.hpp"
#endif

namespace CGF{

#ifdef CUDA
  template<int N, class T>
  class LinSolveCGGPU : public LinSolve<N, T>{
  public:
    LinSolveCGGPU(int d, int t, int n_thr=256, 
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

    virtual void solve(int steps = 100000, T tolerance = (T)1e-6){
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


#ifndef CUDA
  /*Substitute Parallel solver*/
  template<int N, class T>
  class LinSolveCGGPU : public LinSolveCGParallel<N, T>{

  };
#endif
}

#endif/*LINSOLVECGGPU_HPP*/
