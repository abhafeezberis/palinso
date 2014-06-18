#ifndef LINSOLVECGPARALLEL_HPP
#define LINSOLVECGPARALLEL_HPP

#include "math/LinSolve.hpp"
#include "core/ThreadPool.hpp"
#include "math/ParallelCGTask.hpp"

namespace CGF{
  template<int N, class T>
  class LinSolveCGParallel : public LinSolve<N, T>{
  public:
    LinSolveCGParallel(int d, int t):LinSolve<N, T>(d){
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

    virtual void solve(int steps = 100000, T tolerance = (T)1e-6){
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
}


#endif/*LINSOLVECGPARALLEL_HPP*/
