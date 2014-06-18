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

#ifndef THREAD_HPP
#define THREAD_HPP
#ifdef USE_THREADS
#include "core/cgfdefs.hpp"
#include "core/Task.hpp"
#include "core/Barrier.hpp"
#include "util/cuda_util.hpp"

namespace CGF{

#define TID (caller?caller->getId():0)

  class CGFAPI Thread{
  public:
    Thread(int id);
    Thread(const Thread&);
    ~Thread();
    
    void start();
    void stop();
    void run();
    
    Task* getTask()const{
      return task;
    }
    void     setTask(Task* task){
      this->task = task;
    }
    
    int getId()const{
      return id;
    }
    
    void setLastId(int last_id){
      this->last_id = last_id;
    }

    int getLastId()const{
      return last_id;
    }

    void setBarrier(const Barrier* bar){
      barrier = bar;
    }

    void setThreadPool(ThreadPool* pool){
      this->pool = pool;
    }

    void sync()const{
      barrier->sync();
    }

    void block()const{
      barrier->block();
    }

    void setCuda()const{
#ifdef CUDA
      if(cudaAssigned == false){
        init_cuda_thread(id);
        cudaAssigned = true;
      }
#endif
    }
    
  protected:
    Task*          task;          /*The task that is currently being executed*/
    pthread_t      thread_id;     /*OS ID*/
    pthread_attr_t attr;          /*Thread attributes*/
    int            status;
    const Barrier* barrier;
    int            id;            /*Logical ID*/
    int            last_id;       /*Last logical ID*/

    ThreadPool*    pool;
    friend class Task;
    friend class ThreadPool;

    mutable bool cudaAssigned;
  };
}
#endif/*USE_THREADS*/
#endif/*THREAD_HPP*/
