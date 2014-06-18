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

#ifdef USE_THREADS
#include "core/Thread.hpp"
#include "core/ThreadPool.hpp"
#include "core/Exception.hpp"
#include "core/ShutDownTask.hpp"
#include <iostream>
#endif

#ifdef CUDA
#include "util/cuda_util.hpp"
#ifndef USE_THREADS
#error CUDA requires USED_THREADS to be defined
#endif
#endif

namespace CGF{
#ifdef USE_THREADS
  /*Interface between C and C++*/
    
  void thread_cleanup(void* arg){
    //message("Cleaning up thread objects for %d", pthread_self());
#ifdef CUDA
    exit_cuda_thread();
#endif
  }
  
  void* thread_start(void* params){
    Thread* thread = (Thread*)params;
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
    pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL);

    //message("set cleanup handler for %d", pthread_self());
    pthread_cleanup_push(thread_cleanup, NULL);

    thread->run();

    pthread_cleanup_pop(1);
    
    return NULL;
  }
  
  Thread::Thread(int _id):task(NULL), thread_id(0), 
                          barrier(NULL), id(_id) {
    status = 0;
    cudaAssigned = false;
  }
  
  Thread::~Thread(){
  }

  void Thread::start(){
    struct sched_param sched;
    sched.sched_priority = -19;

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    //pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&attr, SCHED_RR);
    pthread_attr_setschedparam(&attr, &sched);

    pthread_create(&thread_id, &attr, thread_start, this);
    //message("Thread %d started with pid %ul", id, thread_id);
    //status = 1;
    pthread_detach(thread_id);
  }
  
  void Thread::stop(){
    status = 0;
    pthread_exit(NULL);

#if 0
    if(pthread_cancel(thread_id)!=0){
      message("Thread cancel error");
    }
#endif
    if(pthread_attr_destroy(&attr)!=0){
      message("Thread attr destroy error");
    }

    message("Stopping thread %d", thread_id);
  }
  
  void Thread::run(){
    status = 1;
    while(status){
      /*Wait until a new task has been submitted and scheduled*/
      pool->waitForNewTask();
      
      /*Execute the task*/
      try{
        if(task->subTask == SHUTDOWN_TASK){
          status = 0;
          return;
        }
        task->execute(this);
        pool->removeLastTask();
      }catch(Exception& e){
        message("Execution of task failed");
        std::cerr << e.getError();
        /*Remove current task from pool by removing all (sub)tasks*/
        pool->removeFaultyTask(task);
      }catch(...){
        message("Unknown error");
      }
    }
  }
#endif
}
