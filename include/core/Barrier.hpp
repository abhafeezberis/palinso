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

#ifndef BARRIER_HPP
#define BARRIER_HPP

#ifdef USE_THREADS

#include "core/cgfdefs.hpp"
#include <pthread.h>

namespace CGF{
  class ThreadPool;

#define PBARRIER

  class CGFAPI Barrier{
  public:
    Barrier(ThreadPool* p);
    Barrier(const Barrier&);
    ~Barrier();
    
    /*Blocks calling thread until all other threads have reached this
      point*/
    void sync()const{
      if(n_shared_threads == 1){
	return;
      }
#ifdef PBARRIER
      pthread_barrier_wait(&bar);
#else
      if(n_shared_threads == 1){
        /*If a pool is created with one thread, we don't need
          syncronization and barriers between the worker threads. So
          we can safely return.*/
        return;
      }
      pthread_yield();
      
      pthread_mutex_lock(&mutex);
      
      block_status++;
      
      if(block_status == n_shared_threads){
        block_status = 0;
        /*All threads have reached this point, lets continue*/
        pthread_cond_broadcast(&cond);
      }else{
        /*Not all threads have reached this point, wait for them*/
        pthread_cond_wait(&cond, &mutex);
      }
      
      pthread_mutex_unlock(&mutex);
#endif
    }
    
    void block()const;               /*Blocks the calling thread until
                                       all threads have reached this
                                       this point. If all threads have
                                       reached this point the threads
                                       can continue by calling
                                       unblock() from a different
                                       thread.*/
    
    void unblock()const;             /*Continues the blocked
                                       threads. This function should
                                       be called from the main thread
                                       since all child threads are
                                       blocked*/
    
  protected:
    int n_shared_threads;            /*Contains the number of threads
                                       sharing this barrier*/
    
    int n_threads_sync;             /*Number of unique threads that
                                      have reached this point*/
    
    mutable pthread_cond_t block_cond;
    /*Condition value used to block /
      signal all associated blocked
      threads*/
    
    mutable pthread_mutex_t block_mutex;     
    /*Mutex object associated with
      the blocking barrier*/
    
    ThreadPool* pool;                /*The pool assiciated with this
                                       barrier*/
    int id;
    
#ifdef PBARRIER
    mutable pthread_barrier_t bar;
#else
    mutable pthread_mutex_t mutex;   /*Mutex object associated with
                                       this barrier*/
    
    mutable pthread_cond_t cond;     /*Condition value used to signal
                                       all associated threads*/
    mutable int block_status;
#endif
  };
}

#endif/*USE_THREADS*/

#endif/*BARRIER_HPP*/
