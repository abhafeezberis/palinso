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

#include "core/Barrier.hpp"
#include "core/ThreadPool.hpp"

namespace CGF{
  Barrier::Barrier(ThreadPool* p){
    n_shared_threads = p->getSize();
    pool = p;

    pthread_mutex_init(&block_mutex, NULL);
    pthread_cond_init(&block_cond, NULL);

#ifdef PBARRIER
    pthread_barrier_init(&bar, NULL, n_shared_threads);
#else
    block_status = 0;
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond, NULL);
#endif
  }

  Barrier::~Barrier(){
    pthread_mutex_destroy(&block_mutex);
    pthread_cond_destroy(&block_cond);

#ifdef PBARRIER
    pthread_barrier_destroy(&bar);
#else
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);
#endif
  }

  void Barrier::block()const{
    message("%s %ul", __PRETTY_FUNCTION__, pthread_self());
    /*Be shure that each associated thread reaches this point*/
    sync();
    pthread_mutex_lock(&block_mutex);
    pthread_cond_wait(&block_cond, &block_mutex);
    pthread_mutex_unlock(&block_mutex);
  }

  void Barrier::unblock()const{
    message("%s %ul", __PRETTY_FUNCTION__, pthread_self());
    pthread_mutex_lock(&block_mutex);
    pthread_cond_broadcast(&block_cond);
    pthread_mutex_unlock(&block_mutex);
  }
};
