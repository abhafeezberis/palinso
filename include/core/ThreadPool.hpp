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

#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <pthread.h>

#include "core/cgfdefs.hpp"
#include "core/Barrier.hpp"
#include "core/ShutDownTask.hpp"
#include <queue>

#define DEFAULT_N_THREADS 8

/*A thread executes a part of a complete task. Also one thread can
  execute exactly one task. We can synchronize between parts of a task
  or tasks. The tasks need to share a synchronization object for this
  in order to communicate with each other.*/

/*Synchronization using a barier, each thread must reach a certain
  barrier until all threads can continue, can be implemented using a
  pthread_cond_broadcast*/

namespace CGF{
  class Thread;
  class ThreadPool;
  class Barrier;

  typedef struct _task task_t;
  struct _task{
    Task* task;
    uint subTask;
  };

  /*Manages the execution of tasks given a number of threads*/

  /*When a thread has executed its associated task(s), the thread is
    set into a blocking state. If a new task is assigned the threads
    are unblocked and the execution can be continued.*/
  class CGFAPI ThreadPool{
  public:
    friend class Thread;
    ThreadPool();
    ThreadPool(uint n);
    ThreadPool(const ThreadPool&);
    ~ThreadPool();

    void start(); /*Starts all the threads*/

    /*Pauses the execution of all threads*/
    void pause(uint usec);

    /*Continues the execution of all blocked threads. For instance
      after the addition of new tasks*/
    void unblock();

    /*Executes the task with the specified number of threads*/
    void assignTask(Task* task, uint subTask = 0);

    void assignNewTask();

    void sync();

    uint getSize()const{
      return n_threads;
    }

    bool empty()const{
      message("size = %d", taskQueue.size());
      return taskQueue.size()==0?true:false;
    }

  protected:
    void removeLastTask();

    void removeFaultyTask(Task* task);

    void waitForEmptyQueue();

    void waitForNewTask();      /*Called by an active thread. If no
				  tasks are avaialble this function
				  will block*/

    uint n_blocked_threads;

    uint n_threads;
    Barrier* barrier;
    Thread** threads;

    std::queue<task_t> taskQueue;

    Task* getCurrentTask();  /*Called by child threads to obtain a
			       current task.  If NULL is returned, no
			       tasks are in the queue and the calling
			       thread should block. If the main
			       thread adds a new task, the blocked
			       threads shoud be signaled*/

    pthread_mutex_t condMutex;
    pthread_cond_t condition;

    pthread_mutex_t blockMutex;
    pthread_cond_t blockCondition;

    pthread_mutex_t syncMutex;
    pthread_cond_t syncCondition;

    pthread_mutex_t queueMutex;    

    pthread_mutex_t blockedMutex;
  }; 
}

#endif/*THREADPOOL*/
