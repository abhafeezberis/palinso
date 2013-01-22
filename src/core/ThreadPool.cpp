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

#include <iostream>
#include "core/ThreadPool.hpp"
#include "core/Timer.hpp"
#include "core/cgfdefs.hpp"
#include "core/Exception.hpp"
#include <stdlib.h>
#include <unistd.h>

namespace CGF{

  ThreadPool::ThreadPool(){
    uint i;
    n_threads = DEFAULT_N_THREADS;
    n_blocked_threads = 0;

    barrier = new Barrier(this);
    threads = new Thread*[n_threads];
    
    for(i=0;i<n_threads;i++){
      threads[i] = new Thread(i);
      threads[i]->setBarrier(barrier);
      threads[i]->setLastId(n_threads);
      threads[i]->setThreadPool(this);
    }

    pthread_mutex_init(&condMutex, NULL);
    pthread_mutex_init(&blockMutex, NULL);
    pthread_mutex_init(&syncMutex, NULL);
    pthread_cond_init(&condition, NULL);
    pthread_cond_init(&blockCondition, NULL);
    pthread_cond_init(&syncCondition, NULL);
    //message("Threadpool created with %d threads", n_threads);
  }

  ThreadPool::ThreadPool(uint n){
    uint i;
    n_threads = n;
    n_blocked_threads = 0;

    barrier = new Barrier(this);
    threads = new Thread*[n_threads];
    
    for(i=0;i<n_threads;i++){
      threads[i] = new Thread(i);
      threads[i]->setBarrier(barrier);
      threads[i]->setLastId(n_threads);
      threads[i]->setThreadPool(this);
    }

    pthread_mutex_init(&condMutex, NULL);
    pthread_mutex_init(&blockMutex, NULL);
    pthread_mutex_init(&syncMutex, NULL);
    pthread_mutex_init(&queueMutex, NULL);
    pthread_mutex_init(&blockedMutex, NULL);
    pthread_cond_init(&condition, NULL);
    pthread_cond_init(&blockCondition, NULL);
    pthread_cond_init(&syncCondition, NULL);
    //message("Threadpool created with %d threads", n_threads);
  }

  ThreadPool::~ThreadPool(){
    Task* t = new ShutDownTask(n_threads);
    
    assignTask(t, SHUTDOWN_TASK);
    
    /*Join threads*/
    for(uint i=0;i<n_threads;i++){
      uint* thread_return;
      pthread_join(threads[i]->thread_id, (void**)&thread_return);
    }

    //sync();
    sleep(1);

    delete t;

    for(uint i=0;i<n_threads;i++){
      delete threads[i];
      threads[i] = 0;
    }

    delete [] threads;
    n_threads = 0;

    delete barrier;

    pthread_mutex_destroy(&condMutex);
    pthread_cond_destroy(&condition);
    pthread_mutex_destroy(&blockMutex);
    pthread_cond_destroy(&blockCondition);
    pthread_mutex_destroy(&syncMutex);
    pthread_cond_destroy(&syncCondition);

    pthread_mutex_destroy(&queueMutex);

    pthread_mutex_destroy(&blockedMutex);
  }

  void ThreadPool::start(){
    uint i = 0;
    for(i=0;i<n_threads;i++){
      threads[i]->start();
    }
  } 

  void ThreadPool::unblock(){
    barrier->unblock();
  }

  void ThreadPool::waitForNewTask(){
    /*All threads must be finished with their work*/
    barrier->sync();

    /*All threads have reached this point, lets continue and figure
      out if we can add a new task to the threads. Else all threads
      have to wait for this*/
    
    pthread_mutex_lock(&condMutex);
    pthread_mutex_lock(&queueMutex);
    if(taskQueue.empty()){
      pthread_mutex_unlock(&queueMutex);
      /*The task queue is empty, let the threads wait and signal the
	main thread which might requested a synchronization*/
      pthread_cond_signal(&syncCondition);
      pthread_cond_wait(&condition, &condMutex);
    }else{
      pthread_mutex_unlock(&queueMutex);
    }
    pthread_mutex_unlock(&condMutex);


    pthread_mutex_lock(&blockMutex);
      
    n_blocked_threads++;
    if(n_blocked_threads == n_threads){
      /*The last arrived thread will schedule a new task*/
      
      pthread_mutex_lock(&queueMutex);
      task_t curTask = taskQueue.front();
            
      pthread_mutex_unlock(&queueMutex);
      
      curTask.task->setSubTask(curTask.subTask);

      /*Assign a new task to each thread*/
      for(uint i=0;i<n_threads;i++){
	threads[i]->setTask(curTask.task);
      }

      /*Wake up all other threads*/
      pthread_cond_broadcast(&blockCondition);
      n_blocked_threads = 0;
    }else{
      pthread_cond_wait(&blockCondition, &blockMutex);
    }
    pthread_mutex_unlock(&blockMutex);
  }

  void ThreadPool::assignTask(Task* task, uint t){
    pthread_mutex_lock(&condMutex);
    pthread_mutex_lock(&blockMutex);

    if(task->getNumberOfThreads() != n_threads){
      throw ThreadPoolException(__LINE__, __FILE__, 
				"Task number of threads mismatch number of threads in selected pool");
    }

    task_t ts;
    ts.task = task;
    ts.subTask = t;
    
    pthread_mutex_lock(&queueMutex);
    taskQueue.push(ts);
    pthread_mutex_unlock(&queueMutex);

    /*Signal worker threads*/
    pthread_cond_broadcast(&condition);

    pthread_mutex_unlock(&blockMutex);
    pthread_mutex_unlock(&condMutex);
  }

  void ThreadPool::sync(){
    pthread_mutex_lock(&syncMutex);
    pthread_mutex_lock(&queueMutex);

    if(taskQueue.empty()){
      pthread_mutex_unlock(&queueMutex);
    }else{
      pthread_mutex_unlock(&queueMutex);
      /*Block until the queue becomes empty.*/
      pthread_cond_wait(&syncCondition, &syncMutex);
    }

    pthread_mutex_unlock(&syncMutex);
  }

  void ThreadPool::removeLastTask(){
    /*Remove the task when the task has been executed since the sync()
      function checks the status of the queue. If the queue is empty
      sync() will unblock.*/
    pthread_mutex_lock(&blockedMutex);
    n_blocked_threads++;

    if(n_blocked_threads == n_threads){

      n_blocked_threads = 0;
      pthread_mutex_lock(&queueMutex);

      taskQueue.pop();

      pthread_mutex_unlock(&queueMutex);
    }
    pthread_mutex_unlock(&blockedMutex);

    barrier->sync();
  }

  void ThreadPool::removeFaultyTask(Task* task){
    message("Removing faulty task %p from task queue", task);
    pthread_mutex_lock(&queueMutex);
    
    while(taskQueue.front().task == task){
      message("Removing subtask %d", taskQueue.front().subTask);
      taskQueue.pop();
    }

    pthread_mutex_unlock(&queueMutex);
    message("Faulty task %p removed from task queue", task);
  }
}
