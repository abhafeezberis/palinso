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

#ifndef TASK_HPP
#define TASK_HPP
#ifdef USE_THREADS
#include "core/cgfdefs.hpp"

namespace CGF{
  class CGFAPI Thread;
  class CGFAPI ThreadPool;

  enum CGSubTask{Allocate, Deallocate, CopyResult, UpdateBlocks, SolveSystem};
  
  class CGFAPI Task{
  public:
    Task(const int _n):subTask(0), valid(true), n_threads(_n){}
    virtual ~Task(){}
    virtual void execute(const Thread* caller) = 0;

    void setSubTask(int t){
      subTask = t;
    };

    bool isValid()const{
      return valid;
    }
    
    int getNumberOfThreads()const{
      return n_threads;
    }

  protected:
    int subTask;
    bool valid;
    const int n_threads;
    friend class Thread;
  };
}
#endif/*USE_THREADS*/
#endif/*TASK_HPP*/
