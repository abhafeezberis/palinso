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

#ifndef SHUTDOWNTASK_HPP
#define SHUTDOWNTASK_HPP

#ifdef USE_THREADS

#include "core/Task.hpp"
#include "core/Thread.hpp"

#define SHUTDOWN_TASK 999999

namespace CGF{
  class CGFAPI ShutDownTask : public Task{
  public:
    ShutDownTask(int _n):Task(_n){
    }
    
    virtual ~ShutDownTask(){
    }
    
    virtual void exportSummary(){}
    
    void execute(const Thread* caller){
    }
  };
}

#endif/*USE_THREADS*/
#endif/*SHUTDOWNTASK_HPP*/
