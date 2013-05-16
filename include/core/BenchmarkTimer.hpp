/* Copyright (C) 2012 by Mickeal Verschoor

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

#ifndef BENCHMARKTIMER_HPP
#define BENCHMARKTIMER_HPP

#include "core/cgfdefs.hpp"
#include <map>
#include <string>

namespace CGF{
  typedef struct _timer timer;

  struct _timer{
    struct timeval total_time;
    struct timeval last_time;
    uint n;
    uint state;
  };


  class CGFAPI BenchmarkTimer{
  public:
    BenchmarkTimer(){
      timeMap.clear();
    }

    void start(const char* timerName);
    void stop(const char* timerName);
    ulong getAverageTimeUSec(const char* timerName);
    ulong getTotalTimeUSec(const char* timerName);
    ulong getTotalCalls(const char* timerName);

    ulong getAccumulativeUSec();

    void printAverageUSec(const char* timerName);
    void printTotalUSec(const char* timerName);

    void printAllAverageUSec();
    void printAllTotalUSec();

    void printAccumulativeUSec();

  protected:
    std::map<std::string, timer>  timeMap;
    
  private:
    BenchmarkTimer(const BenchmarkTimer&);
    BenchmarkTimer& operator=(const BenchmarkTimer&);
  };

#ifdef BENCHMARK
#define TIME(timer, name, call)			\
  timer.start(name);				\
  call;						\
  timer.stop(name);				
#else
#define TIME(timer, name, call)			\
  call;
#endif
}

#endif/*BENCHMARKTIMER_HPP*/
