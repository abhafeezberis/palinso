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

#include "core/BenchmarkTimer.hpp"

namespace CGF{
  void BenchmarkTimer::start(const char* timerName){
    std::map<std::string, timer>::iterator it;
    std::string key(timerName);

    it = timeMap.find(key);
    if(it == timeMap.end()){
      /*Timer does not exist*/
      timer t;
      timerclear(&t.total_time);
      timerclear(&t.last_time);
      t.n = 0;
      t.state = 0;
      timeMap[key] = t;
    }
    /*The timer must be in a stoppped state*/
    cgfassert(timeMap[key].state == 0);

    timeMap[key].state = 1;
    gettimeofday(&(timeMap[key].last_time), 0);
  }

  void BenchmarkTimer::stop(const char* timerName){
    struct timeval now;
    struct timeval diff;
    gettimeofday(&now, 0);

    std::string key(timerName);

    /*Timer must be in a running state in order to stop*/
    cgfassert(timeMap[key].state == 1);

    std::map<std::string, timer>::iterator it = timeMap.find(key);
    if(it == timeMap.end()){
      error("Trying to access non existing timer");
    }

    timersub(&now, &(timeMap[key].last_time), &diff);
    timeradd(&diff, &(timeMap[key].total_time), 
	     &(timeMap[key].total_time));
    timeMap[key].n++;
    timeMap[key].state = 0;
  }

  ulong BenchmarkTimer::getAverageTimeUSec(const char* timerName){
    ulong usec = 0;

    std::string key(timerName);

    std::map<std::string, timer>::iterator it = timeMap.find(key);
    if(it == timeMap.end()){
      return 0;
    }else{
      usec = timeMap[key].total_time.tv_sec * 1000000 + 
	timeMap[key].total_time.tv_usec;
      usec = (long)((double) usec/(double)timeMap[key].n);
    return usec; 
    }
  }

  ulong BenchmarkTimer::getTotalTimeUSec(const char* timerName){
    ulong usec = 0;

    std::string key(timerName);
    
    std::map<std::string, timer>::iterator it = timeMap.find(key);
    if(it == timeMap.end()){
      return 0;
    }else{
      usec = timeMap[key].total_time.tv_sec * 1000000 + 
	timeMap[key].total_time.tv_usec;
      return usec;
    }
  }

  ulong BenchmarkTimer::getTotalCalls(const char* timerName){
    std::string key(timerName);

    std::map<std::string, timer>::iterator it = timeMap.find(key);
    if(it == timeMap.end()){
      return 0;
    }else{
      return timeMap[key].n;
    }
  }

  ulong BenchmarkTimer::getAccumulativeUSec(){
    std::map<std::string, timer>::iterator it;
    struct timeval acc;
    timerclear(&acc);

    for(it = timeMap.begin(); it != timeMap.end();it++){
      timeradd(&acc, &((*it).second.total_time), &acc);
    }

    ulong usec = acc.tv_sec * 1000000 + acc.tv_usec;
    return usec;
  }

  void BenchmarkTimer::printAverageUSec(const char* timerName){
    ulong usec = getAverageTimeUSec(timerName);
    message("Timer %s took %u usec on avarage after %d calls", 
	    timerName, usec, timeMap[std::string(timerName)].n);
  }

  void BenchmarkTimer::printTotalUSec(const char* timerName){
    ulong usec = getTotalTimeUSec(timerName);
    message("Timer %s took %u usec", timerName, usec);
  }

  void BenchmarkTimer::printAllAverageUSec(){
    std::map<std::string, timer>::iterator it;
    for(it = timeMap.begin(); it != timeMap.end();it++){
      timer t = (*it).second;

      ulong usec = t.total_time.tv_sec * 1000000 + 
	t.total_time.tv_usec;
      usec = (long)((double) usec/(double)t.n);
      message("Timer %s took %u usec on avarage after %d calls", 
	      (*it).first.c_str(), usec, t.n);
    }
  }

  void BenchmarkTimer::printAllTotalUSec(){
    std::map<std::string, timer>::iterator it;
    for(it = timeMap.begin(); it != timeMap.end();it++){
      timer t = (*it).second;

      ulong usec = t.total_time.tv_sec * 1000000 + 
	t.total_time.tv_usec;
      message("Timer %s took %u usec", 
	      (*it).first.c_str(), usec);
    }
  }


  void BenchmarkTimer::printAccumulativeUSec(){
    std::map<std::string, timer>::iterator it;
    struct timeval acc;
    timerclear(&acc);

    for(it = timeMap.begin(); it != timeMap.end();it++){
      timeradd(&acc, &((*it).second.total_time), &acc);
    }

    ulong usec = acc.tv_sec * 1000000 + acc.tv_usec;
    message("Accumulative time %u usec", usec);
  }
}
