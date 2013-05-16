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

#include <pthread.h>
#include "core/cgfdefs.hpp"
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <string.h>
#include <map>
#include <iostream>

#ifdef NOISE
#include "math/Noise.hpp"
#include "math/Random.hpp"
#endif
#include "core/ThreadPool.hpp"
#include "core/daemon.hpp"

#ifndef CUDA
#include <cuda_runtime_api.h>
#endif

using namespace CGF;

namespace CGF{

  uint traceLevel = 0;
  struct timeval start_time;
  pthread_mutex_t msg_mutex;

  /*Static map which contains all instances of threadpools*/
  static std::map<uint, ThreadPool*> threadPools;

  void gettimediffstring(char* buffer, timeval diff){
    sprintf(buffer, "%6d.%06d", (int)diff.tv_sec,(int)diff.tv_usec);
  }

  void init(uint daemon){
    gettimeofday(&start_time, NULL);
    pthread_mutex_init(&msg_mutex, NULL);

#ifdef NOISE
    /*Initialize random number generators*/
    Random<float>::init();
    Random<double>::init();

    Noise<float>::init();
    Noise<double>::init();
#endif

    threadPools.clear();

    if(daemon){
      continue_as_daemon_process();
      if(is_daemon()){
	redirect_std_file_descriptors();
      }else{
	message("Exit %d", getpid());
	//destroy();
	exit(0);
      }
    }
  }

  void destroyThreadPools(){
    std::map<uint, ThreadPool*>::iterator it = threadPools.begin();
    for(;it!=threadPools.end();it++){
      ThreadPool* p = (*it).second;
      delete p;
    }
    threadPools.clear();
  }

#ifdef CUDA
  extern bool cudaHostthreadInitialized;
#endif

  void destroy(){
#ifdef NOISE
    Random<float>::destroy();
    Random<double>::destroy();

    Noise<float>::destroy();
    Noise<double>::destroy();
#endif
    destroyThreadPools();

#ifdef CUDA
    if(cudaHostthreadInitialized){
      cudaThreadExit();
      cudaHostthreadInitialized = false;
    }
#endif
  }

  ThreadPool* getThreadPool(uint i){
    if(threadPools[i] == 0){
      threadPools[i] = new ThreadPool(i);
      threadPools[i]->start();
    }
    return threadPools[i];
  }

  int cgfrandom(int& seed){
    seed = 1664525UL*seed+1013904223UL;
    return seed;
  }

  void message(const char* format, ...){
    va_list args;
    timeval now;
    timeval diff;
    char buffer[80];

    gettimeofday(&now, NULL);
    timersub(&now, &start_time, &diff);
    gettimediffstring(buffer, diff);

    pthread_mutex_lock(&msg_mutex);

    fprintf(stdout,"[M   %s] ", buffer);

    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);
    fprintf(stdout,"\n");
    fflush(stdout);
    pthread_mutex_unlock(&msg_mutex);
  }

  void error(const char* format, ...){
    va_list args;
    timeval now;
    timeval diff;
    char buffer[80];

    gettimeofday(&now, NULL);
    timersub(&now, &start_time, &diff);
    gettimediffstring(buffer, diff);

    pthread_mutex_lock(&msg_mutex);

    fprintf(stderr,"[  E %s] ", buffer);

    va_start(args, format);
    vfprintf(stderr, format, args);
    fprintf(stderr,"\n");
    fflush(stderr);
    /*Since we abort the program, flush stdout*/
    fflush(stdout);
    va_end(args);

    pthread_mutex_unlock(&msg_mutex);
    abort();
  }

  void warning(const char* format, ...){
    va_list args;
    timeval now;
    timeval diff;
    char buffer[80];

    gettimeofday(&now, NULL);
    timersub(&now, &start_time, &diff);
    gettimediffstring(buffer, diff);

    pthread_mutex_lock(&msg_mutex);

    fprintf(stderr,"[ W  %s] ", buffer);

    va_start(args, format);
    vfprintf(stderr, format, args);
    fflush(stderr);
    va_end(args);
    fprintf(stderr,"\n");

    pthread_mutex_unlock(&msg_mutex);
  }

  void _cgfassert(const char* expr, const char* file, uint line){
    error("%s:%d: ASSERT(%s) failed.\n", file, line, expr);
  }

  void trace(unsigned int level, const char* format, ...){
    if(traceLevel>level){
      va_list args;
      va_start(args, format);
      vfprintf(stderr, format, args);
      fflush(stderr);
      va_end(args);
    }
  }

  void msleep(uint n){

  }

  void usleep(uint n){

  }

  uint strhash(const char* string){
    register const unsigned char *s = (const unsigned char*)string;
    register uint h=0;
    register uint c;
    while((c=*s++)!='\0'){
      h = ((h<<5)+h)^c;
    }
    return h;
  }
}

