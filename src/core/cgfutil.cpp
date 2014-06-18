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

#include "core/cgfdefs.hpp"
#ifdef USE_THREADS
#include <pthread.h>
#endif
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#ifdef _WIN32

#else
#include <unistd.h>
#include <sys/time.h>
#endif
#include <string.h>
#include <map>
#include <iostream>

#ifdef NOISE
#include "math/Noise.hpp"
#include "math/Random.hpp"
#endif
#ifdef USE_THREADS
#include "core/ThreadPool.hpp"
#include "core/daemon.hpp"
#endif

using namespace CGF;

namespace CGF{

  int traceLevel = 0;
  struct timeval start_time;
#ifdef USE_THREADS
  pthread_mutex_t msg_mutex;
#endif

  /*Static map which contains all instances of threadpools*/
  static std::map<int, ThreadPool*> threadPools;

  void gettimediffstring(char* buffer, timeval diff){
    sprintf(buffer, "%6d.%06d", (int)diff.tv_sec,(int)diff.tv_usec);
  }

  void init(int daemon){
    gettimeofday(&start_time, NULL);
#ifdef USE_THREADS
    pthread_mutex_init(&msg_mutex, NULL);
#endif

#ifdef NOISE
    /*Initialize random number generators*/
    //StaticRandom<float>::init();
    //StaticRandom<double>::init();

    Noise<float>::init();
    Noise<double>::init();
#endif

#ifdef USE_THREADS
    threadPools.clear();

    if(daemon == 1){
      continue_as_daemon_process();
      if(is_daemon()){
        redirect_std_file_descriptors();
      }else{
        message("Exit %d", getpid());
        //destroy();
        exit(0);
      }
    }

    if(daemon == 2){
      continue_as_daemon_process();
      if(is_daemon()){
        redirect_std_file_descriptors_null();
      }else{
        message("Exit %d", getpid());
        //destroy();
        exit(0);
      }
    }
#endif
  }

  void destroyThreadPools(){
#ifdef USE_THREADS
    std::map<int, ThreadPool*>::iterator it = threadPools.begin();
    for(;it!=threadPools.end();it++){
      ThreadPool* p = (*it).second;
      delete p;
    }
    threadPools.clear();
#endif
  }

#ifdef CUDA
  extern bool cudaHostthreadInitialized;
#endif

  void destroy(){
#ifdef NOISE
    //StaticRandom<float>::destroy();
    //StaticRandom<double>::destroy();

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

  ThreadPool* getThreadPool(int i){
#ifdef USE_THREADS
    if(threadPools[i] == 0){
      threadPools[i] = new ThreadPool(i);
      threadPools[i]->start();
    }
    return threadPools[i];
#endif
	return 0;
  }

  void message(const char* format, ...){
    va_list args;
    timeval now;
    timeval diff;
    char buffer[80];

    gettimeofday(&now, NULL);
    timersub(&now, &start_time, &diff);
    gettimediffstring(buffer, diff);

#ifdef USE_THREADS
    pthread_mutex_lock(&msg_mutex);
#endif

    fprintf(stdout,"[M   %s] ", buffer);

    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);
    fprintf(stdout,"\n");
    fflush(stdout);

#ifdef USE_THREADS
    pthread_mutex_unlock(&msg_mutex);
#endif
  }

  void error(const char* format, ...){
    va_list args;
    timeval now;
    timeval diff;
    char buffer[80];

    gettimeofday(&now, NULL);
    timersub(&now, &start_time, &diff);
    gettimediffstring(buffer, diff);

#ifdef USE_THREADS
    pthread_mutex_lock(&msg_mutex);
#endif

    fprintf(stderr,"[  E %s] ", buffer);

    va_start(args, format);
    vfprintf(stderr, format, args);
    fprintf(stderr,"\n");
    fflush(stderr);
    /*Since we abort the program, flush stdout*/
    fflush(stdout);
    va_end(args);

#ifdef USE_THREADS
    pthread_mutex_unlock(&msg_mutex);
#endif
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

#ifdef USE_THREADS
    pthread_mutex_lock(&msg_mutex);
#endif

    fprintf(stderr,"[ W  %s] ", buffer);

    va_start(args, format);
    vfprintf(stderr, format, args);
    fflush(stderr);
    va_end(args);
    fprintf(stderr,"\n");

#ifdef USE_THREADS
    pthread_mutex_unlock(&msg_mutex);
#endif
  }

  void _cgfassert(const char* expr, const char* file, int line){
    error("%s:%d: ASSERT(%s) failed.\n", file, line, expr);
  }

  int strhash(const char* string){
    register const unsigned char *s = (const unsigned char*)string;
    register int h=0;
    register int c;
    while((c=*s++)!='\0'){
      h = ((h<<5)+h)^c;
    }
    return h;
  }
}

