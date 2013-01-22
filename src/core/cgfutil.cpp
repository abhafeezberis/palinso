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

#ifndef NO_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace CGF;

namespace CGF{

  uint traceLevel = 0;
  struct timeval start_time;
  pthread_mutex_t msg_mutex;
  static bool messages_enabled = true;
  //static std::streambuf *oldstdcout = 0;
  //static int message_enabled_stack = 0;

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

  void destroy(){
#ifdef NOISE
    Random<float>::destroy();
    Random<double>::destroy();

    Noise<float>::destroy();
    Noise<double>::destroy();
#endif
    destroyThreadPools();

#ifndef NO_CUDA
    cudaThreadExit();
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

  /*if message_enabled_stack >= 0 -> messages_enabled = true
    if message_enabled_stack <  0 -> messages_enabled = false
   */

  void enableMessages(){
#if 0
    pthread_mutex_lock(&msg_mutex);
    message_enabled_stack++;
#if 1
    if(message_enabled_stack >= 0){
      if(!messages_enabled){
	messages_enabled = true;
	std::cout.rdbuf(oldstdcout);
      } 
    }
#endif
    pthread_mutex_unlock(&msg_mutex);
#endif
  }

  void disableMessages(){
#if 0
    pthread_mutex_lock(&msg_mutex);
    message_enabled_stack--;
#if 1
    if(message_enabled_stack < 0){
      if(messages_enabled){
	messages_enabled = false;
	oldstdcout = std::cout.rdbuf();
	std::cout.rdbuf(0);
      }
    }
#endif
    pthread_mutex_unlock(&msg_mutex);
#endif
  }

  void message(const char* format, ...){
    if(!messages_enabled){
      return;
    }
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
    fflush(stderr);
    /*Since we abort the program, flush stdout*/
    fflush(stdout);
    va_end(args);
    fprintf(stderr,"\n");
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

#if 0
  extern CGFAPI CGFuint cpuid();

#define CPU_HAS_TSC               0x001
#define CPU_HAS_MMX               0x002
#define CPU_HAS_MMXEX             0x004
#define CPU_HAS_SSE               0x008
#define CPU_HAS_SSE2              0x010
#define CPU_HAS_3DNOW             0x020
#define CPU_HAS_3DNOWEXT          0x040
#define CPU_HAS_SSE3              0x080
#define CPU_HAS_HT                0x100

#define cpuid(op, eax, ecx, edx)		\
  asm volatile ("pushl %%ebx \n\t"		\
                "cpuid       \n\t"		\
		"popl  %%ebx \n\t"		\
		: "=a" (eax),			\
		  "=c" (ecx),			\
		  "=d" (edx)			\
		: "a"  (op)			\
		: "cc")

  uint cpuid(){
    uint eax, ecx, edx, caps;
#if !(defined(__i586__) || defined(__i686__) || defined(__athlon__) || defined(__pentium4__) || defined(__x86_64__))
    asm volatile ("pushfl             \n\t"
		  "popl  %0           \n\t"
		  "movl  %0,%1        \n\t"
		  "xorl  $0x200000,%0 \n\t"
		  "pushl %0           \n\t"
		  "popfl              \n\t"
		  "pushfl             \n\t"
		  "popl  %0           \n\t"
		  : "=a" (eax),
		    "=d" (edx)
		  :
		  : "cc");
    if(eax==edx)
      return 0;
#endif
    caps = 0;
    cpuid(0x00000000, eax, ecx, edx);
    if(eax){
      // AMD:   ebx="Auth" edx="enti" ecx="cAMD"
      // Intel: ebx="Genu" edx="ineI" ecx="ntel"
      // VIAC3: ebx="Cent" edx="aurH" ecx="auls"

      if((ecx==0x444d4163) && (edx==0x69746e65)){
	cpuid(0x80000000, eax, ecx, edx);
	if(eax>0x80000000){
	  cpuid(0x80000001, eax, ecx, edx);
	  if(edx&0x08000000) caps |= CPU_HAS_MMXEX;
	  if(edx&0x80000000) caps |= CPU_HAS_3DNOW;
	  if(edx&0x40000000) caps |= CPU_HAS_3DNOWEXT;
	}
      }

      cpuid(0x00000001, eax, ecx, edx);
      if(edx&0x00000010) caps |= CPU_HAS_TSC;
      if(edx&0x00800000) caps |= CPU_HAS_MMX;
      if(edx&0x02000000) caps |= CPU_HAS_SSE;
      if(edx&0x04000000) caps |= CPU_HAS_SSE2;
      if(edx&0x10000000) caps |= CPU_HAS_HT;
      if(edx&0x00000001) caps |= CPU_HAS_SSE3;
    }

    return caps;
  }
#endif
}

