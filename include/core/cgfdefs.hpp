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

#ifndef CGFDEFS_H
#define CGFDEFS_H

#define CGFAPI

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif
#ifndef NULL
#define NULL 0
#endif

#define DEFAULT_BLOCK_SIZE 8

#ifdef _WIN32
#include <Windows.h>
typedef unsigned int uint;
typedef unsigned long ulong;
#undef max
#undef min

int gettimeofday(struct timeval * tp, struct timezone * tzp);

#define timeradd(a, b, result)                           \
  do {                                                   \
    (result)->tv_sec = (a)->tv_sec + (b)->tv_sec;        \
    (result)->tv_usec = (a)->tv_usec + (b)->tv_usec;     \
    if ((result)->tv_usec >= 1000000)                    \
      {                                                  \
        ++(result)->tv_sec;                              \
        (result)->tv_usec -= 1000000;                    \
      }                                                  \
  } while (0)
#define    timersub(a, b, result)                         \
  do {                                                    \
    (result)->tv_sec = (a)->tv_sec - (b)->tv_sec;         \
    (result)->tv_usec = (a)->tv_usec - (b)->tv_usec;      \
    if ((result)->tv_usec < 0) {                          \
      --(result)->tv_sec;                                 \
      (result)->tv_usec += 1000000;                       \
    }                                                     \
  } while (0)
#else
#include <sys/time.h>
#include <sys/types.h>
#endif
#include <limits.h>

#include <stdlib.h>

namespace CGF{
#if defined __ANDROID__
  typedef unsigned long ulong;
#endif
  typedef unsigned char uchar;

  /*Math macros*/
#define ABS(a)        (((a)>=0)?(a):-(a))

#define MAX(a,b)      (((a)>(b))?(a):(b))

#define MIN(a,b)      (((a)>(b))?(b):(a))

#define MAX3(a,b,c)   (((a)>(b))?MAX(a,c):MAX(b,c))

#define MIN3(a,b,c)   (((a)>(b))?MAX(b,c):MAX(a,c))

#define MAX4(a,b,c,d) (MAX(MAX(a,b), MAX(c,d)))

#define MIN4(a,b,c,d) (MIN(MIN(a,b), MIN(c,d)))

#define CLAMP(a,x,b)  ((x)<(a))?(a):(((x)>(b))?(b):(x))

#define LERP(a,b,x)   ((a)+((b)-(a))*(x))

#define SQR(a)        ((a)*(a))

#ifdef _DEBUG
#define cgfassert(exp)   ((exp)?((void)0):(void)CGF::_cgfassert(#exp, __FILE__, __LINE__))
#else
#define cgfassert(exp)   ((void)0)
#endif

  class CGFAPI ThreadPool;

  /*Global functions*/
  extern CGFAPI void init(int daemon=0);
  extern CGFAPI void destroy();
  extern CGFAPI ThreadPool* getThreadPool(int i);
  extern CGFAPI int  cgfrandom(int& seed);
  extern CGFAPI int  cgfmalloc(void** ptr, ulong size);
  extern CGFAPI void cgffree(void** ptr);
  extern CGFAPI void gettimediffstring(char* buffer, timeval diff);
  extern CGFAPI void swapOutputStream();
  extern CGFAPI void warning(const char* format = "", ... );
  extern CGFAPI void message(const char* format = "", ... );
  extern CGFAPI void error(const char* format = "", ... );
  extern CGFAPI void _cgfassert(const char* expr, const char* filename,
                int line);
  extern CGFAPI void msleep(int n);
  extern CGFAPI void usleep(int n);
  extern CGFAPI int  strhash(const char* string);

  template<class T>
  void swap(T** a, T** b){
    T* tmp = *a;
    *a = *b;
    *b = tmp;
  }

  template<class T>
  void swap(T* a, T* b){
    T tmp = *a;
    *a = *b;
    *b = tmp;
  }

  inline void alignedMalloc(void** p, size_t align, size_t size){
#ifdef _WIN32
    *p = _aligned_malloc(size, align);
#else
    posix_memalign(p, align, size);
#endif
  }

  inline void alignedFree(void* p){
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
  }

  /*Checks if p is (align)-bytes aligned*/
  inline bool alignment(const void* p, unsigned int align){
    const unsigned long addr = (unsigned long)p;
    if(addr % align == 0){
      return true;
    }
    warning("pointer %p not %d bytes aligned, offset = %d", p, align, addr%align);
    return false;
  }

  /*Shortcut macros*/
#ifdef _DEBUG
#ifdef _WIN32
#define PRINT_FUNCTION message("%s", __FUNCTION__);
#else
#define PRINT_FUNCTION message("%s", __PRETTY_FUNCTION__);
#endif
#else
#define PRINT_FUNCTION
#endif

#ifdef _DEBUG
#define DUMPINT(x) message("%s = %d", #x, x);
#else
#define DUMPINT(x)
#endif

#ifdef _DEBUG
#define PRINT(x)                    \
  message("%s =>", #x);             \
  std::cout << (x) << std::endl;
#else
#define PRINT(x)
#endif

  /*Debug macros*/
#ifdef _DEBUG
#define DBG(x) x;
#else
#define DBG(x)
#endif

#ifdef _DEBUG
#define START_DEBUG
#else
#define START_DEBUG if(false){
#endif

#ifdef _DEBUG
#define END_DEBUG
#else
#define END_DEBUG }
#endif

#ifndef CUDA
#define NO_CUDA
#endif

}

#endif/*CGFDEFS_H*/
