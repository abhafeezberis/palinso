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

#include <sys/time.h>
#include <sys/types.h>
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
  extern CGFAPI void init(uint daemon=0);
  extern CGFAPI void destroy();
  extern CGFAPI ThreadPool* getThreadPool(uint i);
  extern CGFAPI uint cgfrandom(uint& seed);
  extern CGFAPI uint cgfmalloc(void** ptr, ulong size);
  extern CGFAPI void cgffree(void** ptr);
  extern CGFAPI void gettimediffstring(char* buffer, timeval diff);
  extern CGFAPI void swapOutputStream();
  extern CGFAPI void warning(const char* format = "", ... );
  extern CGFAPI void message(const char* format = "", ... );
  extern CGFAPI void error(const char* format = "", ... );
  extern CGFAPI void _cgfassert(const char* expr, const char* filename, 
				   uint line);
  extern CGFAPI void msleep(uint n);
  extern CGFAPI void usleep(uint n);
  extern CGFAPI uint strhash(const char* string);

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

  /*Shortcut macros*/
#ifdef _DEBUG
#define PRINT_FUNCTION message("%s", __PRETTY_FUNCTION__);
#else
#define PRINT_FUNCTION
#endif

#ifdef _DEBUG
#define DUMPINT(x) message("%s = %d", #x, x);
#else
#define DUMPINT(x)
#endif

#ifdef _DEBUG
#define PRINT(x)				\
  message("%s =>", #x);				\
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
}

#endif/*CGFDEFS_H*/
