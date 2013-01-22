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

#ifndef MATH_HPP
#define MATH_HPP
#include <math.h>
#include "core/cgfdefs.hpp"

#ifndef PI
#define PI 3.14159265358979323846

#endif
#ifndef H_PI
#define H_PI 1.570796327
#endif
#ifndef D_PI
#define D_PI 6.283185307
#endif

#ifndef EULER
#define EULER 2.71828183
#endif

#define FLOAT_MAX  1e+100
#define FLOAT_MIN -1e+100

namespace CGF{
  inline double log2(double n){
    return log(n) / log(2);
  }

  inline double sign(double n){
    if(n>0)return 1;
    return -1;
  }

  inline double sqrd(double d){
    return d*d;
  }

  inline float sqrf(float f){
    return f*f;
  }

  inline float maxf(float a, float b){
    if(a>b)
      return a;
    return b;
  }

  inline float minf(float a, float b){
    if(a>b)
      return b;
    return a;
  }

  inline float posf(float a){
    return maxf(a,0);
  }

  inline float negf(float a){
    return minf(a,0);
  }

  inline uint maxui(uint a, uint b){
    if(a>b)
      return a;
    return b;
  }

  inline uint minui(uint a, uint b){
    if(a>b)
      return b;
    return a;
  }

  inline int maxi(int a, int b){
    if(a>b)
      return a;
    return b;
  }

  inline int mini(int a, int b){
    if(a>b)
      return b;
    return a;
  }

  inline double maxd(double a, double b){
    if(a>b)
      return a;
    return b;
  }

  inline double mind(double a, double b){
    if(a>b)
      return b;
    return a;
  }

  inline float round(float f){
    //f *= 10;
    if(sqrf(f) < 1E-5){
      return 0;
    }else if(f > 0){
      return ceil(f);
    }else{
      return floor(f);
    }
  }

  template<class T>
  inline T linterp(float iso, T* d1, T* d2, float val1, float val2, 
		   float tol = 1E-40){
    T *op1, *op2;
    float opv1, opv2;
    if(val1 > val2){
      op1 = d1;
      op2 = d2;
      opv1 = val1;
      opv2 = val2;
    }else{
      op1 = d2;
      op2 = d1;
      opv1 = val2;
      opv2 = val1;
    }

    float mu;
    
    //message("iso - opv1 = %10.10e", iso - opv1);
    if(fabs(iso - opv1) < tol)
      return *op1;
    //message("iso - opv2 = %10.10e", iso - opv2);
    if(fabs(iso - opv2) < tol)
      return *op2;
    //message("opv1 - opv2 = %10.10e", opv1 - opv2);
    if(fabs(opv1 - opv2) < tol)
      return *op1;

    mu = (iso - opv1)/(opv2 - opv1);
    //message("mu = %10.10e", mu);
    return *op1 + mu * (*op2 - *op1);
  }

  template<class T>
  void machine_epsilon(){
    T mach_eps = 1.0f;
    do{
      mach_eps /= 2.0f;
    }while((T)(1.0 + (mach_eps/2.0)) != 1.0);
    message("Machine epsilon = %10.10e", mach_eps);
  }
  

  struct _VectorRange{
    uint startBlock;
    uint endBlock;
    uint range;
  };

  typedef struct _VectorRange VectorRange;

  struct _MatrixRange{
    uint startRow;
    uint endRow;
    uint range;
  };

  typedef struct _MatrixRange MatrixRange;
}

#endif/*MATH_HPP*/
