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

#include <limits>

#ifdef _WIN32
#define isnan(x) _isnan((x))
#endif

//#define FLOAT_MAX  1e+100
//#define FLOAT_MIN -1e+100
#define FLOAT_MAX std::numeric_limits<float>::max()
#define FLOAT_MIN std::numeric_limits<float>::min()

namespace CGF{

#if 0

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
#endif


#if 0
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
#else
  template<class T>
  inline T Min(const T& a, const T& b){
    if(a>b){
      return b;
    }
    return a;
  }

  template<class T>
  inline T Max(const T& a, const T& b){
    if(a>b){
      return a;
    }
    return b;
  }

  template<class T>
  inline T Pos(T a){
    return Max(a,(T)0);
  }

  template<class T>
  inline T Neg(T a){
    return Min(a,(T)0);
  }


#endif

  template<class T>
  inline T Clamp(T a, T min, T max){
    //return a;
    if(a < min){
      return min;
    }else if(a > max){
      return max;
    }
    return a;
  }

  template<class T>
  inline T Sqr(T f){
    error("Sqr applied on non a non floating point value");
    return (T)0;
  }

  template<>
  inline float Sqr(float f){
    return f*f;
  }

  template<>
  inline double Sqr(double d){
    return d*d;
  }

  template<>
  inline long double Sqr(long double d){
    return d*d;
  }

  template<class T>
  inline T Sqrt(T f){
    error("Sqrt applied on non a non floating point value");
    return (T)0;
  }

  template<>
  inline float Sqrt(float f){
    return sqrtf(f);
  }

  template<>
  inline double Sqrt(double d){
    return sqrt(d);
  }

  template<>
  inline long double Sqrt(long double d){
    return sqrtl(d);
  }

  template<class T>
  inline T Fabs(T v){
    error("Fabs applied on non a non floating point value");
    return (T)0;
  }

  template<>
  inline float Fabs(float v){
    return fabsf(v);
  }

  template<>
  inline double Fabs(double v){
    return fabs(v);
  }

  template<>
  inline long double Fabs(long double v){
    return fabsl(v);
  }

  template<class T>
  inline T Sign(T a, T b){
    return b >= 0.0 ? Fabs(a) : -Fabs(a);
  }

  template<class T>
  inline T Sign(T n){
    return Sign((T)1.0, n);
  }

  template<class T>
  inline T Log(T v){
    error("Log applied on non a non floating point value");
    return (T)0;
  }

  template<>
  inline float Log(float v){
    return logf(v);
  }

  template<>
  inline double Log(double v){
    return log(v);
  }

  template<>
  inline long double Log(long double v){
    return logl(v);
  }

  template<class T>
  inline T Ceil(T v){
    error("Ceil applied on non a non floating point value");
    return (T)0;
  }

  template<>
  inline float Ceil(float v){
    return ceilf(v);
  }

  template<>
  inline double Ceil(double v){
    return ceil(v);
  }

  template<>
  inline long double Ceil(long double v){
    return ceill(v);
  }

  template<class T>
  inline T Exp(T v){
    error("Exp applied on non a non floating point value");
    return (T)0;
  }

  template<>
  inline float Exp(float v){
    return expf(v);
  }

  template<>
  inline double Exp(double v){
    return exp(v);
  }

  template<>
  inline long double Exp(long double v){
    return expl(v);
  }

  template<class T>
  inline T Round(T v){
    error("Round applied on a non floating point type");
  }

  template<>
  inline float Round(float v){
    return roundf(v);
  }

  template<>
  inline double Round(double v){
    return round(v);
  }

  template<>
  inline long double Round(long double v){
    return roundl(v);
  }

  template<class T>
  inline T Sin(T v){
    error("Sin applied on non a non floating point value");
    return (T)0;
  }

  template<>
  inline float Sin(float v){
    return sinf(v);
  }

  template<>
  inline double Sin(double v){
    return sin(v);
  }

  template<>
  inline long double Sin(long double v){
    return sinl(v);
  }

  template<class T>
  inline T Asin(T v){
    error("Asin applied on non a non floating point value");
    return (T)0;
  }

  template<>
  inline float Asin(float v){
    return asinf(v);
  }

  template<>
  inline double Asin(double v){
    return asin(v);
  }

  template<>
  inline long double Asin(long double v){
    return asinl(v);
  }


  template<class T>
  inline T Cos(T v){
    error("Cos applied on non a non floating point value");
    return (T)0;
  }

  template<>
  inline float Cos(float v){
    return cosf(v);
  }

  template<>
  inline double Cos(double v){
    return cos(v);
  }

  template<>
  inline long double Cos(long double v){
    return cosl(v);
  }

  template<class T>
  inline T Acos(T v){
    error("Acos applied on non a non floating point value");
    return (T)0;
  }

  template<>
  inline float Acos(float v){
    return acosf(v);
  }

  template<>
  inline double Acos(double v){
    return acos(v);
  }

  template<>
  inline long double Acos(long double v){
    return acosl(v);
  }

  template<class T>
  inline T Floor(T v){
    error("Floor applied on non a non floating point value");
    return (T)0;
  }

  template<>
  inline float Floor(float v){
    return floorf(v);
  }

  template<>
  inline double Floor(double v){
    return floor(v);
  }

  template<>
  inline long double Floor(long double v){
    return floorl(v);
  }

  template<class T>
  inline T Log2(T n){
    return Log(n) / Log((T)2.0);
  }


  template<class T>
  inline T Pow(T v, T p){
    error("Pow applied on non a non floating point value");
    return (T)0;
  }

  template<>
  inline float Pow(float v, float p){
    return powf(v, p);
  }

  template<>
  inline double Pow(double v, double p){
    return pow(v, p);
  }

  template<>
  inline long double Pow(long double v, long double p){
    return powl(v, p);
  }

  template<class T>
  inline T Coth(T v){
    error("Coth applied on non float value");
  }

  template<>
  inline float Coth(float v){
    float t = Pow((float)EULER, v);
    return (t + (float)1.0/t) / (t - (float)1.0 / t);
  }

  template<>
  inline double Coth(double v){
    double t = Pow(EULER, v);
    return (t + 1.0/t) / (t - 1.0 / t);
  }

  template<>
  inline long double Coth(long double v){
    long double t = Pow((long double)EULER, v);
    return (t + 1.0/t) / (t - 1.0 / t);
  }

  template<class T, class Y>
  inline Y linterp(T iso, Y* d1, Y* d2, T val1, T val2,
                   T tol = (T)1E-40){
    Y *op1, *op2;
    T opv1, opv2;
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

    T mu;

    //message("iso - opv1 = %10.10e", iso - opv1);
    if(Fabs(iso - opv1) < tol)
      return *op1;
    //message("iso - opv2 = %10.10e", iso - opv2);
    if(Fabs(iso - opv2) < tol)
      return *op2;
    //message("opv1 - opv2 = %10.10e", opv1 - opv2);
    if(Fabs(opv1 - opv2) < tol)
      return *op1;

    mu = (iso - opv1)/(opv2 - opv1);
    //message("mu = %10.10e", mu);
    return *op1 + mu * (*op2 - *op1);
  }

#if 1
  template<class T, class X, class Y>
  class Interpolator{
  public:
    void init(){

    }

    T evaluate(T){
      return 0.0;
    }

    void getResult(T a, X* b, Y* c){

    }
  };

#if 0
  template<class T, class X, class Y>
  T bisection(Interpolator<T, X, Y>* func, T x1, T x2, T tol){
    T dx, f, fmid, xmid, rtb;

    func->init();

    f = func->evaluate(x1);
    fmid = func->evaluate(x2);

    if(f*fmid >= (T)0.0){
      error("Root must be bracketed");
    }

    rtb = f < (T)0.0 ? (dx = x2-x1, x1) : (dx=x1-d2, x2);

    for(int i = 0;i<100;i++){
      fmid = func->evaluate(xmid = rtb + (dx*=(T)0.5));
      if(fmid <= (T)0.0){
        rtb = xmid;
      }

      if(Fabs(dx) < tol || fmid == (T)0.0){
        return rtb;
      }
    }
    error("Too many bisection iterations");
    return (T)0.0;
  }
#endif

  template<class T, class X, class Y>
  T brentDekker(Interpolator<T, X, Y>* func, T x1, T x2, T tol){
    int iter = 0;
    T a = x1;
    T b = x2;
    T c = x2;
    T d = 0, e = 0, min1, min2;

    T meps = (T)5e-8;

    func->init();

    T fa = func->evaluate(a);
    T fb = func->evaluate(b);
    T fc, p, q, r, s, tol1, xm;

    if( (fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0)){
      error("Root is not bracketed, %10.10e, %10.10e", fa, fb);
    }

    fc = fb;

    for(iter=0;iter<100;iter++){
      if( (fb > 0.0 && fc > 0.0) ||
          (fb < 0.0 && fc < 0.0)){
        c  = a;
        fc = fa;
        e  = d = b - a;
      }

      if(Fabs(fc) < Fabs(fb)){
        a = b;
        b = c;
        c = a;
        fa = fb;
        fb = fc;
        fc = fa;
      }

      tol1 = (T)2.0 * meps * Fabs(b) + (T)0.5*tol;
      xm   = (T)0.5 * (c-b);

      if(Fabs(xm) <= tol1 || fb == 0.0){
        return b;
      }

      if(Fabs(e) >= tol1 && Fabs(fa) > Fabs(fb)){
        /*Inverse quadratic interpolation*/
        s = fb/fa;

        if(a == c){
          p = (T)2.0*xm*s;
          q = (T)1.0 - s;
        }else{
          q = fa/fc;
          r = fb/fc;
          p = s*( (T)2.0*xm*q*(q-r) - (b-a)*(r-(T)1.0) );
          q = (q-(T)1.0)*(r-(T)1.0)*(s-(T)1.0);
        }

        if(p > 0.0){
          q = -q;
        }

        p = Fabs(p);

        min1 = (T)3.0*xm*q-Fabs(tol1*q);
        min2 = Fabs(e*q);

        if((T)2.0*p < (min1 < min2 ? min1 : min2)){
          /*Accept interpolation*/
          e = d;
          d = p/q;
        }else{
          /*Interpolation failed, use bisection*/
          d = xm;
          e = d;
        }
      }else{
        /*Bounds decreasing too slowly, use bisection*/
        d = xm;
        e = d;
      }

      a = b;
      fa = fb;
      if( Fabs(d) > tol1){
        b += d;
      }else{
        b += Sign(tol1, xm);
      }
      fb = func->evaluate(b);
    }

    error("Maximum numbers of iterations");
    return 0.0;

  }
#endif

  template<class T>
  void machine_epsilon(){
    T mach_eps = 1.0f;
    do{
      mach_eps /= 2.0f;
    }while((T)(1.0 + (mach_eps/2.0)) != 1.0);
    message("Machine epsilon = %10.10e", mach_eps/2.0);
  }


  struct _VectorRange{
    int startBlock;
    int endBlock;
    int range;
  };

  typedef struct _VectorRange VectorRange;

  struct _MatrixRange{
    int startRow;
    int endRow;
    int range;
  };

  typedef struct _MatrixRange MatrixRange;
}

#endif/*MATH_HPP*/
