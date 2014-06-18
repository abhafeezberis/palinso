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

#ifndef SSE2_VECTOR_INTRINSICS
#define SSE2_VECTOR_INTRINSICS

#include "core/cgfdefs.hpp"

#include <xmmintrin.h>
#ifdef _WIN32
#include <emmintrin.h>
#define _mm_set_pd1 _mm_set1_pd
#endif
#include <string.h>

#include "math/default/default_vector_intrinsics.hpp"

namespace CGF{
  namespace x86_sse2{
    template<class T>
    inline void vector4_load(T r[4], const T a[4]){
      default_proc::vector4_load(r, a);
    }

    template<class T>
    inline void vector4_load(T r[4], const T& a, const T& b,
                             const T& c, const T& d){
      default_proc::vector4_load(r, a);
    }

    template<>
    inline void vector4_load(float r[4], const float& a, const float& b,
                             const float & c, const float & d){
      cgfassert(alignment(r, 16));

      __m128 XMM0 = _mm_set_ps(d, c, b, a);
      _mm_store_ps(r, XMM0);
    }

    template<>
    inline void vector4_load(double r[4], const double&  a, const double& b,
                             const double& c, const double& d){
      cgfassert(alignment(r, 16));

      __m128d XMM0 = _mm_set_pd(b, a);
      __m128d XMM1 = _mm_set_pd(d, c);
      _mm_store_pd(r  , XMM0);
      _mm_store_pd(r+2, XMM1);
    }

    template<class T>
    inline void vector4_load(T r[4], const T& a){
      default_proc::vector4_load(r, a);
    }

    template<>
    inline void vector4_load(float r[4], const float& a){
      cgfassert(alignment(r, 16));

      __m128 XMM0 = _mm_set_ps1(a);
      _mm_store_ps(r, XMM0);
    }

    template<>
    inline void vector4_load(double r[4], const double& a){
      cgfassert(alignment(r, 16));

      __m128d XMM0 = _mm_set_pd1(a);
      _mm_store_pd(r+0, XMM0);
      _mm_store_pd(r+2, XMM0);
    }

    template<class T>
    inline void vector4_muls(T r[4], const T a[4], const T& n){
      default_proc::vector4_muls(r, a, n);
    }

    template<>
    inline void vector4_muls(float r[4], const float a[4], const float& n){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));

      __m128 XMM0 = _mm_set_ps1(n);
      __m128 XMM1 = _mm_load_ps(a);
      XMM1 = _mm_mul_ps(XMM1, XMM0);
      _mm_store_ps(r, XMM1);
    }

    template<>
    inline void vector4_muls(double r[4], const double a[4], const double& n){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));

      __m128d XMM0 = _mm_set_pd1(n);
      __m128d XMM1 = _mm_load_pd(a);
      __m128d XMM2 = _mm_load_pd(a+2);
      _mm_store_pd(r+0, _mm_mul_pd(XMM1, XMM0));
      _mm_store_pd(r+2, _mm_mul_pd(XMM2, XMM0));
    }

    template<class T>
    inline void vector4_mul(T r[4], const T a[4], const T b[4]){
      default_proc::vector4_mul(r, a, b);
    }

    template<>
    inline void vector4_mul(float r[4], const float a[4], const float b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);
      XMM0 = _mm_mul_ps(XMM0, XMM1);
      _mm_store_ps(r, XMM0);
    }

    template<>
    inline void vector4_mul(double r[4], const double a[4], const double b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128d XMM0 = _mm_load_pd(a+0);
      __m128d XMM1 = _mm_load_pd(b+0);
      XMM0 = _mm_mul_pd(XMM0, XMM1);
      _mm_store_pd(r+0, XMM0);

      XMM0 = _mm_load_pd(a+2);
      XMM1 = _mm_load_pd(b+2);
      XMM0 = _mm_mul_pd(XMM0, XMM1);
      _mm_store_pd(r+2, XMM0);
    }

    template<class T>
    inline void vector4_divs(T r[4], const T a[4], const T& n){
      default_proc::vector4_divs(r, a, n);
    }

    template<>
    inline void vector4_divs(float r[4], const float a[4], const float& n){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));

      __m128 XMM0 = _mm_set_ps1(n);
      __m128 XMM1 = _mm_load_ps(a);
      XMM1 = _mm_div_ps(XMM1, XMM0);
      _mm_store_ps(r, XMM1);
    }

    template<>
    inline void vector4_divs(double r[4], const double a[4], const double& n){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));

      __m128d XMM0 = _mm_set_pd1(n);
      __m128d XMM1 = _mm_load_pd(a+0);
      XMM1 = _mm_div_pd(XMM1, XMM0);
      _mm_store_pd(r+0, XMM1);

      XMM1 = _mm_load_pd(a+2);
      XMM1 = _mm_div_pd(XMM1, XMM0);
      _mm_store_pd(r+2, XMM1);
    }

    template<class T>
    inline void vector4_div(T r[4], const T a[4], const T b[4]){
      default_proc::vector4_div(r, a, b);
    }

    template<>
    inline void vector4_div(float r[4], const float a[4], const float b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);
      XMM1 = _mm_div_ps(XMM0, XMM1);
      _mm_store_ps(r, XMM1);
    }

    template<>
    inline void vector4_div(double r[4], const double a[4], const double b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128d XMM0 = _mm_load_pd(a+0);
      __m128d XMM1 = _mm_load_pd(b+0);
      XMM1 = _mm_div_pd(XMM0, XMM1);
      _mm_store_pd(r+0, XMM1);

      XMM0 = _mm_load_pd(a+2);
      XMM1 = _mm_load_pd(b+2);
      XMM1 = _mm_div_pd(XMM0, XMM1);
      _mm_store_pd(r+2, XMM1);
    }

    template<class T>
    inline void vector4_add(T r[4], const T a[4], const T b[4]){
      default_proc::vector4_add(r, a, b);
    }

    template<>
    inline void vector4_add(float r[4], const float a[4], const float b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);
      XMM1 = _mm_add_ps(XMM0, XMM1);
      _mm_store_ps(r, XMM1);
    }

    template<>
    inline void vector4_add(double r[4], const double a[4], const double b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128d XMM0 = _mm_load_pd(a+0);
      __m128d XMM1 = _mm_load_pd(b+0);
      XMM1 = _mm_add_pd(XMM0, XMM1);
      _mm_store_pd(r+0, XMM1);

      XMM0 = _mm_load_pd(a+2);
      XMM1 = _mm_load_pd(b+2);
      XMM1 = _mm_add_pd(XMM0, XMM1);
      _mm_store_pd(r+2, XMM1);
    }


    template<class T>
    inline void vector4_sub(T r[4], const T a[4], const T b[4]){
      default_proc::vector4_sub(r, a, b);
    }

    template<>
    inline void vector4_sub(float r[4], const float a[4], const float b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);
      XMM1 = _mm_sub_ps(XMM0, XMM1);
      _mm_store_ps(r, XMM1);
    }

    template<>
    inline void vector4_sub(double r[4], const double a[4], const double b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128d XMM0 = _mm_load_pd(a+0);
      __m128d XMM1 = _mm_load_pd(b+0);
      XMM1 = _mm_sub_pd(XMM0, XMM1);
      _mm_store_pd(r+0, XMM1);

      XMM0 = _mm_load_pd(a+2);
      XMM1 = _mm_load_pd(b+2);
      XMM1 = _mm_sub_pd(XMM0, XMM1);
      _mm_store_pd(r+2, XMM1);
    }

    template<class T>
    inline T vector4_dot(const T a[4], const T b[4]){
      return default_proc::vector4_dot(a, b);
    }

#if 0
    template<>
    inline float vector4_dot(const float a[4], const float b[4]){
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      float r[4];
      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);
      XMM0 = _mm_mul_ps(XMM0, XMM1);                           //abcd
      XMM1 = _mm_shuffle_ps(XMM0, XMM0, _MM_SHUFFLE(3,2,3,2)); //cdcd
      XMM0 = _mm_add_ps(XMM0, XMM1);                           //a+c b+d c+c d+d
      XMM1 = _mm_shuffle_ps(XMM0, XMM0, _MM_SHUFFLE(1,1,1,1)); //b+d b+d b+d b+d
      XMM0 = _mm_add_ps(XMM0, XMM1);                           //(a+c)+(b+d) ..
      _mm_store_ps(r, XMM0);
      return r[0];
    }
#endif

    template<class T>
    inline T vector4_sum(const T a[4]){
      return default_proc::vector4_sum(a);
    }

#if 0
    template<>
    inline float vector4_sum(const float a[4]){
      float r[4];
      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_shuffle_ps(XMM0, XMM0, _MM_SHUFFLE(3,2,3,2));
      XMM0 = _mm_add_ps(XMM0, XMM1);
      XMM1 = _mm_shuffle_ps(XMM0, XMM0, _MM_SHUFFLE(1,1,1,1));
      XMM0 = _mm_add_ps(XMM0, XMM1);
      _mm_store_ps(r, XMM0);
      return r[0];
    }
#endif

    template<class T>
    inline void vector4_cross(T r[4], const T a[4], const T b[4]){
      default_proc::vector4_cross(r, a, b);
    }

#if 0
    template<>
    inline void vector4_cross(float r[4], const float a[4], const float b[4]){
      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);

      __m128 XMM2 = _mm_shuffle_ps(XMM0, XMM0, _MM_SHUFFLE(3,0,2,1));
      __m128 XMM3 = _mm_shuffle_ps(XMM1, XMM1, _MM_SHUFFLE(3,1,0,2));
      __m128 XMM4 = _mm_shuffle_ps(XMM0, XMM0, _MM_SHUFFLE(3,1,0,2));
      __m128 XMM5 = _mm_shuffle_ps(XMM1, XMM1, _MM_SHUFFLE(3,0,2,1));

      XMM0 = _mm_mul_ps(XMM2, XMM3);
      XMM1 = _mm_mul_ps(XMM4, XMM5);
      XMM0 = _mm_sub_ps(XMM0, XMM1);
      _mm_store_ps(r, XMM0);
    }
#endif


    static const __m128 cones = {1, 1, 1, 1};

    template<class T>
    inline void vector4_eq(T r[4], const T a[4], const T b[4]){
      default_proc::vector4_eq(r, a, b);
    }

    template<>
    inline void vector4_eq(float r[4], const float a[4], const float b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);

      XMM0 = _mm_and_ps(_mm_cmpeq_ps(XMM0, XMM1), cones);
      _mm_store_ps(r, XMM0);
    }

    template<class T>
    inline void vector4_neq(T r[4], const T a[4], const T b[4]){
      default_proc::vector4_neq(r, a, b);
    }

    template<>
    inline void vector4_neq(float r[4], const float a[4], const float b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);

      XMM0 = _mm_and_ps(_mm_cmpneq_ps(XMM0, XMM1), cones);
      _mm_store_ps(r, XMM0);
    }

    template<class T>
    inline void vector4_lt(T r[4], const T a[4], const T b[4]){
      default_proc::vector4_lt(r, a, b);
    }

    template<>
    inline void vector4_lt(float r[4], const float a[4], const float b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);

      XMM0 = _mm_and_ps(_mm_cmplt_ps(XMM0, XMM1), cones);
      _mm_store_ps(r, XMM0);
    }

    template<class T>
    inline void vector4_le(T r[4], const T a[4], const T b[4]){
      default_proc::vector4_le(r, a, b);
    }

    template<>
    inline void vector4_le(float r[4], const float a[4], const float b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);

      XMM0 = _mm_and_ps(_mm_cmple_ps(XMM0, XMM1), cones);
      _mm_store_ps(r, XMM0);
    }

    template<class T>
    inline void vector4_gt(T r[4], const T a[4], const T b[4]){
      default_proc::vector4_gt(r, a, b);
    }

    template<>
    inline void vector4_gt(float r[4], const float a[4], const float b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);

      XMM0 = _mm_and_ps(_mm_cmpgt_ps(XMM0, XMM1), cones);
      _mm_store_ps(r, XMM0);
    }

    template<class T>
    inline void vector4_ge(T r[4], const T a[4], const T b[4]){
      default_proc::vector4_ge(r, a, b);
    }

    template<>
    inline void vector4_ge(float r[4], const float a[4], const float b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);

      XMM0 = _mm_and_ps(_mm_cmpge_ps(XMM0, XMM1), cones);
      _mm_store_ps(r, XMM0);
    }

    template<class T>
    inline void vector4_max(T r[4], const T a[4], const T b[4]){
      default_proc::vector4_max(r, a, b);
    }

    template<>
    inline void vector4_max(float r[4], const float a[4], const float b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);

      XMM0 = _mm_max_ps(XMM0, XMM1);

      _mm_store_ps(r, XMM0);
    }

    template<>
    inline void vector4_max(double r[4], const double a[4], const double b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128d XMM00 = _mm_load_pd(a+0);
      __m128d XMM01 = _mm_load_pd(a+2);

      __m128d XMM10 = _mm_load_pd(b+0);
      __m128d XMM11 = _mm_load_pd(b+2);

      XMM00 = _mm_max_pd(XMM00, XMM10);
      XMM01 = _mm_max_pd(XMM01, XMM11);

      _mm_store_pd(r+0, XMM00);
      _mm_store_pd(r+2, XMM01);
    }

    template<class T>
    inline void vector4_min(T r[4], const T a[4], const T b[4]){
      default_proc::vector4_min(r, a, b);
    }

    template<>
    inline void vector4_min(float r[4], const float a[4], const float b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);

      XMM0 = _mm_min_ps(XMM0, XMM1);

      _mm_store_ps(r, XMM0);
    }

    template<>
    inline void vector4_min(double r[4], const double a[4], const double b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128d XMM00 = _mm_load_pd(a+0);
      __m128d XMM01 = _mm_load_pd(a+2);

      __m128d XMM10 = _mm_load_pd(b+0);
      __m128d XMM11 = _mm_load_pd(b+2);

      XMM00 = _mm_min_pd(XMM00, XMM10);
      XMM01 = _mm_min_pd(XMM01, XMM11);

      _mm_store_pd(r+0, XMM00);
      _mm_store_pd(r+2, XMM01);
    }
  }
}

#endif
