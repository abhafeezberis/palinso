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

#include <xmmintrin.h>
#include <string.h>

#include "math/default/default_vector_intrinsics.hpp"

namespace CGF{
  namespace x86_sse2{
    inline void vector4_load(float r[4], const float a[4]){
      default_proc::vector4_load(r, a);
    }

    inline void vector4_load(float r[4], float a, float b, float c, float d){
      __m128 XMM0 = _mm_set_ps(d, c, b, a);
      _mm_store_ps(r, XMM0);
    }

    inline void vector4_load(float r[4], float a){
      __m128 XMM0 = _mm_set_ps1(a);
      _mm_store_ps(r, XMM0);
    }

    inline void vector4_muls(float r[4], const float a[4], float n){
      __m128 XMM0 = _mm_set_ps1(n);
      __m128 XMM1 = _mm_load_ps(a);
      XMM1 = _mm_mul_ps(XMM1, XMM0);
      _mm_store_ps(r, XMM1);
    }

    inline void vector4_mul(float r[4], const float a[4], const float b[4]){
      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);
      XMM0 = _mm_mul_ps(XMM0, XMM1);
      _mm_store_ps(r, XMM0);
    }

    inline void vector4_divs(float r[4], const float a[4], float n){
      __m128 XMM0 = _mm_set_ps1(n);
      __m128 XMM1 = _mm_load_ps(a);
      XMM1 = _mm_div_ps(XMM1, XMM0);
      _mm_store_ps(r, XMM1);
    }

    inline void vector4_div(float r[4], const float a[4], const float b[4]){
      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);
      XMM1 = _mm_div_ps(XMM0, XMM1);
      _mm_store_ps(r, XMM1);
    }

    inline void vector4_add(float r[4], const float a[4], const float b[4]){
      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);
      XMM1 = _mm_add_ps(XMM0, XMM1);
      _mm_store_ps(r, XMM1);
    }

    inline void vector4_sub(float r[4], const float a[4], const float b[4]){
      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);
      XMM1 = _mm_sub_ps(XMM0, XMM1);
      _mm_store_ps(r, XMM1);
    }

    inline float vector4_dot(const float a[4], const float b[4]){
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

    inline void vector4_eq(float r[4], const float a[4], const float b[4]){
      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);
      __m128 XMM8 = _mm_set_ps1(1.0f);
      
      XMM0 = _mm_and_ps(_mm_cmpeq_ps(XMM0, XMM1), XMM8);
      _mm_store_ps(r, XMM0);
    }

    inline void vector4_neq(float r[4], const float a[4], const float b[4]){
      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);
      __m128 XMM8 = _mm_set_ps1(1.0f);
      
      XMM0 = _mm_and_ps(_mm_cmpneq_ps(XMM0, XMM1), XMM8);
      _mm_store_ps(r, XMM0);
    }

    inline void vector4_lt(float r[4], const float a[4], const float b[4]){
      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);
      __m128 XMM8 = _mm_set_ps1(1.0f);
      
      XMM0 = _mm_and_ps(_mm_cmplt_ps(XMM0, XMM1), XMM8);
      _mm_store_ps(r, XMM0);
    }

    inline void vector4_le(float r[4], const float a[4], const float b[4]){
      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);
      __m128 XMM8 = _mm_set_ps1(1.0f);
      
      XMM0 = _mm_and_ps(_mm_cmple_ps(XMM0, XMM1), XMM8);
      _mm_store_ps(r, XMM0);
    }

    inline void vector4_gt(float r[4], const float a[4], const float b[4]){
      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);
      __m128 XMM8 = _mm_set_ps1(1.0f);
      
      XMM0 = _mm_and_ps(_mm_cmpgt_ps(XMM0, XMM1), XMM8);
      _mm_store_ps(r, XMM0);
    }

    inline void vector4_ge(float r[4], const float a[4], const float b[4]){
      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);
      __m128 XMM8 = _mm_set_ps1(1.0f);
      
      XMM0 = _mm_and_ps(_mm_cmpge_ps(XMM0, XMM1), XMM8);
      _mm_store_ps(r, XMM0);
    }
    
    inline void vector4_max(float r[4], const float a[4], const float b[4]){
      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);
      
      XMM0 = _mm_max_ps(XMM0, XMM1);
      
      _mm_store_ps(r, XMM0);
    }

    inline void vector4_min(float r[4], const float a[4], const float b[4]){
      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(b);
      
      XMM0 = _mm_min_ps(XMM0, XMM1);
      
      _mm_store_ps(r, XMM0);
    }
  }
}

#endif
