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

#ifndef DEFAULT_VECTOR_INTRINSICS_HPP
#define DEFAULT_VECTOR_INTRINSICS_HPP

#include <string.h>

namespace CGF{
  namespace default_proc{
    inline void vector4_load(float r[4], const float a[4]){
      memcpy(r, a, sizeof(float)*4);
    }
    
    inline void vector4_load(float r[4], float a, float b, float c, float d){
      r[0] = a;
      r[1] = b;
      r[2] = c;
      r[3] = d;
    }

    inline void vector4_load(float r[4], float a){
      r[0] = a;
      r[1] = a;
      r[2] = a;
      r[3] = a;
    }

    inline void vector4_muls(float r[4], const float a[4], float n){
      r[0] = a[0] * n;
      r[1] = a[1] * n;
      r[2] = a[2] * n;
      r[3] = a[3] * n; 
    }

    inline void vector4_mul(float r[4], const float a[4], const float b[4]){
      r[0] = a[0] * b[0];
      r[1] = a[1] * b[1];
      r[2] = a[2] * b[2];
      r[3] = a[3] * b[3]; 
    }

    inline void vector4_divs(float r[4], const float a[4], float n){
      r[0] = a[0] / n;
      r[1] = a[1] / n;
      r[2] = a[2] / n;
      r[3] = a[3] / n;
    }

    inline void vector4_div(float r[4], const float a[4], const float b[4]){
      r[0] = a[0] / b[0];
      r[1] = a[1] / b[1];
      r[2] = a[2] / b[2];
      r[3] = a[3] / b[3]; 
    }

    inline void vector4_add(float r[4], const float a[4], const float b[4]){
      r[0] = a[0] + b[0];
      r[1] = a[1] + b[1];
      r[2] = a[2] + b[2];
      r[3] = a[3] + b[3];
    }

    inline void vector4_sub(float r[4], const float a[4], const float b[4]){
      r[0] = a[0] - b[0];
      r[1] = a[1] - b[1];
      r[2] = a[2] - b[2];
      r[3] = a[3] - b[3];
    }

    /*dot and sum are computed in the same order as in the sse2 equivalent*/
    inline float vector4_dot(const float a[4], const float b[4]){
      return ((a[0]*b[0]) + (a[2]*b[2])) + 
	((a[1]*b[1]) + (a[3]*b[3]));
    } 

    inline float vector4_sum(const float a[4]){
      return (a[0] + a[3]) + (a[1] + a[3]);
    } 

    inline void vector4_cross(float r[4], const float a[4], const float b[4]){
      r[0] = a[1]*b[2] - a[2]*b[1];
      r[1] = a[2]*b[0] - a[0]*b[2];
      r[2] = a[0]*b[1] - a[1]*b[0];
      r[3] = 0;
    }

    inline void vector4_eq(float r[4], const float a[4], const float b[4]){
      r[0] = (a[0] == b[0]);
      r[1] = (a[1] == b[1]);
      r[2] = (a[2] == b[2]);
      r[3] = (a[3] == b[3]);
    }

   inline void vector4_neq(float r[4], const float a[4], const float b[4]){
      r[0] = (a[0] != b[0]);
      r[1] = (a[1] != b[1]);
      r[2] = (a[2] != b[2]);
      r[3] = (a[3] != b[3]);
    }

   inline void vector4_lt(float r[4], const float a[4], const float b[4]){
      r[0] = (a[0] < b[0]);
      r[1] = (a[1] < b[1]);
      r[2] = (a[2] < b[2]);
      r[3] = (a[3] < b[3]);
    }

   inline void vector4_le(float r[4], const float a[4], const float b[4]){
      r[0] = (a[0] <= b[0]);
      r[1] = (a[1] <= b[1]);
      r[2] = (a[2] <= b[2]);
      r[3] = (a[3] <= b[3]);
    }

   inline void vector4_gt(float r[4], const float a[4], const float b[4]){
      r[0] = (a[0] > b[0]);
      r[1] = (a[1] > b[1]);
      r[2] = (a[2] > b[2]);
      r[3] = (a[3] > b[3]);
    }

   inline void vector4_ge(float r[4], const float a[4], const float b[4]){
      r[0] = (a[0] >= b[0]);
      r[1] = (a[1] >= b[1]);
      r[2] = (a[2] >= b[2]);
      r[3] = (a[3] >= b[3]);
    }

    inline void vector4_max(float r[4], const float a[4], const float b[4]){
      r[0] = (a[0] > b[0])?a[0]:b[0];
      r[1] = (a[1] > b[1])?a[1]:b[1];
      r[2] = (a[2] > b[2])?a[2]:b[2];
      r[3] = (a[3] > b[3])?a[3]:b[3];
    }

    inline void vector4_min(float r[4], const float a[4], const float b[4]){
      r[0] = (a[0] < b[0])?a[0]:b[0];
      r[1] = (a[1] < b[1])?a[1]:b[1];
      r[2] = (a[2] < b[2])?a[2]:b[2];
      r[3] = (a[3] < b[3])?a[3]:b[3];
    }
  }
}

#endif//DEFAULT_VECTOR_INTRINSICS_HPP
