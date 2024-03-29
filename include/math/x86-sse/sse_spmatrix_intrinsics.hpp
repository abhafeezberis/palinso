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

#ifndef SSE2_SPMATRIX_INTRINSICS
#define SSE2_SPMATRIX_INTRINSICS

#include "core/cgfdefs.hpp"

#include <xmmintrin.h>
#ifdef _WIN32
#include <emmintrin.h>
#define _mm_set_pd1 _mm_set1_pd
#endif
#include "math/default/default_spmatrix_intrinsics.hpp"

namespace CGF{
  namespace x86_sse2{
    template<int N, class T>
    inline void spmatrix_block_load(T r[N*N], const T a[N*N]){
      default_proc::spmatrix_block_load<N, T>(r, a);
    }

#define SSE_SCALAR_OPERATION_2(r, a, f, sse_op) \
    __m128 XMM0 = _mm_set_ps1(f);               \
    __m128 XMM1 = _mm_load_ps(a);               \
                                                \
    XMM1 = sse_op(XMM1, XMM0);                  \
    _mm_store_ps(r, XMM1);

#define SSE_SCALAR_OPERATION_2D(r, a, f, sse_op)        \
    __m128d XMM0 = _mm_set_pd1(f);                      \
                                                        \
    _mm_store_pd(r+0, sse_op(_mm_load_pd(a+0), XMM0));	\
    _mm_store_pd(r+2, sse_op(_mm_load_pd(a+2), XMM0));


#define SSE_SCALAR_OPERATION_4(r, a, f, sse_op) \
    __m128 XMM0 = _mm_set_ps1(f);               \
    __m128 XMM1 = _mm_load_ps(a);               \
    __m128 XMM2 = _mm_load_ps(a+4);             \
    __m128 XMM3 = _mm_load_ps(a+8);             \
    __m128 XMM4 = _mm_load_ps(a+12);			\
                                                \
    XMM1 = sse_op(XMM1, XMM0);                  \
    XMM2 = sse_op(XMM2, XMM0);                  \
    XMM3 = sse_op(XMM3, XMM0);                  \
    XMM4 = sse_op(XMM4, XMM0);                  \
                                                \
    _mm_store_ps(r,    XMM1);                   \
    _mm_store_ps(r+4,  XMM2);                   \
    _mm_store_ps(r+8,  XMM3);                   \
    _mm_store_ps(r+12, XMM4);

#define SSE_SCALAR_OPERATION_4D(r, a, f, sse_op)            \
    __m128d XMM0 = _mm_set_pd1(f);                          \
    _mm_store_pd(r+0, sse_op(_mm_load_pd(a+0), XMM0));		\
    _mm_store_pd(r+2, sse_op(_mm_load_pd(a+2), XMM0));		\
    _mm_store_pd(r+4, sse_op(_mm_load_pd(a+4), XMM0));		\
    _mm_store_pd(r+6, sse_op(_mm_load_pd(a+6), XMM0));		\
    _mm_store_pd(r+8, sse_op(_mm_load_pd(a+8), XMM0));		\
    _mm_store_pd(r+10, sse_op(_mm_load_pd(a+10), XMM0));	\
    _mm_store_pd(r+12, sse_op(_mm_load_pd(a+12), XMM0));	\
    _mm_store_pd(r+14, sse_op(_mm_load_pd(a+14), XMM0));


#define SSE_SCALAR_OPERATION_8(r, a, f, sse_op) \
    __m128 XMM0 = _mm_set_ps1(f);               \
    for(uint j=0;j<4;j++){                      \
      uint idx = j * 16;                        \
      __m128 XMM1 = _mm_load_ps(a+idx);			\
      __m128 XMM2 = _mm_load_ps(a+idx+4);		\
      __m128 XMM3 = _mm_load_ps(a+idx+8);		\
      __m128 XMM4 = _mm_load_ps(a+idx+12);		\
                                                \
      XMM1 = sse_op(XMM1, XMM0);                \
      XMM2 = sse_op(XMM2, XMM0);                \
      XMM3 = sse_op(XMM3, XMM0);                \
      XMM4 = sse_op(XMM4, XMM0);                \
                                                \
      _mm_store_ps(r+idx,    XMM1);             \
      _mm_store_ps(r+idx+4,  XMM2);             \
      _mm_store_ps(r+idx+8,  XMM3);             \
      _mm_store_ps(r+idx+12, XMM4);             \
    }

#define SSE_SCALAR_OPERATION_8D(r, a, f, sse_op)                \
    __m128d XMM0 = _mm_set_pd1(f);                              \
    for(uint j=0;j<32;j++){                                     \
      _mm_store_pd(r+j*2, sse_op(_mm_load_pd(a+j*2), XMM0));	\
    }

#define SSE_SCALAR_OPERATION_16(r, a, f, sse_op)	\
    __m128 XMM0 = _mm_set_ps1(f);                   \
    for(uint j=0;j<16;j++){                         \
      uint idx = j * 16;                            \
      __m128 XMM1 = _mm_load_ps(a+idx);             \
      __m128 XMM2 = _mm_load_ps(a+idx+4);           \
      __m128 XMM3 = _mm_load_ps(a+idx+8);           \
      __m128 XMM4 = _mm_load_ps(a+idx+12);          \
                                                    \
      XMM1 = sse_op(XMM1, XMM0);                    \
      XMM2 = sse_op(XMM2, XMM0);                    \
      XMM3 = sse_op(XMM3, XMM0);                    \
      XMM4 = sse_op(XMM4, XMM0);                    \
                                                    \
      _mm_store_ps(r+idx,    XMM1);                 \
      _mm_store_ps(r+idx+4,  XMM2);                 \
      _mm_store_ps(r+idx+8,  XMM3);                 \
      _mm_store_ps(r+idx+12, XMM4);                 \
    }

#define SSE_SCALAR_OPERATION_16D(r, a, f, sse_op)               \
    __m128d XMM0 = _mm_set_pd1(f);                              \
    for(uint j=0;j<128;j++){                                    \
      _mm_store_pd(r+j*2, sse_op(_mm_load_pd(a+j*2), XMM0));	\
    }

    template<int N, class T>
    inline void spmatrix_block_muls(T* r, const T* a, T f){
      default_proc::spmatrix_block_muls<N, T>(r, a, f);
    }

    template<>
    inline void spmatrix_block_muls<2, float>(float* r, const float* a,
                                              float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_2(r, a, f, _mm_mul_ps);
    }

    template<>
    inline void spmatrix_block_muls<2, double>(double* r, const double* a,
                                               double f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_2D(r, a, f, _mm_mul_pd);
    }

    template<>
    inline void spmatrix_block_muls<4, float>(float* r, const float* a,
                                              float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_4(r, a, f, _mm_mul_ps);
    }

    template<>
    inline void spmatrix_block_muls<4, double>(double* r, const double* a,
                                               double f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_4D(r, a, f, _mm_mul_pd);
    }

    template<>
    inline void spmatrix_block_muls<8, float>(float* r, const float* a,
                                              float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_8(r, a, f, _mm_mul_ps);
    }

    template<>
    inline void spmatrix_block_muls<8, double>(double* r,
                                               const double* a,
                                               double f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_8D(r, a, f, _mm_mul_pd);
    }

    template<>
    inline void spmatrix_block_muls<16, float>(float* r,
                                               const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_16(r, a, f, _mm_mul_ps);
    }

    template<>
    inline void spmatrix_block_muls<16, double>(double* r,
                                                const double* a,
                                                double f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_16D(r, a, f, _mm_mul_pd);
    }

    template<int N, class T>
    inline void spmatrix_block_divs(T* r, const T* a, T f){
      default_proc::spmatrix_block_divs<N, T>(r, a, f);
    }

    template<>
    inline void spmatrix_block_divs<2, float>(float* r, const float* a,
                                              float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_2(r, a, f, _mm_div_ps);
    }

    template<>
    inline void spmatrix_block_divs<2, double>(double* r, const double* a,
                                               double f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_2D(r, a, f, _mm_div_pd);
    }

    template<>
    inline void spmatrix_block_divs<4, float>(float* r, const float* a,
                                              float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_4(r, a, f, _mm_div_ps);
    }

    template<>
    inline void spmatrix_block_divs<4, double>(double* r,
                                               const double* a,
                                               double f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_4D(r, a, f, _mm_div_pd);
    }

    template<>
    inline void spmatrix_block_divs<8, float>(float* r, const float* a,
                                              float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_8(r, a, f, _mm_div_ps);
    }

    template<>
    inline void spmatrix_block_divs<8, double>(double* r,
                                               const double* a,
                                               double f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_8D(r, a, f, _mm_div_pd);
    }

    template<>
    inline void spmatrix_block_divs<16, float>(float* r,
                                               const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_16(r, a, f, _mm_div_ps);
    }

    template<>
    inline void spmatrix_block_divs<16, double>(double* r,
                                                const double* a,
                                                double f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_16D(r, a, f, _mm_div_pd);
    }

    template<int N, class T>
    inline void spmatrix_block_adds(T* r, const T* a, T f){
      default_proc::spmatrix_block_adds<N, T>(r, a, f);
    }

    template<>
    inline void spmatrix_block_adds<2, float>(float* r, const float* a,
                                              float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_2(r, a, f, _mm_add_ps);
    }

    template<>
    inline void spmatrix_block_adds<2, double>(double* r, const double* a,
                                               double f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_2D(r, a, f, _mm_add_pd);
    }

    template<>
    inline void spmatrix_block_adds<4, float>(float* r, const float* a,
                                              float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_4(r, a, f, _mm_add_ps);
    }

    template<>
    inline void spmatrix_block_adds<4, double>(double* r,
                                               const double* a,
                                               double f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_4D(r, a, f, _mm_add_pd);
    }

    template<>
    inline void spmatrix_block_adds<8, float>(float* r, const float* a,
                                              float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_8(r, a, f, _mm_add_ps);
    }

    template<>
    inline void spmatrix_block_adds<8, double>(double* r,
                                               const double* a,
                                               double f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_8D(r, a, f, _mm_add_pd);
    }

    template<>
    inline void spmatrix_block_adds<16, float>(float* r,
                                               const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_16(r, a, f, _mm_add_ps);
    }

    template<>
    inline void spmatrix_block_adds<16, double>(double* r,
                                                const double* a,
                                                double f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_16D(r, a, f, _mm_add_pd);
    }

    template<int N, class T>
    inline void spmatrix_block_subs(T* r, const T* a, T f){
      default_proc::spmatrix_block_subs<N, T>(r, a, f);
    }

    template<>
    inline void spmatrix_block_subs<2, float>(float* r, const float* a,
                                              float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_2(r, a, f, _mm_sub_ps);
    }

    template<>
    inline void spmatrix_block_subs<2, double>(double* r, const double* a,
                                               double f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_2D(r, a, f, _mm_sub_pd);
    }

    template<>
    inline void spmatrix_block_subs<4, float>(float* r, const float* a,
                                              float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_4(r, a, f, _mm_sub_ps);
    }

    template<>
    inline void spmatrix_block_subs<4, double>(double* r,
                                               const double* a,
                                               double f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_4D(r, a, f, _mm_sub_pd);
    }

    template<>
    inline void spmatrix_block_subs<8, float>(float* r, const float* a,
                                              float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_8(r, a, f, _mm_sub_ps);
    }

    template<>
    inline void spmatrix_block_subs<8, double>(double* r,
                                               const double* a,
                                               double f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_8D(r, a, f, _mm_sub_pd);
    }

    template<>
    inline void spmatrix_block_subs<16, float>(float* r,
                                               const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_16(r, a, f, _mm_sub_ps);
    }

    template<>
    inline void spmatrix_block_subs<16, double>(double* r,
                                                const double* a,
                                                double f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_SCALAR_OPERATION_16D(r, a, f, _mm_sub_pd);
    }

    /*Macros for comparing a block with a scalar*/
#define SSE_COMPARE_BLOCK_S2(r, a, f, sse_cmp)      \
    __m128 XMM8 = _mm_set_ps1(1.0f);				\
                                                    \
    __m128 XMM0 = _mm_set_ps1(f);                   \
    __m128 XMM1 = _mm_load_ps(a);                   \
                                                    \
    XMM1 = _mm_and_ps(sse_cmp(XMM0, XMM1),XMM8);    \
    _mm_store_ps(r, XMM1);

#define SSE_COMPARE_BLOCK_S4(r, a, f, sse_cmp)		\
    __m128 XMM8 = _mm_set_ps1(1.0f);                \
                                                    \
    __m128 XMM0 = _mm_set_ps1(f);                   \
    __m128 XMM1 = _mm_load_ps(a);                   \
    __m128 XMM2 = _mm_load_ps(a+4);                 \
    __m128 XMM3 = _mm_load_ps(a+8);                 \
    __m128 XMM4 = _mm_load_ps(a+12);                \
                                                    \
    XMM1 = _mm_and_ps(sse_cmp(XMM1, XMM0), XMM8);	\
    XMM2 = _mm_and_ps(sse_cmp(XMM2, XMM0), XMM8);	\
    XMM3 = _mm_and_ps(sse_cmp(XMM3, XMM0), XMM8);	\
    XMM4 = _mm_and_ps(sse_cmp(XMM4, XMM0), XMM8);	\
                                                    \
    _mm_store_ps(r,    XMM1);                       \
    _mm_store_ps(r+4,  XMM2);                       \
    _mm_store_ps(r+8,  XMM3);                       \
    _mm_store_ps(r+12, XMM4);

#define SSE_COMPARE_BLOCK_S8(r, a, f, sse_cmp)		\
    __m128 XMM8 = _mm_set_ps1(1.0f);				\
                                                    \
    __m128 XMM0 = _mm_set_ps1(f);                   \
    for(uint j=0;j<4;j++){                          \
      uint idx = j * 16;                            \
      __m128 XMM1 = _mm_load_ps(a+idx);				\
      __m128 XMM2 = _mm_load_ps(a+idx+4);			\
      __m128 XMM3 = _mm_load_ps(a+idx+8);			\
      __m128 XMM4 = _mm_load_ps(a+idx+12);			\
                                                    \
      XMM1 = _mm_and_ps(sse_cmp(XMM1, XMM0), XMM8); \
      XMM2 = _mm_and_ps(sse_cmp(XMM2, XMM0), XMM8); \
      XMM3 = _mm_and_ps(sse_cmp(XMM3, XMM0), XMM8); \
      XMM4 = _mm_and_ps(sse_cmp(XMM4, XMM0), XMM8); \
                                                    \
      _mm_store_ps(r+idx,    XMM1);                 \
      _mm_store_ps(r+idx+4,  XMM2);                 \
      _mm_store_ps(r+idx+8,  XMM3);                 \
      _mm_store_ps(r+idx+12, XMM4);                 \
    }

#define SSE_COMPARE_BLOCK_S16(r, a, f, sse_cmp)     \
    __m128 XMM8 = _mm_set_ps1(1.0f);				\
                                                    \
    __m128 XMM0 = _mm_set_ps1(f);                   \
    for(uint j=0;j<16;j++){                         \
      uint idx = j * 16;                            \
      __m128 XMM1 = _mm_load_ps(a+idx);				\
      __m128 XMM2 = _mm_load_ps(a+idx+4);			\
      __m128 XMM3 = _mm_load_ps(a+idx+8);			\
      __m128 XMM4 = _mm_load_ps(a+idx+12);			\
      XMM1 = _mm_and_ps(sse_cmp(XMM1, XMM0), XMM8); \
      XMM2 = _mm_and_ps(sse_cmp(XMM2, XMM0), XMM8); \
      XMM3 = _mm_and_ps(sse_cmp(XMM3, XMM0), XMM8); \
      XMM4 = _mm_and_ps(sse_cmp(XMM4, XMM0), XMM8); \
                                                    \
      _mm_store_ps(r+idx,    XMM1);                 \
      _mm_store_ps(r+idx+4,  XMM2);                 \
      _mm_store_ps(r+idx+8,  XMM3);                 \
      _mm_store_ps(r+idx+12, XMM4);                 \
    }

    template<int N, class T>
    inline void spmatrix_block_cmpeq(T* r, const T* a, T f){
      default_proc::spmatrix_block_cmpeq<N, T>(r, a, f);
    }

    template<>
    inline void spmatrix_block_cmpeq<2, float>(float* r, const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S2(r, a, f, _mm_cmpeq_ps);
    }

    template<>
    inline void spmatrix_block_cmpeq<4, float>(float* r, const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S4(r, a, f, _mm_cmpeq_ps);
    }

    template<>
    inline void spmatrix_block_cmpeq<8, float>(float* r, const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S8(r, a, f, _mm_cmpeq_ps);
    }

    template<>
    inline void spmatrix_block_cmpeq<16, float>(float* r,
                                                const float* a,
                                                float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S16(r, a, f, _mm_cmpeq_ps);
    }

    template<int N, class T>
    inline void spmatrix_block_cmpneq(T* r, const T* a, T f){
      default_proc::spmatrix_block_cmpneq<N, T>(r, a, f);
    }

    template<>
    inline void spmatrix_block_cmpneq<2, float>(float* r, const float* a,
                                                float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S2(r, a, f, _mm_cmpneq_ps);
    }

    template<>
    inline void spmatrix_block_cmpneq<4, float>(float* r, const float* a,
                                                float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S4(r, a, f, _mm_cmpneq_ps);
    }

    template<>
    inline void spmatrix_block_cmpneq<8, float>(float* r, const float* a,
                                                float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S8(r, a, f, _mm_cmpneq_ps);
    }

    template<>
    inline void spmatrix_block_cmpneq<16, float>(float* r,
                                                 const float* a,
                                                 float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S16(r, a, f, _mm_cmpneq_ps);
    }

    template<int N, class T>
    inline void spmatrix_block_cmplt(T* r, const T* a, T f){
      default_proc::spmatrix_block_cmplt<N, T>(r, a, f);
    }

    template<>
    inline void spmatrix_block_cmplt<2, float>(float* r, const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S2(r, a, f, _mm_cmplt_ps);
    }

    template<>
    inline void spmatrix_block_cmplt<4, float>(float* r, const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S4(r, a, f, _mm_cmplt_ps);
    }

    template<>
    inline void spmatrix_block_cmplt<8, float>(float* r, const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S8(r, a, f, _mm_cmplt_ps);
    }

    template<>
    inline void spmatrix_block_cmplt<16, float>(float* r,
                                                const float* a,
                                                float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S16(r, a, f, _mm_cmplt_ps);
    }

    template<int N, class T>
    inline void spmatrix_block_cmple(T* r, const T* a, T f){
      default_proc::spmatrix_block_cmple<N, T>(r, a, f);
    }

    template<>
    inline void spmatrix_block_cmple<2, float>(float* r, const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S16(r, a, f, _mm_cmple_ps);
    }

    template<>
    inline void spmatrix_block_cmple<4, float>(float* r, const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S4(r, a, f, _mm_cmple_ps);
    }

    template<>
    inline void spmatrix_block_cmple<8, float>(float* r, const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S8(r, a, f, _mm_cmple_ps);
    }

    template<>
    inline void spmatrix_block_cmple<16, float>(float* r,
                                                const float* a,
                                                float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S16(r, a, f, _mm_cmple_ps);
    }

    template<int N, class T>
    inline void spmatrix_block_cmpgt(T* r, const T* a, T f){
      default_proc::spmatrix_block_cmpgt<N, T>(r, a, f);
    }

    template<>
    inline void spmatrix_block_cmpgt<2, float>(float* r, const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S2(r, a, f, _mm_cmpgt_ps);
    }

    template<>
    inline void spmatrix_block_cmpgt<4, float>(float* r, const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S4(r, a, f, _mm_cmpgt_ps);
    }

    template<>
    inline void spmatrix_block_cmpgt<8, float>(float* r, const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S8(r, a, f, _mm_cmpgt_ps);
    }

    template<>
    inline void spmatrix_block_cmpgt<16, float>(float* r,
                                                const float* a,
                                                float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S16(r, a, f, _mm_cmpgt_ps);
    }

    template<int N, class T>
    inline void spmatrix_block_cmpge(T* r, const T* a, T f){
      default_proc::spmatrix_block_cmpge<N, T>(r, a, f);
    }

    template<>
    inline void spmatrix_block_cmpge<2, float>(float* r, const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S2(r, a, f, _mm_cmpge_ps);
    }

    template<>
    inline void spmatrix_block_cmpge<4, float>(float* r, const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S4(r, a, f, _mm_cmpge_ps);
    }

    template<>
    inline void spmatrix_block_cmpge<8, float>(float* r, const float* a,
                                               float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S8(r, a, f, _mm_cmpge_ps);
    }

    template<>
    inline void spmatrix_block_cmpge<16, float>(float* r,
                                                const float* a,
                                                float f){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      SSE_COMPARE_BLOCK_S16(r, a, f, _mm_cmpge_ps);
    }

    template<int N, class T>
    inline void spmatrix_block_mul(T* r, const T* a, const T* b){
      default_proc::spmatrix_block_mul<N,T>(r, a, b);
    }

#define SSE_BLOCK_OPERATION_2(r, a, b, sse_op)	\
    __m128 XMM1  = _mm_load_ps(a);              \
    __m128 XMM01 = _mm_load_ps(b);              \
                                                \
    XMM1 = sse_op(XMM1, XMM01);                 \
    _mm_store_ps(r, XMM1);

#define SSE_BLOCK_OPERATION_2D(r, a, b, sse_op)	\
    __m128d XMM1  = _mm_load_pd(a+0);           \
    __m128d XMM01 = _mm_load_pd(b+0);           \
                                                \
    XMM1 = sse_op(XMM1, XMM01);                 \
    _mm_store_pd(r+0, XMM1);                    \
                                                \
    XMM1  = _mm_load_pd(a+2);                   \
    XMM01 = _mm_load_pd(b+2);                   \
                                                \
    XMM1 = sse_op(XMM1, XMM01);                 \
    _mm_store_pd(r+2, XMM1);

#define SSE_BLOCK_OPERATION_4(r, a, b, sse_op)	\
    __m128 XMM1  = _mm_load_ps(a);              \
    __m128 XMM2  = _mm_load_ps(a+4);            \
    __m128 XMM3  = _mm_load_ps(a+8);            \
    __m128 XMM4  = _mm_load_ps(a+12);           \
    __m128 XMM01 = _mm_load_ps(b);              \
    __m128 XMM02 = _mm_load_ps(b+4);            \
    __m128 XMM03 = _mm_load_ps(b+8);            \
    __m128 XMM04 = _mm_load_ps(b+12);           \
                                                \
    XMM1 = sse_op(XMM1, XMM01);                 \
    XMM2 = sse_op(XMM2, XMM02);                 \
    XMM3 = sse_op(XMM3, XMM03);                 \
    XMM4 = sse_op(XMM4, XMM04);                 \
                                                \
    _mm_store_ps(r,    XMM1);                   \
    _mm_store_ps(r+4,  XMM2);                   \
    _mm_store_ps(r+8,  XMM3);                   \
    _mm_store_ps(r+12, XMM4);

#define SSE_BLOCK_OPERATION_4D(r, a, b, sse_op) \
    for(uint j=0;j<8;j++){                      \
      __m128d XMM0  = _mm_load_pd(a+j*2);		\
      __m128d XMM1  = _mm_load_pd(b+j*2);		\
                                                \
      XMM0 = sse_op(XMM0, XMM1);                \
                                                \
      _mm_store_pd(r+j*2, XMM0);                \
    }

#define SSE_BLOCK_OPERATION_8(r, a, b, sse_op)	\
    for(uint j=0;j<4;j++){                      \
      uint idx = j * 16;                        \
      __m128 XMM1  = _mm_load_ps(a+idx);        \
      __m128 XMM2  = _mm_load_ps(a+idx+4);      \
      __m128 XMM3  = _mm_load_ps(a+idx+8);      \
      __m128 XMM4  = _mm_load_ps(a+idx+12);     \
      __m128 XMM01 = _mm_load_ps(b+idx);        \
      __m128 XMM02 = _mm_load_ps(b+idx+4);      \
      __m128 XMM03 = _mm_load_ps(b+idx+8);      \
      __m128 XMM04 = _mm_load_ps(b+idx+12);     \
                                                \
      XMM1 = sse_op(XMM1, XMM01);               \
      XMM2 = sse_op(XMM2, XMM02);               \
      XMM3 = sse_op(XMM3, XMM03);               \
      XMM4 = sse_op(XMM4, XMM04);               \
                                                \
      _mm_store_ps(r+idx,    XMM1);             \
      _mm_store_ps(r+idx+4,  XMM2);             \
      _mm_store_ps(r+idx+8,  XMM3);             \
      _mm_store_ps(r+idx+12, XMM4);             \
    }

#define SSE_BLOCK_OPERATION_8D(r, a, b, sse_op) \
    for(uint j=0;j<32;j++){                     \
      __m128d XMM0  = _mm_load_pd(a+j*2);		\
      __m128d XMM1  = _mm_load_pd(b+j*2);		\
                                                \
      XMM0 = sse_op(XMM0, XMM1);                \
                                                \
      _mm_store_pd(r+j*2, XMM0);                \
    }

#define SSE_BLOCK_OPERATION_16(r, a, b, sse_op)	\
    for(uint j=0;j<16;j++){                     \
      uint idx = j * 16;                        \
      __m128 XMM1  = _mm_load_ps(a+idx);        \
      __m128 XMM2  = _mm_load_ps(a+idx+4);      \
      __m128 XMM3  = _mm_load_ps(a+idx+8);      \
      __m128 XMM4  = _mm_load_ps(a+idx+12);     \
      __m128 XMM01 = _mm_load_ps(b+idx);        \
      __m128 XMM02 = _mm_load_ps(b+idx+4);      \
      __m128 XMM03 = _mm_load_ps(b+idx+8);      \
      __m128 XMM04 = _mm_load_ps(b+idx+12);     \
                                                \
      XMM1 = sse_op(XMM1, XMM01);               \
      XMM2 = sse_op(XMM2, XMM02);               \
      XMM3 = sse_op(XMM3, XMM03);               \
      XMM4 = sse_op(XMM4, XMM04);               \
                                                \
      _mm_store_ps(r+idx,    XMM1);             \
      _mm_store_ps(r+idx+4,  XMM2);             \
      _mm_store_ps(r+idx+8,  XMM3);             \
      _mm_store_ps(r+idx+12, XMM4);             \
    }

#define SSE_BLOCK_OPERATION_16D(r, a, b, sse_op)	\
    for(uint j=0;j<128;j++){                        \
      __m128d XMM0  = _mm_load_pd(a+j*2);           \
      __m128d XMM1  = _mm_load_pd(b+j*2);           \
                                                    \
      XMM0 = sse_op(XMM0, XMM1);                    \
                                                    \
      _mm_store_pd(r+j*2, XMM0);                    \
    }

    template<>
    inline void spmatrix_block_mul<2, float>(float r[4], const float a[4],
                                             const float b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_2(r, a, b, _mm_mul_ps);
    }

    template<>
    inline void spmatrix_block_mul<2, double>(double r[4], const double a[4],
                                              const double b[4]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_2D(r, a, b, _mm_mul_pd);
    }

    template<>
    inline void spmatrix_block_mul<4, float>(float r[16], const float a[16],
                                             const float b[16]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_4(r, a, b, _mm_mul_ps);
    }

    template<>
    inline void spmatrix_block_mul<4, double>(double r[16], const double a[16],
                                              const double b[16]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_4D(r, a, b, _mm_mul_pd);
    }

    template<>
    inline void spmatrix_block_mul<8, float>(float r[64], const float a[64],
                                             const float b[64]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_8(r, a, b, _mm_mul_ps);
    }

    template<>
    inline void spmatrix_block_mul<8, double>(double r[64], const double a[64],
                                              const double b[64]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_8D(r, a, b, _mm_mul_pd);
    }

    template<>
    inline void spmatrix_block_mul<16, float>(float r[256], const float a[256],
                                              const float b[256]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_16(r, a, b, _mm_mul_ps);
    }

    template<>
    inline void spmatrix_block_mul<16, double>(double r[256],
                                               const double a[256],
                                               const double b[256]){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_16D(r, a, b, _mm_mul_pd);
    }

    template<int N, class T>
    inline void spmatrix_block_div(T* r, const T* a, const T* b){
      default_proc::spmatrix_block_div<N,T>(r, a, b);
    }

    template<>
    inline void spmatrix_block_div<2, float>(float* r, const float* a,
                                             const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_2(r, a, b, _mm_div_ps);
    }

    template<>
    inline void spmatrix_block_div<2, double>(double* r, const double* a,
                                              const double* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_2D(r, a, b, _mm_div_pd);
    }

    template<>
    inline void spmatrix_block_div<4, float>(float* r, const float* a,
                                             const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_4(r, a, b, _mm_div_ps);
    }

    template<>
    inline void spmatrix_block_div<4, double>(double* r, const double* a,
                                              const double* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_4D(r, a, b, _mm_div_pd);
    }

    template<>
    inline void spmatrix_block_div<8, float>(float* r, const float* a,
                                             const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_8(r, a, b, _mm_div_ps);
    }

    template<>
    inline void spmatrix_block_div<8, double>(double* r, const double* a,
                                              const double* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_8D(r, a, b, _mm_div_pd);
    }

    template<>
    inline void spmatrix_block_div<16, float>(float* r, const float* a,
                                              const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_16(r, a, b, _mm_div_ps);
    }

    template<>
    inline void spmatrix_block_div<16, double>(double* r,
                                               const double* a,
                                               const double* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_16D(r, a, b, _mm_div_pd);
    }

#define SSE_COMPARE_BLOCK_2(r, a, b, sse2_cmp)		\
    __m128 XMM8 = _mm_set_ps1(1.0f);                \
                                                    \
    __m128 XMM1  = _mm_load_ps(a);                  \
    __m128 XMM01 = _mm_load_ps(b);                  \
                                                    \
    XMM1 = _mm_and_ps(sse2_cmp(XMM01, XMM1), XMM8);	\
    _mm_store_ps(r, XMM1);

#define SSE_COMPARE_BLOCK_4(r, a, b, sse2_cmp)		\
    __m128 XMM8 = _mm_set_ps1(1.0f);                \
                                                    \
    __m128 XMM1  = _mm_load_ps(a);                  \
    __m128 XMM2  = _mm_load_ps(a+4);                \
    __m128 XMM3  = _mm_load_ps(a+8);                \
    __m128 XMM4  = _mm_load_ps(a+12);               \
    __m128 XMM01 = _mm_load_ps(b);                  \
    __m128 XMM02 = _mm_load_ps(b+4);                \
    __m128 XMM03 = _mm_load_ps(b+8);                \
    __m128 XMM04 = _mm_load_ps(b+12);               \
                                                    \
    XMM1 = _mm_and_ps(sse2_cmp(XMM1, XMM01), XMM8);	\
    XMM2 = _mm_and_ps(sse2_cmp(XMM2, XMM02), XMM8);	\
    XMM3 = _mm_and_ps(sse2_cmp(XMM3, XMM03), XMM8);	\
    XMM4 = _mm_and_ps(sse2_cmp(XMM4, XMM04), XMM8);	\
                                                    \
    _mm_store_ps(r,    XMM1);                       \
    _mm_store_ps(r+4,  XMM2);                       \
    _mm_store_ps(r+8,  XMM3);                       \
    _mm_store_ps(r+12, XMM4);

#define SSE_COMPARE_BLOCK_8(r, a, b, sse2_cmp)          \
    __m128 XMM8 = _mm_set_ps1(1.0f);                    \
                                                        \
    for(uint j=0;j<4;j++){                              \
      uint idx = j * 16;                                \
      __m128 XMM1  = _mm_load_ps(a+idx);                \
      __m128 XMM2  = _mm_load_ps(a+idx+4);              \
      __m128 XMM3  = _mm_load_ps(a+idx+8);              \
      __m128 XMM4  = _mm_load_ps(a+idx+12);             \
      __m128 XMM01 = _mm_load_ps(b+idx);                \
      __m128 XMM02 = _mm_load_ps(b+idx+4);              \
      __m128 XMM03 = _mm_load_ps(b+idx+8);              \
      __m128 XMM04 = _mm_load_ps(b+idx+12);             \
                                                        \
      XMM1 = _mm_and_ps(sse2_cmp(XMM1, XMM01), XMM8);	\
      XMM2 = _mm_and_ps(sse2_cmp(XMM2, XMM02), XMM8);	\
      XMM3 = _mm_and_ps(sse2_cmp(XMM3, XMM03), XMM8);	\
      XMM4 = _mm_and_ps(sse2_cmp(XMM4, XMM04), XMM8);	\
                                                        \
      _mm_store_ps(r+idx,    XMM1);                     \
      _mm_store_ps(r+idx+4,  XMM2);                     \
      _mm_store_ps(r+idx+8,  XMM3);                     \
      _mm_store_ps(r+idx+12, XMM4);                     \
    }

#define SSE_COMPARE_BLOCK_16(r, a, b, sse2_cmp)         \
    __m128 XMM8 = _mm_set_ps1(1.0f);                    \
                                                        \
    for(uint j=0;j<16;j++){                             \
      uint idx = j * 16;                                \
      __m128 XMM1  = _mm_load_ps(a+idx);                \
      __m128 XMM2  = _mm_load_ps(a+idx+4);              \
      __m128 XMM3  = _mm_load_ps(a+idx+8);              \
      __m128 XMM4  = _mm_load_ps(a+idx+12);             \
      __m128 XMM01 = _mm_load_ps(b+idx);                \
      __m128 XMM02 = _mm_load_ps(b+idx+4);              \
      __m128 XMM03 = _mm_load_ps(b+idx+8);              \
      __m128 XMM04 = _mm_load_ps(b+idx+12);             \
                                                        \
      XMM1 = _mm_and_ps(sse2_cmp(XMM1, XMM01), XMM8);	\
      XMM2 = _mm_and_ps(sse2_cmp(XMM2, XMM02), XMM8);	\
      XMM3 = _mm_and_ps(sse2_cmp(XMM3, XMM03), XMM8);	\
      XMM4 = _mm_and_ps(sse2_cmp(XMM4, XMM04), XMM8);	\
                                                        \
      _mm_store_ps(r+idx,    XMM1);                     \
      _mm_store_ps(r+idx+4,  XMM2);                     \
      _mm_store_ps(r+idx+8,  XMM3);                     \
      _mm_store_ps(r+idx+12, XMM4);                     \
    }

    template<int N, class T>
    inline void spmatrix_block_cmpeq(T* r, const T* a,
                                     const T* b){
      default_proc::spmatrix_block_cmpeq<N, T>(r, a, b);
    }

    template<>
    inline void spmatrix_block_cmpeq<2, float>(float* r,
                                               const float* a,
                                               const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_2(r, a, b, _mm_cmpeq_ps);
    }

    template<>
    inline void spmatrix_block_cmpeq<4, float>(float* r,
                                               const float* a,
                                               const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_4(r, a, b, _mm_cmpeq_ps);
    }

    template<>
    inline void spmatrix_block_cmpeq<8, float>(float* r,
                                               const float* a,
                                               const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_8(r, a, b, _mm_cmpeq_ps);
    }

    template<>
    inline void spmatrix_block_cmpeq<16, float>(float* r,
                                                const float* a,
                                                const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_16(r, a, b, _mm_cmpeq_ps);
    }


    template<int N, class T>
    inline void spmatrix_block_cmpneq(T* r, const T* a,
                                      const T* b){
      default_proc::spmatrix_block_cmpneq<N, T>(r, a, b);
    }

    template<>
    inline void spmatrix_block_cmpneq<2, float>(float* r,
                                                const float* a,
                                                const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_2(r, a, b, _mm_cmpneq_ps);
    }

    template<>
    inline void spmatrix_block_cmpneq<4, float>(float* r,
                                                const float* a,
                                                const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_4(r, a, b, _mm_cmpneq_ps);
    }

    template<>
    inline void spmatrix_block_cmpneq<8, float>(float* r,
                                                const float* a,
                                                const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_8(r, a, b, _mm_cmpneq_ps);
    }

    template<>
    inline void spmatrix_block_cmpneq<16, float>(float* r,
                                                 const float* a,
                                                 const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_16(r, a, b, _mm_cmpneq_ps);
    }

    template<int N, class T>
    inline void spmatrix_block_cmplt(T* r, const T* a,
                                     const T* b){
      default_proc::spmatrix_block_cmplt<N, T>(r, a, b);
    }

    template<>
    inline void spmatrix_block_cmplt<2, float>(float* r,
                                               const float* a,
                                               const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_2(r, a, b, _mm_cmplt_ps);
    }

    template<>
    inline void spmatrix_block_cmplt<4, float>(float* r,
                                               const float* a,
                                               const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_4(r, a, b, _mm_cmplt_ps);
    }

    template<>
    inline void spmatrix_block_cmplt<8, float>(float* r,
                                               const float* a,
                                               const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_8(r, a, b, _mm_cmplt_ps);
    }

    template<>
    inline void spmatrix_block_cmplt<16, float>(float* r,
                                                const float* a,
                                                const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_16(r, a, b, _mm_cmplt_ps);
    }

    template<int N, class T>
    inline void spmatrix_block_cmple(T* r, const T* a,
                                     const T* b){
      default_proc::spmatrix_block_cmple<N, T>(r, a, b);
    }

    template<>
    inline void spmatrix_block_cmple<2, float>(float* r,
                                               const float* a,
                                               const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_2(r, a, b, _mm_cmple_ps);
    }

    template<>
    inline void spmatrix_block_cmple<4, float>(float* r,
                                               const float* a,
                                               const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_4(r, a, b, _mm_cmple_ps);
    }

    template<>
    inline void spmatrix_block_cmple<8, float>(float* r,
                                               const float* a,
                                               const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_8(r, a, b, _mm_cmple_ps);
    }

    template<>
    inline void spmatrix_block_cmple<16, float>(float* r,
                                                const float* a,
                                                const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_16(r, a, b, _mm_cmple_ps);
    }

    template<int N, class T>
    inline void spmatrix_block_cmpgt(T* r, const T* a,
                                     const T* b){
      default_proc::spmatrix_block_cmpgt<N, T>(r, a, b);
    }

    template<>
    inline void spmatrix_block_cmpgt<2, float>(float* r,
                                               const float* a,
                                               const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_2(r, a, b, _mm_cmpgt_ps);
    }

    template<>
    inline void spmatrix_block_cmpgt<4, float>(float* r,
                                               const float* a,
                                               const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_4(r, a, b, _mm_cmpgt_ps);
    }

    template<>
    inline void spmatrix_block_cmpgt<8, float>(float* r,
                                               const float* a,
                                               const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_8(r, a, b, _mm_cmpgt_ps);
    }

    template<>
    inline void spmatrix_block_cmpgt<16, float>(float* r,
                                                const float* a,
                                                const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_16(r, a, b, _mm_cmpgt_ps);
    }

    template<int N, class T>
    inline void spmatrix_block_cmpge(T* r, const T* a,
                                     const T* b){
      default_proc::spmatrix_block_cmpge<N, T>(r, a, b);
    }

    template<>
    inline void spmatrix_block_cmpge<2, float>(float* r,
                                               const float* a,
                                               const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_2(r, a, b, _mm_cmpge_ps);
    }

    template<>
    inline void spmatrix_block_cmpge<4, float>(float* r,
                                               const float* a,
                                               const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_4(r, a, b, _mm_cmpge_ps);
    }

    template<>
    inline void spmatrix_block_cmpge<8, float>(float* r,
                                               const float* a,
                                               const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_8(r, a, b, _mm_cmpge_ps);
    }

    template<>
    inline void spmatrix_block_cmpge<16, float>(float* r,
                                                const float* a,
                                                const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_COMPARE_BLOCK_16(r, a, b, _mm_cmpge_ps);
    }

    template<int N, class T>
    inline void spmatrix_block_row_sum_reduce(T* r, const T* a){
      default_proc::spmatrix_block_row_sum_reduce<N, T>(r, a);
    }

    template<>
    inline void spmatrix_block_row_sum_reduce<2, float>(float* r,
                                                        const float* a){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));

      __m128 XMM0 = _mm_set_ps(0, a[2], 0, a[0]);
      __m128 XMM1 = _mm_set_ps(0, a[3], 0, a[1]);
      __m128 XMM2 = _mm_add_ps(XMM0, XMM1);
      _mm_store_ps(r, XMM2);
    }

    template<>
    inline void spmatrix_block_row_sum_reduce<4, float>(float* r,
                                                        const float* a){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));

      __m128 XMM0 = _mm_load_ps(a);
      __m128 XMM1 = _mm_load_ps(a+4);
      __m128 XMM2 = _mm_load_ps(a+8);
      __m128 XMM3 = _mm_load_ps(a+12);

      /*d0 <- XMM0,0 d1 <- XMM0, 2 d2 <- XMM1,0 d3 <- XMM1,2*/
      __m128 XMM4 = _mm_shuffle_ps(XMM0, XMM1, _MM_SHUFFLE(2,0,2,0));
      __m128 XMM5 = _mm_shuffle_ps(XMM0, XMM1, _MM_SHUFFLE(3,1,3,1));
      __m128 XMM6 = _mm_add_ps(XMM4, XMM5);

      __m128 XMM7 = _mm_shuffle_ps(XMM2, XMM3, _MM_SHUFFLE(2,0,2,0));
      __m128 XMM8 = _mm_shuffle_ps(XMM2, XMM3, _MM_SHUFFLE(3,1,3,1));
      __m128 XMM9 = _mm_add_ps(XMM7, XMM8);

      __m128 XMM10 = _mm_shuffle_ps(XMM6, XMM9, _MM_SHUFFLE(2,0,2,0));
      __m128 XMM11 = _mm_shuffle_ps(XMM6, XMM9, _MM_SHUFFLE(3,1,3,1));

      __m128 XMM12 = _mm_add_ps(XMM10, XMM11);

      __m128 XMMZ = _mm_setzero_ps();

      XMM0 = _mm_shuffle_ps(XMM12, XMMZ, _MM_SHUFFLE(0,0,0,0));
      XMM1 = _mm_shuffle_ps(XMM12, XMMZ, _MM_SHUFFLE(1,1,1,1));
      XMM2 = _mm_shuffle_ps(XMM12, XMMZ, _MM_SHUFFLE(2,2,2,2));
      XMM3 = _mm_shuffle_ps(XMM12, XMMZ, _MM_SHUFFLE(3,3,3,3));

      XMM0 = _mm_shuffle_ps(XMM0, XMM0, _MM_SHUFFLE(3,3,3,0));
      XMM1 = _mm_shuffle_ps(XMM1, XMM1, _MM_SHUFFLE(3,3,3,0));
      XMM2 = _mm_shuffle_ps(XMM2, XMM2, _MM_SHUFFLE(3,3,3,0));
      XMM3 = _mm_shuffle_ps(XMM3, XMM3, _MM_SHUFFLE(3,3,3,0));

      _mm_store_ps(r + 0,  XMM0);
      _mm_store_ps(r + 4,  XMM1);
      _mm_store_ps(r + 8,  XMM2);
      _mm_store_ps(r + 12, XMM3);
    }

    template<>
    inline void spmatrix_block_row_sum_reduce<8, float>(float* r,
                                                        const float* a){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));

      /*First reduce complete 4x4 sub blocks in one row into a single
        4x4 block, then reduce each single 4x4 block per block row*/
      for(uint i=0;i<2;i++){
        __m128 XMM0 = _mm_load_ps(a + (i*32));
        __m128 XMM1 = _mm_load_ps(a + (i*32) + 4);
        __m128 XMM2 = _mm_load_ps(a + (i*32) + 8);
        __m128 XMM3 = _mm_load_ps(a + (i*32) + 12);
        __m128 XMM4 = _mm_load_ps(a + (i*32) + 16);
        __m128 XMM5 = _mm_load_ps(a + (i*32) + 20);
        __m128 XMM6 = _mm_load_ps(a + (i*32) + 24);
        __m128 XMM7 = _mm_load_ps(a + (i*32) + 28);

        XMM0 = _mm_add_ps(XMM0, XMM1);
        XMM1 = _mm_add_ps(XMM2, XMM3);
        XMM2 = _mm_add_ps(XMM4, XMM5);
        XMM3 = _mm_add_ps(XMM6, XMM7);

        __m128 XMMZ = _mm_setzero_ps();

        _mm_store_ps(r + (i*32) + 0,  XMM0);
        _mm_store_ps(r + (i*32) + 4,  XMMZ);
        _mm_store_ps(r + (i*32) + 8,  XMM1);
        _mm_store_ps(r + (i*32) + 12, XMMZ);
        _mm_store_ps(r + (i*32) + 16, XMM2);
        _mm_store_ps(r + (i*32) + 20, XMMZ);
        _mm_store_ps(r + (i*32) + 24, XMM3);
        _mm_store_ps(r + (i*32) + 28, XMMZ);

      }
      for(uint i=0;i<2;i++){
        __m128 XMM0 = _mm_load_ps(r + (i*32) + 0);
        __m128 XMM1 = _mm_load_ps(r + (i*32) + 8);
        __m128 XMM2 = _mm_load_ps(r + (i*32) + 16);
        __m128 XMM3 = _mm_load_ps(r + (i*32) + 24);

        __m128 XMM4 = _mm_shuffle_ps(XMM0, XMM1, _MM_SHUFFLE(2,0,2,0));
        __m128 XMM5 = _mm_shuffle_ps(XMM0, XMM1, _MM_SHUFFLE(3,1,3,1));
        __m128 XMM6 = _mm_add_ps(XMM4, XMM5);

        __m128 XMM7 = _mm_shuffle_ps(XMM2, XMM3, _MM_SHUFFLE(2,0,2,0));
        __m128 XMM8 = _mm_shuffle_ps(XMM2, XMM3, _MM_SHUFFLE(3,1,3,1));
        __m128 XMM9 = _mm_add_ps(XMM7, XMM8);

        __m128 XMM10 = _mm_shuffle_ps(XMM6, XMM9, _MM_SHUFFLE(2,0,2,0));
        __m128 XMM11 = _mm_shuffle_ps(XMM6, XMM9, _MM_SHUFFLE(3,1,3,1));

        __m128 XMM12 = _mm_add_ps(XMM10, XMM11);

        __m128 XMMZ = _mm_setzero_ps();

        XMM0 = _mm_shuffle_ps(XMM12, XMMZ, _MM_SHUFFLE(0,0,0,0));
        XMM1 = _mm_shuffle_ps(XMM12, XMMZ, _MM_SHUFFLE(1,1,1,1));
        XMM2 = _mm_shuffle_ps(XMM12, XMMZ, _MM_SHUFFLE(2,2,2,2));
        XMM3 = _mm_shuffle_ps(XMM12, XMMZ, _MM_SHUFFLE(3,3,3,3));

        XMM0 = _mm_shuffle_ps(XMM0, XMM0, _MM_SHUFFLE(3,3,3,0));
        XMM1 = _mm_shuffle_ps(XMM1, XMM1, _MM_SHUFFLE(3,3,3,0));
        XMM2 = _mm_shuffle_ps(XMM2, XMM2, _MM_SHUFFLE(3,3,3,0));
        XMM3 = _mm_shuffle_ps(XMM3, XMM3, _MM_SHUFFLE(3,3,3,0));

        _mm_store_ps(r + (i*32) + 0,  XMM0);
        _mm_store_ps(r + (i*32) + 8,  XMM1);
        _mm_store_ps(r + (i*32) + 16, XMM2);
        _mm_store_ps(r + (i*32) + 24, XMM3);
      }
    }

    template<>
    inline void spmatrix_block_row_sum_reduce<16, float>(float* r,
                                                         const float* a){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));

      /*First reduce complete 4x4 sub blocks in one row into a single
        4x4 block, then reduce each single 4x4 block per block row*/
      for(uint i=0;i<4;i++){
        __m128 XMM0  = _mm_load_ps(a + (i*64));
        __m128 XMM1  = _mm_load_ps(a + (i*64) + 4);
        __m128 XMM2  = _mm_load_ps(a + (i*64) + 8);
        __m128 XMM3  = _mm_load_ps(a + (i*64) + 12);
        __m128 XMM4  = _mm_load_ps(a + (i*64) + 16);
        __m128 XMM5  = _mm_load_ps(a + (i*64) + 20);
        __m128 XMM6  = _mm_load_ps(a + (i*64) + 24);
        __m128 XMM7  = _mm_load_ps(a + (i*64) + 28);
        __m128 XMM8  = _mm_load_ps(a + (i*64) + 32);
        __m128 XMM9  = _mm_load_ps(a + (i*64) + 36);
        __m128 XMM10 = _mm_load_ps(a + (i*64) + 40);
        __m128 XMM11 = _mm_load_ps(a + (i*64) + 44);
        __m128 XMM12 = _mm_load_ps(a + (i*64) + 48);
        __m128 XMM13 = _mm_load_ps(a + (i*64) + 52);
        __m128 XMM14 = _mm_load_ps(a + (i*64) + 56);
        __m128 XMM15 = _mm_load_ps(a + (i*64) + 60);

        XMM0 = _mm_add_ps(_mm_add_ps(XMM0,  XMM2),  _mm_add_ps(XMM1,  XMM3));
        XMM1 = _mm_add_ps(_mm_add_ps(XMM4,  XMM6),  _mm_add_ps(XMM5,  XMM7));
        XMM2 = _mm_add_ps(_mm_add_ps(XMM8,  XMM10), _mm_add_ps(XMM9,  XMM11));
        XMM3 = _mm_add_ps(_mm_add_ps(XMM12, XMM14), _mm_add_ps(XMM13, XMM15));

        __m128 XMMZ = _mm_setzero_ps();

        _mm_store_ps(r + (i*64) + 0,  XMM0);
        _mm_store_ps(r + (i*64) + 4,  XMMZ);
        _mm_store_ps(r + (i*64) + 8,  XMMZ);
        _mm_store_ps(r + (i*64) + 12, XMMZ);
        _mm_store_ps(r + (i*64) + 16, XMM1);
        _mm_store_ps(r + (i*64) + 20, XMMZ);
        _mm_store_ps(r + (i*64) + 24, XMMZ);
        _mm_store_ps(r + (i*64) + 28, XMMZ);
        _mm_store_ps(r + (i*64) + 32, XMM2);
        _mm_store_ps(r + (i*64) + 36, XMMZ);
        _mm_store_ps(r + (i*64) + 40, XMMZ);
        _mm_store_ps(r + (i*64) + 44, XMMZ);
        _mm_store_ps(r + (i*64) + 48, XMM3);
        _mm_store_ps(r + (i*64) + 52, XMMZ);
        _mm_store_ps(r + (i*64) + 56, XMMZ);
        _mm_store_ps(r + (i*64) + 60, XMMZ);

      }
      for(uint i=0;i<4;i++){
        __m128 XMM0 = _mm_load_ps(r + (i*64) + 0);
        __m128 XMM1 = _mm_load_ps(r + (i*64) + 16);
        __m128 XMM2 = _mm_load_ps(r + (i*64) + 32);
        __m128 XMM3 = _mm_load_ps(r + (i*64) + 48);

        __m128 XMM4 = _mm_shuffle_ps(XMM0, XMM1, _MM_SHUFFLE(2,0,2,0));
        __m128 XMM5 = _mm_shuffle_ps(XMM0, XMM1, _MM_SHUFFLE(3,1,3,1));
        __m128 XMM6 = _mm_add_ps(XMM4, XMM5);

        __m128 XMM7 = _mm_shuffle_ps(XMM2, XMM3, _MM_SHUFFLE(2,0,2,0));
        __m128 XMM8 = _mm_shuffle_ps(XMM2, XMM3, _MM_SHUFFLE(3,1,3,1));
        __m128 XMM9 = _mm_add_ps(XMM7, XMM8);

        __m128 XMM10 = _mm_shuffle_ps(XMM6, XMM9, _MM_SHUFFLE(2,0,2,0));
        __m128 XMM11 = _mm_shuffle_ps(XMM6, XMM9, _MM_SHUFFLE(3,1,3,1));

        __m128 XMM12 = _mm_add_ps(XMM10, XMM11);

        __m128 XMMZ = _mm_setzero_ps();

        XMM0 = _mm_shuffle_ps(XMM12, XMMZ, _MM_SHUFFLE(0,0,0,0));
        XMM1 = _mm_shuffle_ps(XMM12, XMMZ, _MM_SHUFFLE(1,1,1,1));
        XMM2 = _mm_shuffle_ps(XMM12, XMMZ, _MM_SHUFFLE(2,2,2,2));
        XMM3 = _mm_shuffle_ps(XMM12, XMMZ, _MM_SHUFFLE(3,3,3,3));

        XMM0 = _mm_shuffle_ps(XMM0, XMM0, _MM_SHUFFLE(3,3,3,0));
        XMM1 = _mm_shuffle_ps(XMM1, XMM1, _MM_SHUFFLE(3,3,3,0));
        XMM2 = _mm_shuffle_ps(XMM2, XMM2, _MM_SHUFFLE(3,3,3,0));
        XMM3 = _mm_shuffle_ps(XMM3, XMM3, _MM_SHUFFLE(3,3,3,0));

        _mm_store_ps(r + (i*64) + 0,  XMM0);
        _mm_store_ps(r + (i*64) + 16, XMM1);
        _mm_store_ps(r + (i*64) + 32, XMM2);
        _mm_store_ps(r + (i*64) + 48, XMM3);
      }
    }

    template<int N, class T>
    inline void spmatrix_block_column_sum_reduce(T r[N*N], const T a[N*N]){
      default_proc::spmatrix_block_column_sum_reduce<N, T>(r, a);
    }

    template<int N, class T>
    inline void spmatrix_block_vector_mul(T* r, const T* a,
                                          const T* v){
      default_proc::spmatrix_block_vector_mul<N, T>(r, a, v);
    }

    template<>
    inline void spmatrix_block_vector_mul<2, float>(float* r,
                                                    const float* a,
                                                    const float* v){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));

      __m128 XMM0 = _mm_set_ps(v[1], v[0], v[1], v[0]);
      __m128 XMM1 = _mm_load_ps(a);
      __m128 XMM2 = _mm_mul_ps(XMM0, XMM1);
      _mm_store_ps(r, XMM2);
    }

    template<>
    inline void spmatrix_block_vector_mul<4, float>(float* r,
                                                    const float* a,
                                                    const float* v){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(v, 16));

      __m128 XMM0 = _mm_load_ps(v);
      __m128 XMM1 = _mm_load_ps(a);
      __m128 XMM2 = _mm_load_ps(a+4);
      __m128 XMM3 = _mm_load_ps(a+8);
      __m128 XMM4 = _mm_load_ps(a+12);

      XMM1 = _mm_mul_ps(XMM0, XMM1);
      XMM2 = _mm_mul_ps(XMM0, XMM2);
      XMM3 = _mm_mul_ps(XMM0, XMM3);
      XMM4 = _mm_mul_ps(XMM0, XMM4);

      _mm_store_ps(r + 0,  XMM1);
      _mm_store_ps(r + 4,  XMM2);
      _mm_store_ps(r + 8,  XMM3);
      _mm_store_ps(r + 12, XMM4);
    }

    template<>
    inline void spmatrix_block_vector_mul<8, float>(float* r,
                                                    const float* a,
                                                    const float* v){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(v, 16));

      __m128 XMM0 = _mm_load_ps(v + 0);
      __m128 XMM1 = _mm_load_ps(v + 4);

      for(uint i=0;i<2;i++){
        __m128 XMM2 = _mm_load_ps(a+i*32);
        __m128 XMM3 = _mm_load_ps(a+i*32+4);
        __m128 XMM4 = _mm_load_ps(a+i*32+8);
        __m128 XMM5 = _mm_load_ps(a+i*32+12);

        __m128 XMM6 = _mm_load_ps(a+i*32+16);
        __m128 XMM7 = _mm_load_ps(a+i*32+20);
        __m128 XMM8 = _mm_load_ps(a+i*32+24);
        __m128 XMM9 = _mm_load_ps(a+i*32+28);

        XMM2 = _mm_mul_ps(XMM0, XMM2);
        XMM3 = _mm_mul_ps(XMM1, XMM3);
        XMM4 = _mm_mul_ps(XMM0, XMM4);
        XMM5 = _mm_mul_ps(XMM1, XMM5);

        _mm_store_ps(r + i*32 + 0,  XMM2);
        _mm_store_ps(r + i*32 + 4,  XMM3);
        _mm_store_ps(r + i*32 + 8,  XMM4);
        _mm_store_ps(r + i*32 + 12, XMM5);

        XMM6 = _mm_mul_ps(XMM0, XMM6);
        XMM7 = _mm_mul_ps(XMM1, XMM7);
        XMM8 = _mm_mul_ps(XMM0, XMM8);
        XMM9 = _mm_mul_ps(XMM1, XMM9);

        _mm_store_ps(r + i*32 + 16, XMM6);
        _mm_store_ps(r + i*32 + 20, XMM7);
        _mm_store_ps(r + i*32 + 24, XMM8);
        _mm_store_ps(r + i*32 + 28, XMM9);
      }
    }

    template<>
    inline void spmatrix_block_vector_mul<16, float>(float* r,
                                                     const float* a,
                                                     const float* v){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(v, 16));

      __m128 XMM0 = _mm_load_ps(v + 0);
      __m128 XMM1 = _mm_load_ps(v + 4);
      __m128 XMM2 = _mm_load_ps(v + 8);
      __m128 XMM3 = _mm_load_ps(v + 12);

      for(uint i=0;i<4;i++){
        __m128 XMM4 =  _mm_load_ps(a+i*64);
        __m128 XMM5 =  _mm_load_ps(a+i*64+4);
        __m128 XMM6 =  _mm_load_ps(a+i*64+8);
        __m128 XMM7 =  _mm_load_ps(a+i*64+12);
        __m128 XMM8 =  _mm_load_ps(a+i*64+16);
        __m128 XMM9 =  _mm_load_ps(a+i*64+20);
        __m128 XMM10 = _mm_load_ps(a+i*64+24);
        __m128 XMM11 = _mm_load_ps(a+i*64+28);
        __m128 XMM12 = _mm_load_ps(a+i*64+32);
        __m128 XMM13 = _mm_load_ps(a+i*64+36);
        __m128 XMM14 = _mm_load_ps(a+i*64+40);
        __m128 XMM15 = _mm_load_ps(a+i*64+44);
        __m128 XMM16 = _mm_load_ps(a+i*64+48);
        __m128 XMM17 = _mm_load_ps(a+i*64+52);
        __m128 XMM18 = _mm_load_ps(a+i*64+56);
        __m128 XMM19 = _mm_load_ps(a+i*64+60);

        XMM4 =  _mm_mul_ps(XMM0, XMM4);
        XMM5 =  _mm_mul_ps(XMM1, XMM5);
        XMM6 =  _mm_mul_ps(XMM2, XMM6);
        XMM7 =  _mm_mul_ps(XMM3, XMM7);
        XMM8 =  _mm_mul_ps(XMM0, XMM8);
        XMM9 =  _mm_mul_ps(XMM1, XMM9);
        XMM10 = _mm_mul_ps(XMM2, XMM10);
        XMM11 = _mm_mul_ps(XMM3, XMM11);
        XMM12 = _mm_mul_ps(XMM0, XMM12);
        XMM13 = _mm_mul_ps(XMM1, XMM13);
        XMM14 = _mm_mul_ps(XMM2, XMM14);
        XMM15 = _mm_mul_ps(XMM3, XMM15);
        XMM16 = _mm_mul_ps(XMM0, XMM16);
        XMM17 = _mm_mul_ps(XMM1, XMM17);
        XMM18 = _mm_mul_ps(XMM2, XMM18);
        XMM19 = _mm_mul_ps(XMM3, XMM19);

        _mm_store_ps(r + i*64 + 0,  XMM4);
        _mm_store_ps(r + i*64 + 4,  XMM5);
        _mm_store_ps(r + i*64 + 8,  XMM6);
        _mm_store_ps(r + i*64 + 12, XMM7);
        _mm_store_ps(r + i*64 + 16, XMM8);
        _mm_store_ps(r + i*64 + 20, XMM9);
        _mm_store_ps(r + i*64 + 24, XMM10);
        _mm_store_ps(r + i*64 + 28, XMM11);
        _mm_store_ps(r + i*64 + 32, XMM12);
        _mm_store_ps(r + i*64 + 36, XMM13);
        _mm_store_ps(r + i*64 + 40, XMM14);
        _mm_store_ps(r + i*64 + 44, XMM15);
        _mm_store_ps(r + i*64 + 48, XMM16);
        _mm_store_ps(r + i*64 + 52, XMM17);
        _mm_store_ps(r + i*64 + 56, XMM18);
        _mm_store_ps(r + i*64 + 60, XMM19);
      }
    }

    template<int N, class T>
    inline void spmatrix_block_vector_mul_transpose(T r[N*N], const T a[N*N],
                                                    const T v[N]){
      default_proc::spmatrix_block_vector_mul_transpose<N, T>(r, a, v);
    }

    /*r = a+b*v*/
    template<int N, class T>
    inline void spmatrix_block_vmadd(T* r, const T* a,
                                     const T* b,
                                     const T* v){
      default_proc::spmatrix_block_vmadd<N, T>(r, a, b, v);
    }

    template<>
    inline void spmatrix_block_vmadd<2, double>(double* r,
                                                const double* a,
                                                const double* b,
                                                const double* v){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      cgfassert(alignment(v, 16));

      __m128d XMM0 = _mm_load_pd(v);

      __m128d XMM1 = _mm_load_pd(b);
      __m128d XMM2 = _mm_load_pd(a);
      _mm_store_pd(r, _mm_add_pd(_mm_mul_pd(XMM0, XMM1), XMM2));

      XMM1 = _mm_load_pd(b+2);
      XMM2 = _mm_load_pd(a+2);
      _mm_store_pd(r+2, _mm_add_pd(_mm_mul_pd(XMM0, XMM1), XMM2));
    }

    template<>
    inline void spmatrix_block_vmadd<2, float>(float* r,
                                               const float* a,
                                               const float* b,
                                               const float* v){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));

      __m128 XMM0 = _mm_set_ps(v[1], v[0], v[1], v[0]);
      __m128 XMM1 = _mm_load_ps(b);
      __m128 XMM2 = _mm_load_ps(a);
      _mm_store_ps(r, _mm_add_ps(_mm_mul_ps(XMM0, XMM1), XMM2));
    }

    /*r = a + b*v*/
    template<>
    inline void spmatrix_block_vmadd<4, double>(double* r,
                                                const double* a,
                                                const double* b,
                                                const double* v){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      cgfassert(alignment(v, 16));

      __m128d XMM0 = _mm_load_pd(v);
      __m128d XMM1 = _mm_load_pd(v+2);

      __m128d XMM21 = _mm_load_pd(b);
      __m128d XMM22 = _mm_load_pd(b+2);
      __m128d XMM23 = _mm_load_pd(b+4);
      __m128d XMM24 = _mm_load_pd(b+6);
      __m128d XMM25 = _mm_load_pd(b+8);
      __m128d XMM26 = _mm_load_pd(b+10);
      __m128d XMM27 = _mm_load_pd(b+12);
      __m128d XMM28 = _mm_load_pd(b+14);

      __m128d XMM31 = _mm_load_pd(a);
      __m128d XMM32 = _mm_load_pd(a+2);
      __m128d XMM33 = _mm_load_pd(a+4);
      __m128d XMM34 = _mm_load_pd(a+6);
      __m128d XMM35 = _mm_load_pd(a+8);
      __m128d XMM36 = _mm_load_pd(a+10);
      __m128d XMM37 = _mm_load_pd(a+12);
      __m128d XMM38 = _mm_load_pd(a+14);

      _mm_store_pd(r+0,  _mm_add_pd(_mm_mul_pd(XMM0, XMM21), XMM31));
      _mm_store_pd(r+2,  _mm_add_pd(_mm_mul_pd(XMM1, XMM22), XMM32));
      _mm_store_pd(r+4,  _mm_add_pd(_mm_mul_pd(XMM0, XMM23), XMM33));
      _mm_store_pd(r+6,  _mm_add_pd(_mm_mul_pd(XMM1, XMM24), XMM34));
      _mm_store_pd(r+8,  _mm_add_pd(_mm_mul_pd(XMM0, XMM25), XMM35));
      _mm_store_pd(r+10, _mm_add_pd(_mm_mul_pd(XMM1, XMM26), XMM36));
      _mm_store_pd(r+12, _mm_add_pd(_mm_mul_pd(XMM0, XMM27), XMM37));
      _mm_store_pd(r+14, _mm_add_pd(_mm_mul_pd(XMM1, XMM28), XMM38));
    }

    template<>
    inline void spmatrix_block_vmadd<4, float>(float* r,
                                               const float* a,
                                               const float* b,
                                               const float* v){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      cgfassert(alignment(v, 16));

      __m128 XMM0 = _mm_load_ps(v);

      __m128 XMM1 = _mm_load_ps(b);
      __m128 XMM2 = _mm_load_ps(b+4);
      __m128 XMM3 = _mm_load_ps(b+8);
      __m128 XMM4 = _mm_load_ps(b+12);
      __m128 XMM5 = _mm_load_ps(a);
      __m128 XMM6 = _mm_load_ps(a+4);
      __m128 XMM7 = _mm_load_ps(a+8);
      __m128 XMM8 = _mm_load_ps(a+12);

      _mm_store_ps(r + 0,  _mm_add_ps(_mm_mul_ps(XMM0, XMM1), XMM5));
      _mm_store_ps(r + 4,  _mm_add_ps(_mm_mul_ps(XMM0, XMM2), XMM6));
      _mm_store_ps(r + 8,  _mm_add_ps(_mm_mul_ps(XMM0, XMM3), XMM7));
      _mm_store_ps(r + 12, _mm_add_ps(_mm_mul_ps(XMM0, XMM4), XMM8));
    }

    template<>
    inline void spmatrix_block_vmadd<8, float>(float* r,
                                               const float* a,
                                               const float* b,
                                               const float* v){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      cgfassert(alignment(v, 16));

      __m128 XMM0 = _mm_load_ps(v + 0);
      __m128 XMM1 = _mm_load_ps(v + 4);

      for(uint i=0;i<2;i++){
        __m128 XMM2 = _mm_load_ps(b+i*32);
        __m128 XMM3 = _mm_load_ps(b+i*32+4);
        __m128 XMM4 = _mm_load_ps(b+i*32+8);
        __m128 XMM5 = _mm_load_ps(b+i*32+12);

        __m128 XMM10 = _mm_load_ps(a+i*32);
        __m128 XMM11 = _mm_load_ps(a+i*32+4);
        __m128 XMM12 = _mm_load_ps(a+i*32+8);
        __m128 XMM13 = _mm_load_ps(a+i*32+12);

        _mm_store_ps(r + i*32 + 0,  _mm_add_ps(_mm_mul_ps(XMM0, XMM2),XMM10));
        _mm_store_ps(r + i*32 + 4,  _mm_add_ps(_mm_mul_ps(XMM1, XMM3),XMM11));
        _mm_store_ps(r + i*32 + 8,  _mm_add_ps(_mm_mul_ps(XMM0, XMM4),XMM12));
        _mm_store_ps(r + i*32 + 12, _mm_add_ps(_mm_mul_ps(XMM1, XMM5),XMM13));

        XMM2 = _mm_load_ps(b+i*32+16);
        XMM3 = _mm_load_ps(b+i*32+20);
        XMM4 = _mm_load_ps(b+i*32+24);
        XMM5 = _mm_load_ps(b+i*32+28);

        XMM10 = _mm_load_ps(a+i*32+16);
        XMM11 = _mm_load_ps(a+i*32+20);
        XMM12 = _mm_load_ps(a+i*32+24);
        XMM13 = _mm_load_ps(a+i*32+28);

        _mm_store_ps(r + i*32 + 16, _mm_add_ps(_mm_mul_ps(XMM0, XMM2),XMM10));
        _mm_store_ps(r + i*32 + 20, _mm_add_ps(_mm_mul_ps(XMM1, XMM3),XMM11));
        _mm_store_ps(r + i*32 + 24, _mm_add_ps(_mm_mul_ps(XMM0, XMM4),XMM12));
        _mm_store_ps(r + i*32 + 28, _mm_add_ps(_mm_mul_ps(XMM1, XMM5),XMM13));
      }
    }

    template<>
    inline void spmatrix_block_vmadd<16, float>(float* r,
                                                const float* a,
                                                const float* b,
                                                const float* v){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      cgfassert(alignment(v, 16));

      __m128 XMM0 = _mm_load_ps(v + 0);
      __m128 XMM1 = _mm_load_ps(v + 4);
      __m128 XMM2 = _mm_load_ps(v + 8);
      __m128 XMM3 = _mm_load_ps(v + 12);

      for(uint i=0;i<4;i++){
        __m128 XMM4 =  _mm_load_ps(b+i*64);
        __m128 XMM5 =  _mm_load_ps(b+i*64+4);
        __m128 XMM6 =  _mm_load_ps(b+i*64+8);
        __m128 XMM7 =  _mm_load_ps(b+i*64+12);

        __m128 XMM20 = _mm_load_ps(a+i*64);
        __m128 XMM21 = _mm_load_ps(a+i*64+4);
        __m128 XMM22 = _mm_load_ps(a+i*64+8);
        __m128 XMM23 = _mm_load_ps(a+i*64+12);

        _mm_store_ps(r + i*64 + 0,  _mm_add_ps(_mm_mul_ps(XMM0, XMM4),XMM20));
        _mm_store_ps(r + i*64 + 4,  _mm_add_ps(_mm_mul_ps(XMM1, XMM5),XMM21));
        _mm_store_ps(r + i*64 + 8,  _mm_add_ps(_mm_mul_ps(XMM2, XMM6),XMM22));
        _mm_store_ps(r + i*64 + 12, _mm_add_ps(_mm_mul_ps(XMM3, XMM7),XMM23));

        XMM4 = _mm_load_ps(b+i*64+16);
        XMM5 = _mm_load_ps(b+i*64+20);
        XMM6 = _mm_load_ps(b+i*64+24);
        XMM7 = _mm_load_ps(b+i*64+28);

        XMM20 = _mm_load_ps(a+i*64+16);
        XMM21 = _mm_load_ps(a+i*64+20);
        XMM22 = _mm_load_ps(a+i*64+24);
        XMM23 = _mm_load_ps(a+i*64+28);

        _mm_store_ps(r + i*64 + 16, _mm_add_ps(_mm_mul_ps(XMM0, XMM4),XMM20));
        _mm_store_ps(r + i*64 + 20, _mm_add_ps(_mm_mul_ps(XMM1, XMM5),XMM21));
        _mm_store_ps(r + i*64 + 24, _mm_add_ps(_mm_mul_ps(XMM2, XMM6),XMM22));
        _mm_store_ps(r + i*64 + 28, _mm_add_ps(_mm_mul_ps(XMM3, XMM7),XMM23));

        XMM4 = _mm_load_ps(b+i*64+32);
        XMM5 = _mm_load_ps(b+i*64+36);
        XMM6 = _mm_load_ps(b+i*64+40);
        XMM7 = _mm_load_ps(b+i*64+44);

        XMM20 = _mm_load_ps(a+i*64+32);
        XMM21 = _mm_load_ps(a+i*64+36);
        XMM22 = _mm_load_ps(a+i*64+40);
        XMM23 = _mm_load_ps(a+i*64+44);

        _mm_store_ps(r + i*64 + 32, _mm_add_ps(_mm_mul_ps(XMM0, XMM4),XMM20));
        _mm_store_ps(r + i*64 + 36, _mm_add_ps(_mm_mul_ps(XMM1, XMM5),XMM21));
        _mm_store_ps(r + i*64 + 40, _mm_add_ps(_mm_mul_ps(XMM2, XMM6),XMM22));
        _mm_store_ps(r + i*64 + 44, _mm_add_ps(_mm_mul_ps(XMM3, XMM7),XMM23));

        XMM4 = _mm_load_ps(b+i*64+48);
        XMM5 = _mm_load_ps(b+i*64+52);
        XMM6 = _mm_load_ps(b+i*64+56);
        XMM7 = _mm_load_ps(b+i*64+60);

        XMM20 = _mm_load_ps(a+i*64+48);
        XMM21 = _mm_load_ps(a+i*64+52);
        XMM22 = _mm_load_ps(a+i*64+56);
        XMM23 = _mm_load_ps(a+i*64+60);

        _mm_store_ps(r + i*64 + 48, _mm_add_ps(_mm_mul_ps(XMM0, XMM4),XMM20));
        _mm_store_ps(r + i*64 + 52, _mm_add_ps(_mm_mul_ps(XMM1, XMM5),XMM21));
        _mm_store_ps(r + i*64 + 56, _mm_add_ps(_mm_mul_ps(XMM2, XMM6),XMM22));
        _mm_store_ps(r + i*64 + 60, _mm_add_ps(_mm_mul_ps(XMM3, XMM7),XMM23));
      }
    }

    template<int N, class T>
    inline void spmatrix_block_vmadd_p(T r[N*N], const T a[N*N],
                                       const T b[N*N],
                                       const T v[N]){
      default_proc::spmatrix_block_vmadd_p<N, T>(r, a, b, v);
    }

    template<int N, class T>
    inline void spmatrix_block_vmadd_n(T r[N*N], const T a[N*N],
                                       const T b[N*N],
                                       const T v[N]){
      default_proc::spmatrix_block_vmadd_n<N, T>(r, a, b, v);
    }

    /*r = a+b'*v*/
    template<int N, class T>
    inline void spmatrix_block_vmadd_transpose(T r[N*N], const T a[N*N],
                                               const T b[N*N],
                                               const T v[N]){
      default_proc::spmatrix_block_vmadd_transpose<N, T>(r, a, b, v);
    }

    template<int N, class T>
    inline void spmatrix_block_vmadd_jacobi(T r[N*N], const T a[N*N],
                                            const T b[N*N],
                                            const T v[N]){
      default_proc::spmatrix_block_vmadd_jacobi<N, T>(r, a, b, v);
    }

    template<int N, class T>
    inline void spmatrix_block_add(T* r, const T* a, const T* b){
      default_proc::spmatrix_block_add<N, T>(r, a, b);
    }

    template<>
    inline void spmatrix_block_add<2, float>(float* r,
                                             const float* a,
                                             const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_2(r, a, b, _mm_add_ps);
    }

    template<>
    inline void spmatrix_block_add<2, double>(double* r,
                                              const double* a,
                                              const double* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_2D(r, a, b, _mm_add_pd);
    }

    template<>
    inline void spmatrix_block_add<4, float>(float* r,
                                             const float* a,
                                             const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_4(r, a, b, _mm_add_ps);
    }

    template<>
    inline void spmatrix_block_add<4, double>(double* r,
                                              const double* a,
                                              const double* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_4D(r, a, b, _mm_add_pd);
    }

    template<>
    inline void spmatrix_block_add<8, float>(float* r,
                                             const float* a,
                                             const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_8(r, a, b, _mm_add_ps);
    }

    template<>
    inline void spmatrix_block_add<8, double>(double* r,
                                              const double* a,
                                              const double* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_8D(r, a, b, _mm_add_pd);
    }

    template<>
    inline void spmatrix_block_add<16, float>(float* r,
                                              const float* a,
                                              const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_16(r, a, b, _mm_add_ps);
    }

    template<>
    inline void spmatrix_block_add<16, double>(double* r,
                                               const double* a,
                                               const double* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_16D(r, a, b, _mm_add_pd);
    }

    template<int N, class T>
    inline void spmatrix_block_sub(T* r, const T* a, const T* b){
      default_proc::spmatrix_block_sub<N, T>(r, a, b);
    }

    template<>
    inline void spmatrix_block_sub<2, float>(float* r,
                                             const float* a,
                                             const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_2(r, a, b, _mm_sub_ps);
    }

    template<>
    inline void spmatrix_block_sub<2, double>(double* r,
                                              const double* a,
                                              const double* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_2D(r, a, b, _mm_sub_pd);
    }

    template<>
    inline void spmatrix_block_sub<4, float>(float* r,
                                             const float* a,
                                             const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_4(r, a, b, _mm_sub_ps);
    }

    template<>
    inline void spmatrix_block_sub<4, double>(double* r,
                                              const double* a,
                                              const double* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_4D(r, a, b, _mm_sub_pd);
    }

    template<>
    inline void spmatrix_block_sub<8, float>(float* r,
                                             const float* a,
                                             const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_8(r, a, b, _mm_sub_ps);
    }

    template<>
    inline void spmatrix_block_sub<8, double>(double* r,
                                              const double* a,
                                              const double* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_8D(r, a, b, _mm_sub_pd);
    }

    template<>
    inline void spmatrix_block_sub<16, float>(float* r,
                                              const float* a,
                                              const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_16(r, a, b, _mm_sub_ps);
    }

    template<>
    inline void spmatrix_block_sub<16, double>(double* r,
                                               const double* a,
                                               const double* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_OPERATION_16D(r, a, b, _mm_sub_pd);
    }

    /*r = a * b + c*/
    template<int N, class T>
    inline void spmatrix_block_madd(T* r, const T* a, const T* b,
                                    const T* c){
      default_proc::spmatrix_block_madd<N, T>(r, a, b, c);
    }

#define SSE_BLOCK_MADDF(n, r, a, b, c)                                  \
    for(uint i=0;i<n;i++){                                              \
      __m128 XMM0 = _mm_load_ps(a+i*4);                                 \
      __m128 XMM1 = _mm_load_ps(b+i*4);                                 \
      __m128 XMM2 = _mm_load_ps(c+i*4);                                 \
                                                                        \
      _mm_store_ps(r+i*4, _mm_add_ps(_mm_mul_ps(XMM0, XMM1), XMM2));	\
    }

#define SSE_BLOCK_MADDD(n, r, a, b, c)                                  \
    for(uint i=0;i<n;i++){                                              \
      __m128d XMM0  = _mm_load_pd(a+i*4+0);                             \
      __m128d XMM1  = _mm_load_pd(b+i*4+0);                             \
      __m128d XMM2  = _mm_load_pd(c+i*4+0);                             \
      __m128d XMM01 = _mm_load_pd(a+i*4+2);                             \
      __m128d XMM11 = _mm_load_pd(b+i*4+2);                             \
      __m128d XMM21 = _mm_load_pd(c+i*4+2);                             \
                                                                        \
      _mm_store_pd(r+i*4+0, _mm_add_pd(_mm_mul_pd(XMM0, XMM1), XMM2));	\
      _mm_store_pd(r+i*4+2, _mm_add_pd(_mm_mul_pd(XMM01, XMM11), XMM21)); \
    }

    template<>
    inline void spmatrix_block_madd<2,float>(float* r,
                                             const float* a,
                                             const float* b,
                                             const float* c){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      cgfassert(alignment(c, 16));
      SSE_BLOCK_MADDF(1, r, a, b, c);
    }

    template<>
    inline void spmatrix_block_madd<4,float>(float* r,
                                             const float* a,
                                             const float* b,
                                             const float* c){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      cgfassert(alignment(c, 16));
      SSE_BLOCK_MADDF(4, r, a, b, c);
    }

    template<>
    inline void spmatrix_block_madd<8,float>(float* r,
                                             const float* a,
                                             const float* b,
                                             const float* c){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      cgfassert(alignment(c, 16));
      SSE_BLOCK_MADDF(16, r, a, b, c);
    }

    template<>
    inline void spmatrix_block_madd<16,float>(float* r,
                                              const float* a,
                                              const float* b,
                                              const float* c){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      cgfassert(alignment(c, 16));
      SSE_BLOCK_MADDF(64, r, a, b, c);
    }
	/*
      template<>
      inline void spmatrix_block_madd<1, double>(double* r,
      const double* a,
      const double* b,
      const double* c){
      default_proc::spmatrix_block_madd<1, double>(r, a, b, c);
      }*/

    template<>
    inline void spmatrix_block_madd<2, double>(double* r,
                                               const double* a,
                                               const double* b,
                                               const double* c){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      cgfassert(alignment(c, 16));
      SSE_BLOCK_MADDD(1, r, a, b, c);
    }

    template<>
    inline void spmatrix_block_madd<4, double>(double* r,
                                               const double* a,
                                               const double* b,
                                               const double* c){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      cgfassert(alignment(c, 16));
      SSE_BLOCK_MADDD(4, r, a, b, c);
    }

    template<>
    inline void spmatrix_block_madd<8, double>(double* r,
                                               const double* a,
                                               const double* b,
                                               const double* c){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      cgfassert(alignment(c, 16));
      SSE_BLOCK_MADDD(16, r, a, b, c);
    }

    template<>
    inline void spmatrix_block_madd<16, double>(double* r,
                                                const double* a,
                                                const double* b,
                                                const double* c){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      cgfassert(alignment(c, 16));
      SSE_BLOCK_MADDD(64, r, a, b, c);
    }

#define SSE_BLOCK_MSADDF(n, r, f, a, b)                                 \
    __m128 XMM3 = _mm_set_ps1(f);                                       \
    for(uint i=0;i<n;i++){                                              \
      __m128 XMM0 = _mm_load_ps(a+i*4);                                 \
      __m128 XMM1 = _mm_load_ps(b+i*4);                                 \
                                                                        \
      _mm_store_ps(r+i*4, _mm_add_ps(_mm_mul_ps(XMM3, XMM0), XMM1));	\
    }

#define SSE_BLOCK_MSADDD(n, r, f, a, b)                                 \
    __m128d XMM3 = _mm_set_pd1(f);                                      \
    for(uint i=0;i<n;i++){                                              \
      __m128d XMM0  = _mm_load_pd(a+i*4+0);                             \
      __m128d XMM1  = _mm_load_pd(b+i*4+0);                             \
      __m128d XMM01 = _mm_load_pd(a+i*4+2);                             \
      __m128d XMM11 = _mm_load_pd(b+i*4+2);                             \
                                                                        \
      _mm_store_pd(r+i*4+0, _mm_add_pd(_mm_mul_pd(XMM3, XMM0), XMM1));	\
      _mm_store_pd(r+i*4+2, _mm_add_pd(_mm_mul_pd(XMM3, XMM01), XMM11)); \
    }

    template<int N, class T>
    inline void spmatrix_block_msadd(T r[N*N], T f,
                                     const T a[N*N], const T b[N*N]){
      default_proc::spmatrix_block_msadd<N, T>(r, f, a, b);
    }

    template<>
    inline void spmatrix_block_msadd<2, float>(float* r, float f,
                                               const float* a, const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_MSADDF(1, r, f, a, b);
    }

    template<>
    inline void spmatrix_block_msadd<4, float>(float* r, float f,
                                               const float* a, const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_MSADDF(4, r, f, a, b);
    }

    template<>
    inline void spmatrix_block_msadd<8, float>(float* r, float f,
                                               const float* a, const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_MSADDF(16, r, f, a, b);
    }

    template<>
    inline void spmatrix_block_msadd<16, float>(float* r, float f,
                                                const float* a, const float* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_MSADDF(64, r, f, a, b);
    }

    template<>
    inline void spmatrix_block_msadd<2, double>(double* r, double f,
                                                const double* a,
                                                const double* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_MSADDD(1, r, f, a, b);
    }

    template<>
    inline void spmatrix_block_msadd<4, double>(double* r, double f,
                                                const double* a,
                                                const double* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_MSADDD(4, r, f, a, b);
    }

    template<>
    inline void spmatrix_block_msadd<8, double>(double* r, double f,
                                                const double* a,
                                                const double* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_MSADDD(16, r, f, a, b);
    }

    template<>
    inline void spmatrix_block_msadd<16, double>(double* r, double f,
                                                 const double* a,
                                                 const double* b){
      cgfassert(alignment(r, 16));
      cgfassert(alignment(a, 16));
      cgfassert(alignment(b, 16));
      SSE_BLOCK_MSADDD(64, r, f, a, b);
    }

    template<int N, class T>
    inline void spmatrix_block_msadd2(T r[N*N], T f,
                                      const T a[N*N], const T b[N*N],
                                      const T d[N*N]){
      default_proc::spmatrix_block_msadd2<N, T>(r, f, a, b, d);
    }

    template<int N, class T>
    inline void spmatrix_block_clear(T r[N*N]){
      default_proc::spmatrix_block_clear<N, T>(r);
    }

    template<int N, class T>
    inline void spmatrix_block_set(T r[N*N], T f){
      default_proc::spmatrix_block_set<N, T>(r, f);
    }
  }
}
#endif/*SSE2_SPMATRIX_INTRINSICS*/
