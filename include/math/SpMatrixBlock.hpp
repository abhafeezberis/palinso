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

#ifndef SPMATRIXBLOCK_HPP
#define SPMATRIXBLOCK_HPP

#if defined SSE2
#include "math/x86-sse/sse_spmatrix_intrinsics.hpp"
using namespace CGF::x86_sse2;
#elif defined NEON
#include "math/arm-neon/neon_spmatrix_intrinsics.hpp"
using namespace CGF::arm_neon;
#else
#include "math/default/default_spmatrix_intrinsics.hpp"
using namespace CGF::default_proc;
#endif

#include <ostream>
#include <iostream>
#include <string.h>

namespace CGF{
  template <int N=DEFAULT_BLOCK_SIZE, class T=float>
  class CGFAPI SpMatrixBlock{
  public:

#if (defined SSE2)
#ifdef _WIN32
	__declspec(align(16)) T m[N*N];
#else
    T m[N*N] __attribute__((aligned(16)));
#endif
#else
    T m[N*N];
#endif

#ifdef SSE2
    SpMatrixBlock(){
      cgfassert(alignment(m, 16));   /*Check alignment*/
    }
#endif

    void mul   (T f);
    void div   (T f);
    void cmpeq (T f);
    void cmpneq(T f);
    void cmplt (T f);
    void cmple (T f);
    void cmpgt (T f);
    void cmpge (T f);

    void mul   (const SpMatrixBlock<N,T>* const b);
    void div   (const SpMatrixBlock<N,T>* const b);
    void cmpeq (const SpMatrixBlock<N,T>* const b);
    void cmpneq(const SpMatrixBlock<N,T>* const b);
    void cmplt (const SpMatrixBlock<N,T>* const b);
    void cmple (const SpMatrixBlock<N,T>* const b);
    void cmpgt (const SpMatrixBlock<N,T>* const b);
    void cmpge (const SpMatrixBlock<N,T>* const b);

    void add   (const SpMatrixBlock<N,T>* const b);
    void sub   (const SpMatrixBlock<N,T>* const b);

    void clear();
    void set(T f);

    /*reduceAllRows to a singular value using the sum operator*/
    void rowSumReduce();
    void columnSumReduce();

    void vectorMul   (const T* const vec);
    void vectorMulAdd(const SpMatrixBlock<N,T>*const mat,
                      const T* const vec);
    void vectorMulAddP(const SpMatrixBlock<N,T>*const mat,
                       const T* const vec);
    void vectorMulAddN(const SpMatrixBlock<N,T>*const mat,
                       const T* const vec);
    void vectorMulTranspose   (const T* const vec);
    void vectorMulAddTranspose(const SpMatrixBlock<N,T>*const mat,
                               const T* const vec);
    void vectorMulAddJacobi(const SpMatrixBlock<N,T>*const mat,
                            const T* const vec);

    SpMatrixBlock<N,T>& operator=(const SpMatrixBlock<N,T>& mat);

    template<int M, class TT>
      friend std::ostream& operator<<(std::ostream& os,
                                      const SpMatrixBlock<M, TT>& b);
  };

  template<int N, class T>
  std::ostream& operator<<(std::ostream& os,
                           const SpMatrixBlock<N, T>& b){
    int index = 0;
    for(int h=0;h<N;h++){
      for(int w=0;w<N;w++){
        os << b.m[index++] << "\t";
      }
      os << std::endl;
    }
    return os;
  }

  template<int N, class T>
  inline SpMatrixBlock<N, T>& SpMatrixBlock<N, T>::operator=(const SpMatrixBlock<N, T>& mat){
    spmatrix_block_load<N, T>(m, mat.m);
    return *this;
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::mul(T f){
    spmatrix_block_muls<N, T>(m, m, f);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::div(T f){
    spmatrix_block_divs<N, T>(m, m, f);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::cmpeq(T f){
    spmatrix_block_cmpeq<N, T>(m, m, f);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::cmpneq(T f){
    spmatrix_block_cmpneq<N, T>(m, m, f);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::cmplt(T f){
    spmatrix_block_cmplt<N, T>(m, m, f);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::cmple(T f){
    spmatrix_block_cmple<N, T>(m, m, f);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::cmpgt(T f){
    spmatrix_block_cmpgt<N, T>(m, m, f);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::cmpge(T f){
    spmatrix_block_cmpge<N, T>(m, m, f);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::mul(const SpMatrixBlock<N, T>* const b){
    spmatrix_block_mul<N, T>(m, m, b->m);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::div(const SpMatrixBlock<N, T>* const b){
    spmatrix_block_div<N, T>(m, m, b->m);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::cmpeq(const SpMatrixBlock<N, T>* const b){
    spmatrix_block_cmpeq<N, T>(m, m, b->m);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::cmpneq(const SpMatrixBlock<N, T>* const b){
    spmatrix_block_cmpneq<N, T>(m, m, b->m);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::cmplt(const SpMatrixBlock<N, T>* const b){
    spmatrix_block_cmplt<N, T>(m, m, b->m);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::cmple(const SpMatrixBlock<N, T>* const b){
    spmatrix_block_cmple<N, T>(m, m, b->m);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::cmpgt(const SpMatrixBlock<N, T>* const b){
    spmatrix_block_cmpgt<N, T>(m, m, b->m);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::cmpge(const SpMatrixBlock<N, T>* const b){
    spmatrix_block_cmpge<N, T>(m, m, b->m);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::rowSumReduce(){
    spmatrix_block_row_sum_reduce<N, T>(m, m);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::columnSumReduce(){
    spmatrix_block_column_sum_reduce<N, T>(m, m);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::vectorMul(const T* const vec){
    spmatrix_block_vector_mul<N, T>(m, m, vec);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::vectorMulAdd(const SpMatrixBlock<N, T>*const mat,
                                                const T* const vec){
    spmatrix_block_vmadd<N, T>(m, m, mat->m, vec);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::vectorMulAddP(const SpMatrixBlock<N, T>*const mat,
                                                 const T* const vec){
    spmatrix_block_vmadd_p<N, T>(m, m, mat->m, vec);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::vectorMulAddN(const SpMatrixBlock<N, T>*const mat,
                                                 const T* const vec){
    spmatrix_block_vmadd_n<N, T>(m, m, mat->m, vec);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::vectorMulTranspose(const T* const vec){
    spmatrix_block_vector_mul_transpose<N, T>(m, m, vec);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::vectorMulAddTranspose(const SpMatrixBlock<N, T>*const mat,
                                                         const T* const vec){
    spmatrix_block_vmadd_transpose<N, T>(m, m, mat->m, vec);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::vectorMulAddJacobi(const SpMatrixBlock<N, T>*const mat,
                                                      const T* const vec){
    spmatrix_block_vmadd_jacobi<N, T>(m, m, mat->m, vec);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::add(const SpMatrixBlock<N, T>* const b){
    spmatrix_block_add<N, T>(m, m, b->m);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::sub(const SpMatrixBlock<N, T>* const b){
    spmatrix_block_sub<N, T>(m, m, b->m);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::clear(){
    spmatrix_block_clear<N, T>(m);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::set(T f){
    spmatrix_block_set<N, T>(m, f);
  }

  template class SpMatrixBlock<1, float>;
  template class SpMatrixBlock<2, float>;
  template class SpMatrixBlock<4, float>;
  template class SpMatrixBlock<8, float>;
  template class SpMatrixBlock<16, float>;

  template class SpMatrixBlock<1, double>;
  template class SpMatrixBlock<2, double>;
  template class SpMatrixBlock<4, double>;
  template class SpMatrixBlock<8, double>;
  template class SpMatrixBlock<16, double>;
}

#endif/*SPMATRIXBLOCK_HPP*/
