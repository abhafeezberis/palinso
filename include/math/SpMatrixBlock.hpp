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
    T m[N*N];//__attribute__((aligned(16)));
#else
    T m[N*N];
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

    /*reduceAllRows to a singular value using the sum operator*/
    void rowSumReduce();
    void columnSumReduce();
    void vectorMul   (const T* const vec);
    void vectorMulAdd(const SpMatrixBlock<N,T>*const mat, 
		      const T* const vec);
    void vectorMulTranspose   (const T* const vec);
    void vectorMulAddTranspose(const SpMatrixBlock<N,T>*const mat, 
			       const T* const vec);
    
    SpMatrixBlock<N,T>& operator=(const SpMatrixBlock<N,T>& mat);

    float getFillRate()const;
    void analyse(uint n_blocks[5], uint n_zeros[5], 
		 uint n_nonzeros[5], float treshold)const;

    uint getCompressedBits()const;
    uint getNZCompressedBits()const;

    template<int M, class TT>
      friend std::ostream& operator<<(std::ostream& os, 
				      const SpMatrixBlock<M, TT>& b);
  };

  template<int N, class T> 
  std::ostream& operator<<(std::ostream& os, 
			   const SpMatrixBlock<N, T>& b){
    uint index = 0;
    for(uint h=0;h<N;h++){
      for(uint w=0;w<N;w++){
	os << b.m[index++] << "\t";
      }
      os << std::endl;
    }
    return os;
  }

  inline uint trailingNZBits(uint d){
    uint count = 0;
    while(d != 0){
      d >>= 1;
      count++;
    }
    return count;
  }

  template<int N, class T>
  uint SpMatrixBlock<N, T>::getCompressedBits()const{
    uint* p = (uint*)m;
    uint compressedBits = 0;
    uint stride = 0;

    for(uint i=0;i<N*N;i++){
      uint predicted;
      if(i==0){
	predicted = 0 + stride;
      }else{
	predicted = p[i-1] + stride;
      }

      uint x = p[i]^predicted;

      compressedBits += trailingNZBits(x);

      if(i==0)
	stride = p[i] - 0;
      else
	stride = p[i] - p[i-1];
    }
    return compressedBits;
  }

  template<int N, class T>
  uint SpMatrixBlock<N, T>::getNZCompressedBits()const{
    uint* p = (uint*)m;
    uint compressedBits = 0;
    uint stride = 0;
    uint last = 0;
    
    for(uint i=0;i<N*N;i++){
      uint predicted;
      predicted = last + stride;
      if(p[i] == 0)
	continue;
      uint x = p[i]^predicted;

      compressedBits += trailingNZBits(x);

      if(i==0)
	stride = p[i] - 0;
      else
	stride = p[i] - last;
      last = p[i];
    }
    return compressedBits;
  }
  
  template<int N, class T>
  float SpMatrixBlock<N, T>::getFillRate()const{
    float fill = 0;
    for(uint i=0;i<N;i++){
      for(uint j=0;j<N;j++){
	if(m[i*N+j]!= 0)
	  fill += 1;
      }
    }
    return fill / (float)(N*N);
  }

  template<class T>
  class BLK{
  public:
    BLK(){
      m = 0;
      childs = 0;
      dim = 0;
      empty = true;
    }
    ~BLK(){
      delete[] m;
      if(childs)
	delete[]childs;
    }
    void setDim(uint d){
      dim = d;
    }
    void alloc(){
      m = new T[dim*dim];
    }

    void subDivide(float treshold){
      float fill = 0;
      
      for(uint i=0;i<dim;i++){
	for(uint j=0;j<dim;j++){
	  if(m[i*dim + j]!= 0){
	    fill += 1.0/(dim*dim);
	    empty = false;
	  }
	}
      }
      
      if(fill < treshold && !empty){
	/*Subdivide*/
	childs = new BLK<T>[4];
	for(uint i=0;i<2;i++){
	  for(uint j=0;j<2;j++){
	    childs[i*2+j].setDim(dim/2);
	    childs[i*2+j].alloc();
	    
	    for(uint k=0;k<dim/2;k++){
	      for(uint l=0;l<dim/2;l++){
		childs[i*2+j].m[k*dim/2 + l] = 
		  m[(k+i*dim/2)*dim + l + j*dim/2];
		
		m[(k+i*dim/2)*dim + l + j*dim/2] = 0;
		
	      }
	    }
	    if(dim > 1)
	      childs[i*2+j].subDivide(treshold);
	  }
	}
	empty = true;
      }
    }
    
    void analyse(uint n_blocks[5], uint n_zeros[5], uint n_nonzeros[5]){
      uint depth = (uint)log2(dim);
      
      if(!empty){
	for(uint i=0;i<dim;i++){
	  for(uint j=0;j<dim;j++){
	    if(m[i*dim+j] == 0){
	      n_zeros[depth]++;
	    }else{
	      n_nonzeros[depth]++;
	    }
	  }
	}
	n_blocks[depth]++;
      }
      
      if(childs){
	for(uint i=0;i<4;i++){
	  childs[i].analyse(n_blocks, n_zeros, n_nonzeros);
	}
      }
    }
    
    T* m;
    uint dim;
    BLK<T>* childs;
    bool empty;
  };

  template<int N, class T>
  void SpMatrixBlock<N, T>::analyse(uint n_blocks[5], uint n_zeros[5], 
				    uint n_nonzeros[5], float treshold)const{
    BLK<T> blk;
    blk.setDim(N);
    blk.alloc();
    memcpy(blk.m, m, sizeof(T)*N*N);
    blk.subDivide(treshold);
    blk.analyse(n_blocks, n_zeros, n_nonzeros);
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
  inline void SpMatrixBlock<N, T>::vectorMulTranspose(const T* const vec){
    spmatrix_block_vector_mul_transpose<N, T>(m, m, vec);
  }

  template<int N, class T>
  inline void SpMatrixBlock<N, T>::vectorMulAddTranspose(const SpMatrixBlock<N, T>*const mat,
					     const T* const vec){
    spmatrix_block_vmadd_transpose<N, T>(m, m, mat->m, vec);
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
