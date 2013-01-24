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

#ifndef CUDACGOP_HPP
#define CUDACGOP_HPP
#ifdef CUDA

#define NTHREADS 256
#define R_LOOP 1

#include "math/CVector.hpp"

namespace CGF{
  //template<class T>
  //class CGFAPI CVector;
  
  /*N = block_size, T = float|doublem N_THR = 128, 256, 512, TEX=true|false*/
  /*  template<int N, class T, int N_THR, bool TEX>
  struct CUDASPMV{
    static void spmv_ordered(T* d_blocks, 
			     uint* d_col_indices, 
			     uint* d_row_lengths, uint* d_row_indices,
			     uint* d_row_map,
			     const T* d_b, T* d_x, uint dim, 
			     uint n_blocks);
			     };*/


  template<class T>
  void parallel_cg_step_1(const CVector<T>* b, CVector<T>* C,
			  CVector<T>* r, CVector<T>* v, CVector<T>* w, 
			  CVector<T>* tmp, CVector<T>* tmp2, 
			  CVector<T>* full_vec, 
			  T* mapped_memory, uint n_threads, 
			  const Thread* caller);

  template<class T>
  void parallel_cg_step_2(const CVector<T>* v, CVector<T>* tmp, 
			  const Thread* caller);

  template<class T>
  void parallel_cg_step_3(const CVector<T>* v, const CVector<T>* u, 
			  CVector<T>* tmp, const Thread* caller);

  template<class T>
  void parallel_cg_step_4(const CVector<T>* v, const CVector<T>* u,
			  const CVector<T>* C, const T t,
			  CVector<T>* w, CVector<T>* x, CVector<T>* r, 
			  CVector<T>* tmp, const Thread* caller);

  template<class T>
  void parallel_cg_step_5(const CVector<T>* w, const CVector<T>* C, const T s,
			  CVector<T>* v, CVector<T>* tmp, 
			  CVector<T>* full_vec, T* mapped_memory, 
			  uint n_threads, const Thread* caller);

  template<class T>
  void parallel_reduction(const CVector<T>* tmp, CVector<T>* tmp2, T** res, 
			  T* mapped, const Thread* caller);

  template<class T>
  void parallel_reduction2(const CVector<T>* tmp, CVector<T>* tmp2, T** res, 
			   T* mapped, const Thread* caller);

  template<class T>
  void vector_test(CVector<T>* a, const CVector<T>* b, const CVector<T>* c, 
		   const CVector<T>* d, const CVector<T>* e, 
		   const CVector<T>* f, const CVector<T>* g, 
		   const CVector<T>* h, const Thread* caller);

  inline uint optimalLoopCount(uint size, uint n_devices){
    uint steps = ceil((float)size/(float)1024/(float)n_devices/(float)30);
    message("optimal n_loops = %d", steps);
    return steps;
  }

  inline uint minTempReductionSize(uint size, uint n_devices){
    //return n_devices * 4 * 240;
    uint s = n_devices * ceil((float)size/(float)R_LOOP/(float)NTHREADS)*2;
    if(s < n_devices*16){
      s = n_devices * 16;/*Minimal vector size = 16*/
    }
    return s;
  }
}

#endif/*CUDA*/
#endif/*CUDACGOP_HPP*/
