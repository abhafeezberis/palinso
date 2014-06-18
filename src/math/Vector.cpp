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

#include "math/Vector.hpp"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef USE_THREADS
#include "core/Thread.hpp"
#endif
#include <vector>
#include <algorithm>


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

#define RANGE_ASSERT_A                          \
  cgfassert((r.size == a.size));                \
  cgfassert(rg.startBlock % 16 == 0);			\
  cgfassert(rg.endBlock   % 16 == 0);

#define RANGE_ASSERT_AB                         \
  RANGE_ASSERT_A                                \
  cgfassert((a.size == b.size));

namespace CGF{
  /*The size of a vector is always a multiple of 16*/
  template<class T>
  Vector<T>::Vector(int s){
    origSize = s;
    if(s % 16 == 0){
      size = s;
    }else{
      size = ((s/16)+1)*16;
    }

    if(size==0){
      data = 0;
      tmp_buffer = 0;
      return;
    }

#ifdef _WIN32
    data = _aligned_malloc((uint)size * sizeof(T), 16);
    tmp_buffer = _aligned_malloc((uint)size * sizeof(T), 16);
#else
    posix_memalign((void**)&data, 16, (uint)size*sizeof(T));
    posix_memalign((void**)&tmp_buffer, 16, (uint)size*sizeof(T));
#endif

    cgfassert(alignment(data, 16));
    cgfassert(alignment(tmp_buffer, 16));

    /*Initialize extra space to zero since the extra space is being
      used in almost all operations*/
    memset(data, 0, sizeof(T)*(uint)(size));

    /*Note, we only have to initialize the extra data here. When a
      vector is being copied, we copy the complete vector, including
      the extra padded data*/
  }

  template<class T>
  Vector<T>::Vector(const Vector<T>& v){
    origSize = v.origSize;
    size = v.size;

#ifdef _WIN32
    data = _aligned_malloc((uint)size * sizeof(T), 16);
    tmp_buffer = _aligned_malloc((uint)size * sizeof(T), 16);
#else
    posix_memalign((void**)&data, 16, (uint)size*sizeof(T));
    posix_memalign((void**)&tmp_buffer, 16, (uint)size*sizeof(T));
#endif

    cgfassert(alignment(data, 16));
    cgfassert(alignment(tmp_buffer, 16));

    memcpy(data, v.data, sizeof(T)*(uint)size);
  }

  template<class T>
  Vector<T>::~Vector(){
    if(data){
#ifdef _WIN32
      _aligned_free(data);
#else
      free(data);
#endif
    }
    if(tmp_buffer){
#ifdef _WIN32
      _aligned_free(tmp_buffer);
#else
      free(tmp_buffer);
#endif
    }
  }

  template<class T>
  Vector<T>& Vector<T>::operator=(const Vector<T>& v){
    if(this == &v){
      /*Self assignment*/
      return *this;
    }
    copy(&v);
    return *this;
  }

  template<class T>
  void Vector<T>::copy(const Vector<T>* src){
    if(origSize == src->origSize && size == src->size){
      /*No need to update storage*/
    }else{
      origSize = src->origSize;
      size = src->size;
      if(data){
#ifdef _WIN32
        _aligned_free(data);
#else
        free(data);
#endif
      }

      if(tmp_buffer){
#ifdef _WIN32
        _aligned_free(tmp_buffer);
#else
        free(tmp_buffer);
#endif
        tmp_buffer = 0;
      }

#ifdef _WIN32
      data = _aligned_malloc((uint)size * sizeof(T), 16);
      tmp_buffer = _aligned_malloc((uint)size * sizeof(T), 16);
#else
      posix_memalign((void**)&data, 16, (uint)size*sizeof(T));
      posix_memalign((void**)&tmp_buffer, 16, (uint)size*sizeof(T));
#endif

      cgfassert(alignment(data, 16));
      cgfassert(alignment(tmp_buffer, 16));
    }
    memcpy(data, src->data, sizeof(T)*(uint)size);
  }

  template<int N, class T>
  Vector<T> operator*(const SpMatrix<N, T>& m, const Vector<T>& v){
    if(!m.isFinalized()){
      error("Matrix is not finalized");
    }

    Vector<T> r(m.origHeight);
    spmv<N>(r, m, v);
    return r;
  }

  template<class T>
  std::ostream& operator<<(std::ostream& stream, const Vector<T>& v){
    stream << "vector size = " << v.size << " " << v.origSize << std::endl;
    std::ios_base::fmtflags origflags = stream.flags();
    stream.setf(std::ios_base::scientific);

    for(int i=0;i<v.origSize;i++){
      //stream << "v[" << i << "] = " << v.data[i] << "\t";
      stream << i << "\t" << v.data[i] << "\n";
    }
    stream << std::endl;
    stream.flags(origflags);
    return stream;
  }

  template<class T>
  void Vector<T>::randomShuffle(){
    std::random_shuffle(data, data+origSize);
  }

  int computeSteps(int size){
    /*Find a value, smaller than size, which is a power of 2*/
    if(size <= 1){
      return 0;
    }
    int steps = 1;
    while(steps*2<size){
      steps*=2;
    }
    return steps;
  }

  template<class T>
  T Vector<T>::sum()const{
    if(origSize == 0){
      return 0;
    }
    VectorRange range;
    range.startBlock = 0;
    range.endBlock = size;
    return sump(range);
  }

#ifdef USE_THREADS
  /*Parallel sum reduction*/
  template<class T>
  T Vector<T>::sum(T* sharedReductions, const Thread* caller,
                   const VectorRange* rg)const{
    /*Compute partial sum and store result and error in shared memory*/
    sharedReductions[caller->getId()] = sump(rg[caller->getId()]);
    caller->sync();

    /*Copy shared data to local memory and each thread computes the
      total sum*/
    T result = 0;
    T c = 0;

    for(int i=0;i<caller->getLastId();i++){
      T y = sharedReductions[i] - c;
      T t = result + y;
      c = ((t - result) - sharedReductions[i]) + c;
      result = t;
    }

    return result;
  }
#endif

  template<class T>
  T Vector<T>::sump(const VectorRange rg)const{
    if(rg.startBlock == rg.endBlock){
      /*Range is zero.*/
      return 0;
    }

    bool first = true;
    int steps = computeSteps(rg.endBlock - rg.startBlock);

    while(steps >= 1){
      for(int j=0;j<steps;j++){
        int idx1 = j + rg.startBlock;
        int idx2 = j + rg.startBlock + steps;

        if(first){
          if(idx2 < rg.endBlock){
            tmp_buffer[idx1] = data[idx1] + data[idx2];
          }else{
            tmp_buffer[idx1] = data[idx1];
          }
        }else{
          if(idx2 < rg.endBlock){
            tmp_buffer[idx1] += tmp_buffer[idx2];
          }
        }
      }
      steps/=2;
      first = false;
    }
    return tmp_buffer[rg.startBlock];
  }

#if 1
#ifdef SSE2
  template<>
  float Vector<float>::sump(const VectorRange rg)const{
    if(rg.startBlock == rg.endBlock){
      /*Range is zero.*/
      return 0;
    }

    cgfassert(rg.startBlock%16 == 0);
    cgfassert(rg.endBlock%16 == 0);

    int steps = computeSteps(rg.endBlock-rg.startBlock);
    bool first = true;

    while(steps >= 4){
      for(int j=0;j<steps/4;j++){
        int idx1 = rg.startBlock + j*4;
        int idx2 = rg.startBlock + j*4 + steps;

        if(first){
          if(idx2 < rg.endBlock){
            __m128 XMM1 = _mm_load_ps(data + idx1);
            __m128 XMM2 = _mm_load_ps(data + idx2);

            _mm_store_ps(tmp_buffer + idx1, _mm_add_ps(XMM1, XMM2));
          }else{
            __m128 XMM1 = _mm_load_ps(data + idx1);

            _mm_store_ps(tmp_buffer + idx1, XMM1);
          }
        }else{
          if(idx2 < rg.endBlock){
            __m128 XMM1 = _mm_load_ps(tmp_buffer + idx1);
            __m128 XMM2 = _mm_load_ps(tmp_buffer + idx2);

            _mm_store_ps(tmp_buffer + idx1, _mm_add_ps(XMM1, XMM2));
          }
        }
      }
      steps/=2;
      first = false;
    }
    return ((tmp_buffer[rg.startBlock+0] + tmp_buffer[rg.startBlock+2]) +
            (tmp_buffer[rg.startBlock+1] + tmp_buffer[rg.startBlock+3]));
  }
#endif
#endif


#if 1
#ifdef SSE2
  template<>
  double Vector<double>::sump(const VectorRange rg)const{
    if(rg.startBlock == rg.endBlock){
      /*Range is zero.*/
      return 0;
    }

    cgfassert(rg.startBlock%16 == 0);
    cgfassert(rg.endBlock%16 == 0);

    int steps = computeSteps(rg.endBlock-rg.startBlock);
    bool first = true;

    while(steps >= 4){
      for(int j=0;j<steps/4;j++){
        int idx1 = rg.startBlock + j*4;
        int idx2 = rg.startBlock + j*4 + steps;

        if(first){
          if(idx2 < rg.endBlock){
            __m128d XMM1  = _mm_load_pd(data + idx1 + 0);
            __m128d XMM12 = _mm_load_pd(data + idx1 + 2);
            __m128d XMM2  = _mm_load_pd(data + idx2 + 0);
            __m128d XMM22 = _mm_load_pd(data + idx2 + 2);

            _mm_store_pd(tmp_buffer + idx1 + 0, _mm_add_pd(XMM1, XMM2));
            _mm_store_pd(tmp_buffer + idx1 + 2, _mm_add_pd(XMM12, XMM22));
          }else{
            __m128d XMM1 = _mm_load_pd(data + idx1 + 0);
            __m128d XMM2 = _mm_load_pd(data + idx1 + 2);

            _mm_store_pd(tmp_buffer + idx1 + 0, XMM1);
            _mm_store_pd(tmp_buffer + idx1 + 2, XMM2);
          }
        }else{
          if(idx2 < rg.endBlock){
            __m128d XMM1  = _mm_load_pd(tmp_buffer + idx1 + 0);
            __m128d XMM12 = _mm_load_pd(tmp_buffer + idx1 + 2);
            __m128d XMM2  = _mm_load_pd(tmp_buffer + idx2 + 0);
            __m128d XMM22 = _mm_load_pd(tmp_buffer + idx2 + 2);

            _mm_store_pd(tmp_buffer + idx1 + 0, _mm_add_pd(XMM1, XMM2));
            _mm_store_pd(tmp_buffer + idx1 + 2, _mm_add_pd(XMM12, XMM22));
          }
        }
      }
      steps/=2;
      first = false;
    }

    return ((tmp_buffer[rg.startBlock+0] + tmp_buffer[rg.startBlock+2]) +
            (tmp_buffer[rg.startBlock+1] + tmp_buffer[rg.startBlock+3]));
  }
#endif
#endif

  template<class T>
  T Vector<T>::operator*(const Vector<T>& v) const{
    cgfassert(size == v.size);

    if(size == 0){
      /*Range is zero.*/
      return 0;
    }

    bool first = true;
    int steps = computeSteps(size);

    while(steps >= 1){
      for(int j=0;j<steps;j++){
        int idx1 = j;
        int idx2 = j + steps;

        if(first){
          if(idx2 < size){
            tmp_buffer[idx1] = ((data[idx1] * v.data[idx1]) +
                                (data[idx2] * v.data[idx2]));
          }else{
            tmp_buffer[idx1] = data[idx1]*v.data[idx1];
          }
        }else{
          if(idx2 < size){
            tmp_buffer[idx1] += tmp_buffer[idx2];
          }
        }
      }
      steps/=2;
      first = false;
    }
    return tmp_buffer[0];
  }

#if 1
#ifdef SSE2
  template<>
  float Vector<float>::operator*(const Vector<float>& v)const{
    if(size == 0){
      /*Range is zero.*/
      return 0;
    }

    cgfassert(size == v.size);

    int steps = computeSteps(size);
    bool first = true;

    while(steps >= 4){
      for(int j=0;j<steps/4;j++){
        int idx1 = j*4;
        int idx2 = j*4 + steps;

        if(first){
          if(idx2 < size){
            __m128 XMM1 = _mm_load_ps(data + idx1);
            __m128 XMM2 = _mm_load_ps(data + idx2);

            __m128 XMM3 = _mm_load_ps(v.data + idx1);
            __m128 XMM4 = _mm_load_ps(v.data + idx2);

            _mm_store_ps(tmp_buffer + idx1, _mm_add_ps(_mm_mul_ps(XMM1, XMM3),
                                                       _mm_mul_ps(XMM2, XMM4)));
          }else{
            __m128 XMM1 = _mm_load_ps(data + idx1);
            __m128 XMM2 = _mm_load_ps(v.data + idx1);

            _mm_store_ps(tmp_buffer + idx1, _mm_mul_ps(XMM1, XMM2));
          }
        }else{
          if(idx2 < size){
            __m128 XMM1 = _mm_load_ps(tmp_buffer + idx1);
            __m128 XMM2 = _mm_load_ps(tmp_buffer + idx2);

            _mm_store_ps(tmp_buffer + idx1, _mm_add_ps(XMM1, XMM2));
          }
        }
      }
      steps/=2;
      first = false;
    }
    return ((tmp_buffer[0] + tmp_buffer[2]) +
            (tmp_buffer[1] + tmp_buffer[3]));
  }

  template<>
  double Vector<double>::operator*(const Vector<double>& v)const{
    if(size == 0){
      /*Range is zero.*/
      return 0;
    }

    cgfassert(size == v.size);

    int steps = computeSteps(size);
    bool first = true;

    while(steps >= 4){
      for(int j=0;j<steps/4;j++){
        int idx1 = j*4;
        int idx2 = j*4 + steps;

        if(first){
          if(idx2 < size){
            __m128d XMM1  = _mm_load_pd(data + idx1 + 0);
            __m128d XMM12 = _mm_load_pd(data + idx1 + 2);
            __m128d XMM2  = _mm_load_pd(data + idx2 + 0);
            __m128d XMM22 = _mm_load_pd(data + idx2 + 2);


            __m128d XMM3  = _mm_load_pd(v.data + idx1 + 0);
            __m128d XMM32 = _mm_load_pd(v.data + idx1 + 2);
            __m128d XMM4  = _mm_load_pd(v.data + idx2 + 0);
            __m128d XMM42 = _mm_load_pd(v.data + idx2 + 2);


            _mm_store_pd(tmp_buffer + idx1 + 0, _mm_add_pd(_mm_mul_pd(XMM1,
                                                                      XMM3),
                                                           _mm_mul_pd(XMM2,
                                                                      XMM4)));

            _mm_store_pd(tmp_buffer + idx1 + 2, _mm_add_pd(_mm_mul_pd(XMM12,
                                                                      XMM32),
                                                           _mm_mul_pd(XMM22,
                                                                      XMM42)));
          }else{
            __m128d XMM1 = _mm_load_pd(data + idx1 + 0);
            __m128d XMM2 = _mm_load_pd(data + idx1 + 2);

            __m128d XMM3 = _mm_load_pd(v.data + idx1 + 0);
            __m128d XMM4 = _mm_load_pd(v.data + idx1 + 2);

            _mm_store_pd(tmp_buffer + idx1 + 0, _mm_mul_pd(XMM1, XMM3));
            _mm_store_pd(tmp_buffer + idx1 + 2, _mm_mul_pd(XMM2, XMM4));
          }
        }else{
          if(idx2 < size){
            __m128d XMM1  = _mm_load_pd(tmp_buffer + idx1 + 0);
            __m128d XMM12 = _mm_load_pd(tmp_buffer + idx1 + 2);
            __m128d XMM2  = _mm_load_pd(tmp_buffer + idx2 + 0);
            __m128d XMM22 = _mm_load_pd(tmp_buffer + idx2 + 2);

            _mm_store_pd(tmp_buffer + idx1 + 0, _mm_add_pd(XMM1, XMM2));
            _mm_store_pd(tmp_buffer + idx1 + 2, _mm_add_pd(XMM12, XMM22));
          }
        }
      }
      steps/=2;
      first = false;
    }

    return ((tmp_buffer[0] + tmp_buffer[2]) +
            (tmp_buffer[1] + tmp_buffer[3]));
  }
#endif
#endif

  /*r = a - b*/
  template<class T>
  void Vector<T>::sub(Vector<T>& r, const Vector<T>& a, const Vector<T>& b){
    for(int i=0;i<a.size/16;i++){
      spmatrix_block_sub<4, T>(r.data+i*16, a.data+i*16, b.data+i*16);
    }
  }

  template<class T>
  void Vector<T>::subs(Vector<T>& r, const Vector<T>& a, T f){
    for(int i=0;i<a.size/16;i++){
      spmatrix_block_subs<4, T>(r.data+i*16, a.data+i*16, f);
    }
  }

  /*r = a + b*/
  template<class T>
  void Vector<T>::add(Vector<T>& r, const Vector<T>& a, const Vector<T>& b){
    for(int i=0;i<a.size/16;i++){
      spmatrix_block_add<4, T>(r.data+i*16, a.data+i*16, b.data+i*16);
    }
  }

  template<class T>
  void Vector<T>::adds(Vector<T>& r, const Vector<T>& a, T f){
    for(int i=0;i<a.size/16;i++){
      spmatrix_block_adds<4, T>(r.data+i*16, a.data+i*16, f);
    }
  }

  /*r = a * b + c*/
  template<class T>
  void Vector<T>::madd(Vector<T>& r, const Vector<T>& a, const Vector<T>& b,
                       const Vector<T>& c){
    for(int i=0;i<a.size/16;i++){
      spmatrix_block_madd<4, T>(r.data+i*16, a.data+i*16, b.data+i*16,
                                c.data+i*16);
    }
  }

  /*r = f * b + c*/
  template<class T>
  void Vector<T>::mfadd(Vector<T>& r, T f, const Vector<T>& b,
                        const Vector<T>& c){
    for(int i=0;i<b.size/16;i++){
      spmatrix_block_msadd<4, T>(r.data+i*16, f, b.data+i*16, c.data+i*16);
    }
  }

  /*r = f * b + c + d*/
  template<class T>
  void Vector<T>::mfadd2(Vector<T>& r, T f, const Vector<T>& b,
                         const Vector<T>& c, const Vector<T>& d){
    for(int i=0;i<b.size/16;i++){
      spmatrix_block_msadd2<4, T>(r.data+i*16, f, b.data+i*16, c.data+i*16,
                                  d.data+i*16);
    }
  }

  /*r = a .* b*/
  template<class T>
  void Vector<T>::mul(Vector<T>& r, const Vector<T>& a, const Vector<T>& b){
    for(int i=0;i<a.size/16;i++){
      spmatrix_block_mul<4, T>(r.data+i*16, a.data+i*16, b.data+i*16);
    }
  }

  /*r = a .* b*/
  template<class T>
  void Vector<T>::div(Vector<T>& r, const Vector<T>& a, const Vector<T>& b){
    for(int i=0;i<a.size/16;i++){
      spmatrix_block_div<4, T>(r.data+i*16, a.data+i*16, b.data+i*16);
    }
  }

  /*r = a * f*/
  template<class T>
  void Vector<T>::mulf(Vector<T>& r, const Vector<T>& a, T f){
    for(int i=0;i<a.size/16;i++){
      spmatrix_block_muls<4, T>(r.data+i*16, a.data+i*16, f);
    }
  }

  /*Partial functions*/

  /*r = a - b*/
  template<class T>
  void Vector<T>::subp(Vector<T>& r, const Vector<T>& a, const Vector<T>& b,
                       const VectorRange rg){
    RANGE_ASSERT_AB;

    for(int i=rg.startBlock/16; i<rg.endBlock/16; i++){
      if(i < a.size/16){
        spmatrix_block_sub<4, T>(r.data+i*16, a.data+i*16, b.data+i*16);
      }
    }
  }

  template<class T>
  void Vector<T>::subsp(Vector<T>& r, const Vector<T>& a, T f,
                        const VectorRange rg){
    RANGE_ASSERT_A;

    for(int i=rg.startBlock/16; i<rg.endBlock/16; i++){
      if(i < a.size/16){
        spmatrix_block_subs<4, T>(r.data+i*16, a.data+i*16, f);
      }
    }
  }

  /*r = a + b*/
  template<class T>
  void Vector<T>::addp(Vector<T>& r, const Vector<T>& a, const Vector<T>& b,
                       const VectorRange rg){
    RANGE_ASSERT_AB;

    for(int i=rg.startBlock/16;i<rg.endBlock/16;i++){
      if(i<a.size/16){
        spmatrix_block_add<4, T>(r.data+i*16, a.data+i*16, b.data+i*16);
      }
    }
  }

  /*r = a + f*/
  template<class T>
  void Vector<T>::addsp(Vector<T>& r, const Vector<T>& a, T f,
                        const VectorRange rg){
    RANGE_ASSERT_A;

    for(int i=rg.startBlock/16; i<rg.endBlock/16; i++){
      if(i < a.size/16){
        spmatrix_block_adds<4, T>(r.data+i*16, a.data+i*16, f);
      }
    }
  }

  /*r = a * b + c*/
  template<class T>
  void Vector<T>::maddp(Vector<T>& r, const Vector<T>& a,
                        const Vector<T>& b, const Vector<T>& c,
                        const VectorRange rg){
    RANGE_ASSERT_AB;

    for(int i=rg.startBlock/16;i<rg.endBlock/16;i++){
      if(i<a.size/16){
        spmatrix_block_madd<4, T>(r.data+i*16, a.data+i*16, b.data+i*16,
                                  c.data+i*16);
      }
    }
  }

  /*r = f * a + b*/
  template<class T>
  void Vector<T>::mfaddp(Vector<T>& r, T f,
                         const Vector<T>& a,
                         const Vector<T>& b,
                         const VectorRange rg){
    RANGE_ASSERT_AB;

    for(int i=rg.startBlock/16;i<rg.endBlock/16;i++){
      if(i<a.size/16){
        spmatrix_block_msadd<4, T>(r.data+i*16, f, a.data+i*16, b.data+i*16);
      }
    }
  }

  /*r = f * a + b + d*/
  template<class T>
  void Vector<T>::mfadd2p(Vector<T>& r, T f,
                          const Vector<T>& a,
                          const Vector<T>& b,
                          const Vector<T>& d,
                          const VectorRange rg){
    RANGE_ASSERT_AB;

    for(int i=rg.startBlock/16;i<rg.endBlock/16;i++){
      if(i<a.size/16){
        spmatrix_block_msadd2<4, T>(r.data+i*16, f, a.data+i*16, b.data+i*16,
                                    d.data+i*16);
      }
    }
  }

  /*r = a .* b*/
  template<class T>
  void Vector<T>::mulp(Vector<T>& r, const Vector<T>& a,
                       const Vector<T>& b,
                       const VectorRange rg){
    RANGE_ASSERT_AB;

    for(int i=rg.startBlock/16;i<rg.endBlock/16;i++){
      if(i<a.size/16){
        spmatrix_block_mul<4, T>(r.data+i*16, a.data+i*16, b.data+i*16);
      }
    }
  }

  /*r = a * f*/
  template<class T>
  void Vector<T>::mulfp(Vector<T>& r, const Vector<T>& a, T f,
                        const VectorRange rg){
    RANGE_ASSERT_A;

    for(int i=rg.startBlock/16;i<rg.endBlock/16;i++){
      if(i<a.size/16){
        spmatrix_block_muls<4, T>(r.data+i*16, a.data+i*16, f);
      }
    }
  }

  template<class T>
  Vector<T>& Vector<T>::operator+=(const Vector<T>& v){
    Vector<T>::add(*this, *this, v);
    return *this;
  }

  template<class T>
  Vector<T>& Vector<T>::operator-=(const Vector<T>& v){
    Vector<T>::sub(*this, *this, v);
    return *this;
  }

#if 0
  template<class T>
  Vector<T>& Vector<T>::operator*(const Vector<T>& v)const{
    Vector<T> r;
    Vector<T>::mul(r, *this, v);
    return r;
  }

  template<class T>
  Vector<T>& Vector<T>::operator/(const Vector<T>& v)const{
    Vector<T> r;
    Vector<T>::div(r, *this, v);
    return r;
  }
#endif

  template<class T>
  Vector<T>& Vector<T>::operator*=(const Vector<T>& v){
    Vector<T>::mul(*this, *this, v);
    return *this;
  }

  template<class T>
  Vector<T>& Vector<T>::operator/=(const Vector<T>& v){
    Vector<T>::div(*this, *this, v);
    return *this;
  }


  template<class T>
  T Vector<T>::length2() const{
    if(origSize == 0){
      return 0;
    }

    int steps = computeSteps(size);
    bool first = true;

    while(steps >= 1){
      for(int j=0;j<steps;j++){
        int idx1 = j;
        int idx2 = j + steps;

        if(first){
          if(idx2 < size){
            tmp_buffer[idx1] =
              data[idx1] * data[idx1] + data[idx2] * data[idx2];
          }else{
            tmp_buffer[idx1] =
              data[idx1] * data[idx1];
          }
        }else{
          if(idx2 < size){
            tmp_buffer[idx1] += tmp_buffer[idx2];
          }
        }
      }
      steps/=2;
      first = false;
    }

    return tmp_buffer[0];
  }

#ifdef SSE2
  template<>
  float Vector<float>::length2() const{
    cgfassert(size%16 == 0);

    int steps = computeSteps(size);
    bool first = true;

    while(steps >= 4){
      for(int j=0;j<steps/4;j++){
        int idx1 = j*4;
        int idx2 = j*4 + steps;

        if(first){
          if(idx2 < size){
            __m128 XMM1 = _mm_load_ps(data + idx1);
            __m128 XMM2 = _mm_load_ps(data + idx2);
            XMM1 = _mm_mul_ps(XMM1, XMM1);
            XMM2 = _mm_mul_ps(XMM2, XMM2);

            XMM1 = _mm_add_ps(XMM1, XMM2);

            _mm_store_ps(tmp_buffer + idx1, XMM1);
          }else{
            __m128 XMM1 = _mm_load_ps(data + idx1);
            XMM1 = _mm_mul_ps(XMM1, XMM1);
            _mm_store_ps(tmp_buffer + idx1, XMM1);
          }
        }else{
          if(idx2 < size){
            __m128 XMM1 = _mm_load_ps(tmp_buffer + idx1);
            __m128 XMM2 = _mm_load_ps(tmp_buffer + idx2);
            XMM1 = _mm_add_ps(XMM1, XMM2);
            _mm_store_ps(tmp_buffer + idx1, XMM1);
          }
        }
      }
      steps/=2;
      first = false;
    }

    return ((tmp_buffer[0] + tmp_buffer[2]) +
            (tmp_buffer[1] + tmp_buffer[3]));
  }
#endif

  template<class T>
  bool operator==(const Vector<T>& a, T n){
    error("Not implemented yet");
    return false;
  }

  template<class T>
  bool operator!=(const Vector<T>& a, T n){
    error("Not implemented yet");
    return false;
  }

  template<class T>
  bool operator==(T n, const Vector<T>& a){
    error("Not implemented yet");
    return false;
  }

  template<class T>
  bool operator!=(T n, const Vector<T>& a){
    error("Not implemented yet");
    return false;
  }

  template<class T>
  bool operator<(const Vector<T>& a, T n){
    error("Not implemented yet");
    return false;
  }

  template<class T>
  bool operator<=(const Vector<T>& a, T n){
    error("Not implemented yet");
    return false;
  }

  template<class T>
  bool operator>(const Vector<T>& a, T n){
    error("Not implemented yet");
    return false;
  }

  template<class T>
  bool operator>=(const Vector<T>& a, T n){
    error("Not implemented yet");
    return false;
  }

  template<class T>
  bool operator<(T n, const Vector<T>& a){
    error("Not implemented yet");
    return false;
  }

  template<class T>
  bool operator<=(T n, const Vector<T>& a){
    error("Not implemented yet");
    return false;
  }

  template<class T>
  bool operator>(T n, const Vector<T>& a){
    error("Not implemented yet");
    return false;
  }

  template<class T>
  bool operator>=(T n, const Vector<T>& a){
    error("Not implemented yet");
    return false;
  }

  template<class T>
  Vector<T> lo(const Vector<T>& a, Vector<T>& b){
    error("Not implemented yet");
    return a;
  }

  template<class T>
  Vector<T> hi(const Vector<T>& a, Vector<T>& b){
    error("Not implemented yet");
    return a;
  }

  template class Vector<float>;
  template class Vector<double>;

  template Vector<float>  operator*(const Vector<float>& a, float n);
  template Vector<double> operator*(const Vector<double>& a, double n);
  template Vector<float>  operator*(float n, const Vector<float>& a);
  template Vector<double> operator*(double n, const Vector<double>& a);

  template Vector<float>  operator/(const Vector<float>& a, float n);
  template Vector<double> operator/(const Vector<double>& a, double n);
  template Vector<float>  operator/(float n, const Vector<float>& a);
  template Vector<double> operator/(double n, const Vector<double>& a);

  template Vector<float> operator*(const SpMatrix<1, float>&m, const Vector<float>&v);
  template Vector<float> operator*(const SpMatrix<2, float>&m, const Vector<float>&v);
  template Vector<float> operator*(const SpMatrix<4, float>&m, const Vector<float>&v);
  template Vector<float> operator*(const SpMatrix<8, float>&m, const Vector<float>&v);
  //template Vector operator*(const SpMatrix<16>&m, const Vector&v);

  template Vector<double> operator*(const SpMatrix<1, double>&m, const Vector<double>&v);
  template Vector<double> operator*(const SpMatrix<2, double>&m, const Vector<double>&v);
  template Vector<double> operator*(const SpMatrix<4, double>&m, const Vector<double>&v);
  template Vector<double> operator*(const SpMatrix<8, double>&m, const Vector<double>&v);
  //template Vector operator*(const SpMatrix<16>&m, const Vector&v);

  template
  std::ostream& operator<< <float>(std::ostream& stream, const Vector<float>& v);

  template
  std::ostream& operator<< <double>(std::ostream& stream, const Vector<double>& v);
}
