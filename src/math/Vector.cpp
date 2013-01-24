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
#include "core/Thread.hpp"

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

#define RANGE_ASSERT_A					\
  cgfassert((r.size == a.size));			\
  cgfassert(rg.startBlock % 16 == 0);			\
  cgfassert(rg.endBlock   % 16 == 0);  

#define RANGE_ASSERT_AB					\
  RANGE_ASSERT_A					\
  cgfassert((a.size == b.size));

namespace CGF{
  /*The size of a vector is always a multiple of 16*/
  template<class T>
  Vector<T>::Vector(ulong s){
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
    
    data = new T[size];
    tmp_buffer = new T[size];

    /*Initialize extra space to zero since the extra space is being
      used in almost all operations*/
    memset(data, 0, sizeof(T)*(size));
    
    /*Note, we only have to initialize the extra data here. When a
      vector is being copied, we copy the complete vector, including
      the extra padded data*/
  }
  
  template<class T>
  Vector<T>::Vector(const Vector<T>& v){
    origSize = v.origSize;
    size = v.size;
    data = new T[this->size];
    memcpy(data, v.data, sizeof(T)*size);
    tmp_buffer = new T[this->size];
  }
  
  template<class T>
  Vector<T>::~Vector(){
    if(data){
      delete [] data;
    }
    if(tmp_buffer){
      delete [] tmp_buffer;
    }
  }
  
  template<class T>
  Vector<T>& Vector<T>::operator=(const Vector<T>& v){
    if(this == &v){
      /*Self assignment*/
      return *this;
    }
    
    origSize = v.origSize;
    size = v.size;
    if(data){
      delete[] data;
    }

    if(tmp_buffer){
      delete [] tmp_buffer;
      tmp_buffer = 0;
    }
    data = new T[size];
    tmp_buffer = new T[size];
    memcpy(data, v.data, sizeof(T)*size);
    return *this;
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

    for(ulong i=0;i<v.origSize;i++){
      //stream << "v[" << i << "] = " << v.data[i] << "\t";
      stream << i << "\t" << v.data[i] << "\n";
    }
    stream << std::endl;
    stream.flags(origflags);
    return stream;
  }


  /*@TODO: Sum operators must be checked*/
  template<class T>
  T Vector<T>::sum()const{
    if(origSize == 0){
      return 0;
    }
    if(!tmp_buffer){
      tmp_buffer = new T[size];
    }
    
#if 1
    memcpy(tmp_buffer, data, sizeof(T)*size);
    memset(tmp_buffer+origSize, 0, sizeof(T)*(size-origSize));
#else
    for(uint i=0;i<size;i++){
      if(i < origSize){
	tmp_buffer[i] = data[i];
      }else{
	tmp_buffer[i] = 0;
      }
    }
#endif

    /*Find a value, smaller than size, which is a power of 2*/
    uint steps = size/2;
    uint pwr = 1;
    while(steps >1){
      steps/=2;
      pwr++;
    }
    steps = 1<<pwr;

    while(steps >= 1){
      for(uint j=0;j<steps;j++){
	uint idx1 = j;
	uint idx2 = j + steps;

	if(idx2 < size){
	  tmp_buffer[idx1] += tmp_buffer[idx2];
	}
      } 
      steps/=2;
    }

    return tmp_buffer[0];
  }

#if 0
#ifdef SSE2
  template<>
  float Vector<float>::sum()const{
    float res[4];

    //error();

    __m128 XMM0 = _mm_set_ps1(0);

    for(uint i=0;i<size/16;i++){
      __m128 XMM1 = _mm_load_ps(data + i * 16 + 0);
      __m128 XMM2 = _mm_load_ps(data + i * 16 + 4);
      __m128 XMM3 = _mm_load_ps(data + i * 16 + 8);
      __m128 XMM4 = _mm_load_ps(data + i * 16 + 12);
      
      XMM1 = _mm_add_ps(XMM1, XMM2);
      XMM2 = _mm_add_ps(XMM3, XMM4);

      XMM1 = _mm_add_ps(XMM1, XMM2);
      
      XMM0 = _mm_add_ps(XMM0, XMM1);
    }
    _mm_store_ps(res, XMM0);
    return res[0] + res[1] + res[2] + res[3];
  }
#endif
#endif

  template<class T>
  T Vector<T>::sum(T* sharedReductions, const Thread* caller, 
		   const VectorRange* rg)const{
    sharedReductions[caller->getId()] = sump(rg[caller->getId()]);
    caller->sync();

    T result = 0;
    for(uint i=0;i<caller->getLastId();i++){
      result += sharedReductions[i];
    }
    return result;
  }

  template<class T>
  T Vector<T>::sump(const VectorRange rg)const{
    if(rg.startBlock == rg.endBlock){
      /*Range is zero.*/
      return 0;
    }

    memcpy(tmp_buffer+rg.startBlock, data + rg.startBlock, 
	   sizeof(T)*(rg.endBlock - rg.startBlock));

    /*Find a value, smaller than size, which is a power of 2*/
    uint steps = size/2;
    uint pwr = 1;
    while(steps >1){
      steps/=2;
      pwr++;
    }
    steps = 1<<(pwr);

    while(steps >= 1){
      for(uint j=0;j<steps;j++){
	uint idx1 = j + rg.startBlock;
	uint idx2 = j + rg.startBlock + steps;

	if(idx2 < rg.endBlock){
	  tmp_buffer[idx1] += tmp_buffer[idx2];
	}
      } 
      steps/=2;
    }

    return tmp_buffer[rg.startBlock];
  }

#if 0
#ifdef SSE2
  template<>
  float Vector<float>::sump(const VectorRange rg)const{
    cgfassert(rg.startBlock%16 == 0);
    cgfassert(rg.endBlock%16 == 0);
    float res[4];

    __m128 XMM0 = _mm_set_ps1(0);

    for(uint i=rg.startBlock/16;i<rg.endBlock/16;i++){
      __m128 XMM1 = _mm_load_ps(data + i * 16 + 0);
      __m128 XMM2 = _mm_load_ps(data + i * 16 + 4);
      __m128 XMM3 = _mm_load_ps(data + i * 16 + 8);
      __m128 XMM4 = _mm_load_ps(data + i * 16 + 12);
      
      XMM1 = _mm_add_ps(XMM1, XMM2);
      XMM2 = _mm_add_ps(XMM3, XMM4);

      XMM1 = _mm_add_ps(XMM1, XMM2);
      
      XMM0 = _mm_add_ps(XMM0, XMM1);
    }
    _mm_store_ps(res, XMM0);
    return res[0] + res[1] + res[2] + res[3];
  }
#endif
#endif

  template<class T>
  T Vector<T>::operator*(const Vector<T>& v) const{
    error("Not implemented yet");
    return 0;
  }

#ifdef SSE2
  template<>
  float Vector<float>::operator*(const Vector<float>& v) const{
    /**/
    error("Not tested yet");
    PRINT_FUNCTION;
    uint steps = size/16;
    //uint remainder = size%16;
    float res[4];

    __m128 XMM8 = _mm_set_ps1(0);
    uint i = 0;
    for(;i<steps;i++){
      __m128 XMM0 = _mm_load_ps(data + i * 16 + 0);
      __m128 XMM1 = _mm_load_ps(data + i * 16 + 4);
      __m128 XMM2 = _mm_load_ps(data + i * 16 + 8);
      __m128 XMM3 = _mm_load_ps(data + i * 16 + 12);

      __m128 XMM4 = _mm_load_ps(v.data + i * 16 + 0);
      __m128 XMM5 = _mm_load_ps(v.data + i * 16 + 4);
      __m128 XMM6 = _mm_load_ps(v.data + i * 16 + 8);
      __m128 XMM7 = _mm_load_ps(v.data + i * 16 + 12);

      XMM0 = _mm_mul_ps(XMM0, XMM4);
      XMM1 = _mm_mul_ps(XMM1, XMM5);
      XMM2 = _mm_mul_ps(XMM2, XMM6);
      XMM3 = _mm_mul_ps(XMM3, XMM7);

      XMM0 = _mm_add_ps(XMM0, XMM1);
      XMM1 = _mm_add_ps(XMM2, XMM3);

      XMM0 = _mm_add_ps(XMM0, XMM1);

      XMM8 = _mm_add_ps(XMM0, XMM8);
    }

    _mm_store_ps(res, XMM8);
    res[0] += res[1] + res[2] + res[3];

    i*=16;
    /*Add remaining*/
    for(;i<size;i++){
      res[0] += data[i] * v.data[i];
    }
    return res[0];
  }
#endif  

  /*r = a - b*/
  template<class T>
  void Vector<T>::sub(Vector<T>& r, const Vector<T>& a, const Vector<T>& b){
    for(uint i=0;i<a.size/16;i++){
      spmatrix_block_sub<4, T>(r.data+i*16, a.data+i*16, b.data+i*16);
    }
  }

  template<class T>
  void Vector<T>::subs(Vector<T>& r, const Vector<T>& a, T f){
    for(uint i=0;i<a.size/16;i++){
      spmatrix_block_subs<4, T>(r.data+i*16, a.data+i*16, f);
    }
  }

  /*r = a + b*/
  template<class T>
  void Vector<T>::add(Vector<T>& r, const Vector<T>& a, const Vector<T>& b){
    for(uint i=0;i<a.size/16;i++){
      spmatrix_block_add<4, T>(r.data+i*16, a.data+i*16, b.data+i*16);
    }
  }

  template<class T>
  void Vector<T>::adds(Vector<T>& r, const Vector<T>& a, T f){
    for(uint i=0;i<a.size/16;i++){
      spmatrix_block_adds<4, T>(r.data+i*16, a.data+i*16, f);
    }
  }

  /*r = a * b + c*/
  template<class T>
  void Vector<T>::madd(Vector<T>& r, const Vector<T>& a, const Vector<T>& b,
		       const Vector<T>& c){
    for(uint i=0;i<a.size/16;i++){
      spmatrix_block_madd<4, T>(r.data+i*16, a.data+i*16, b.data+i*16, 
				c.data+i*16);
    }
  }

  /*r = f * b + c*/
  template<class T>
  void Vector<T>::mfadd(Vector<T>& r, T f, const Vector<T>& b, 
			const Vector<T>& c){
    for(uint i=0;i<b.size/16;i++){
      spmatrix_block_msadd<4, T>(r.data+i*16, f, b.data+i*16, c.data+i*16);
    }
  }

  /*r = a .* b*/
  template<class T>
  void Vector<T>::mul(Vector<T>& r, const Vector<T>& a, const Vector<T>& b){
    for(uint i=0;i<a.size/16;i++){
      spmatrix_block_mul<4, T>(r.data+i*16, a.data+i*16, b.data+i*16);
    }
  }

  /*r = a .* b*/
  template<class T>
  void Vector<T>::div(Vector<T>& r, const Vector<T>& a, const Vector<T>& b){
    for(uint i=0;i<a.size/16;i++){
      spmatrix_block_div<4, T>(r.data+i*16, a.data+i*16, b.data+i*16);
    }
  }

  /*r = a * f*/
  template<class T>
  void Vector<T>::mulf(Vector<T>& r, const Vector<T>& a, T f){
    for(uint i=0;i<a.size/16;i++){
      spmatrix_block_muls<4, T>(r.data+i*16, a.data+i*16, f);
    }
  }

  /*Partial functions*/

  /*r = a - b*/
  template<class T>
  void Vector<T>::subp(Vector<T>& r, const Vector<T>& a, const Vector<T>& b, 
		    const VectorRange rg){
    RANGE_ASSERT_AB;

    for(uint i=rg.startBlock/16; i<rg.endBlock/16; i++){
      if(i < a.size/16){
	spmatrix_block_sub<4, T>(r.data+i*16, a.data+i*16, b.data+i*16);
      }
    }
  }

  template<class T>
  void Vector<T>::subsp(Vector<T>& r, const Vector<T>& a, T f,
			const VectorRange rg){
    RANGE_ASSERT_A;

    for(uint i=rg.startBlock/16; i<rg.endBlock/16; i++){
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

    for(uint i=rg.startBlock/16;i<rg.endBlock/16;i++){
      if(i<a.size/16){
	spmatrix_block_add<4, T>(r.data+i*16, a.data+i*16, b.data+i*16);
      }
    }
  }

  template<class T>
  void Vector<T>::addsp(Vector<T>& r, const Vector<T>& a, T f,
			const VectorRange rg){
    RANGE_ASSERT_A;

    for(uint i=rg.startBlock/16; i<rg.endBlock/16; i++){
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

    for(uint i=rg.startBlock/16;i<rg.endBlock/16;i++){
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

    for(uint i=rg.startBlock/16;i<rg.endBlock/16;i++){
      if(i<a.size/16){
	spmatrix_block_msadd<4, T>(r.data+i*16, f, a.data+i*16, b.data+i*16);
      }
    }
  }


  /*r = a .* b*/
  template<class T>
  void Vector<T>::mulp(Vector<T>& r, const Vector<T>& a, 
		       const Vector<T>& b, 
		       const VectorRange rg){
    RANGE_ASSERT_AB;

    for(uint i=rg.startBlock/16;i<rg.endBlock/16;i++){
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

    for(uint i=rg.startBlock/16;i<rg.endBlock/16;i++){
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
#if 0
    cgfassert(size == v.size);
    for(ulong i=0;i<size;i++){
      data[i]/=v.data[i];
    }
#else
    Vector<T>::div(*this, *this, v);
#endif
    return *this;
  }


  template<class T>
  T Vector<T>::length2() const{
    if(origSize == 0){
      return 0;
    }
#if 1
    memcpy(tmp_buffer, data, sizeof(T)*size);
    memset(tmp_buffer+origSize, 0, sizeof(T)*(size-origSize));
#else
    for(uint i=0;i<size;i++){
      if(i < origSize){
	tmp_buffer[i] = data[i];
      }else{
	tmp_buffer[i] = 0;
      }
    }
#endif

    for(uint i=0;i<size;i++){
      tmp_buffer[i] *= tmp_buffer[i];
    }

    /*Find a value, smaller than size, which is a power of 2*/
    uint steps = size/2;
    uint pwr = 1;
    while(steps >1){
      steps/=2;
      pwr++;
    }
    steps = 1<<pwr;


    while(steps >= 1){
      for(uint j=0;j<steps;j++){
	uint idx1 = j;
	uint idx2 = j + steps;

	if(idx2 < size){
	  tmp_buffer[idx1] += tmp_buffer[idx2];
	}
      } 
      steps/=2;
    }

    return tmp_buffer[0];
  }

#if 0
#ifdef SSE2
  template<>
  float Vector<float>::length2() const{
    float res[4];
    __m128 sum = _mm_set_ps(0,0,0,0);
    for(ulong i=0;i<size/16;i++){
      __m128 XMM0 = _mm_load_ps(  data+i*16 + 0);
      __m128 XMM1 = _mm_load_ps(  data+i*16 + 4);
      __m128 XMM2 = _mm_load_ps(  data+i*16 + 8);
      __m128 XMM3 = _mm_load_ps(  data+i*16 + 12);
      
      XMM0 = _mm_mul_ps(XMM0, XMM0);
      XMM1 = _mm_mul_ps(XMM1, XMM1);
      XMM2 = _mm_mul_ps(XMM2, XMM2);
      XMM3 = _mm_mul_ps(XMM3, XMM3);
      
      XMM0 = _mm_add_ps(XMM0,  XMM1);	
      XMM1 = _mm_add_ps(XMM2,  XMM3);
      XMM0 = _mm_add_ps(XMM0,  XMM1);
      
      sum = _mm_add_ps(sum,  XMM0);
    }
    _mm_store_ps(res, sum);      
    return res[0] + res[1] + res[2] + res[3];
  }
#endif
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
