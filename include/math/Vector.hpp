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

#ifndef VECTOR_HPP
#define VECTOR_HPP
#ifdef SSE2
#include <xmmintrin.h>
#endif

#include "core/cgfdefs.hpp"
#include "math/SpMatrix.hpp"
#include <math.h>
#include <ostream>
#include <string.h>

namespace CGF{

  //#define VEC_BLOCK_SIZE SPM_BLK_DIM

  class BlockDiagMatrix;

  class Thread;

  template<class T=float>
  class CGFAPI Vector{
  public:
    template<int M, class TT>
      friend class SpMatrix;
    friend class BlockDiagMatrix;
    template<int M, class TT>
      friend class ParallelCGTask;
    template<int M, class TT>
      friend class ParallelCGCudaTask;
    template<int M, class TT>
      friend class ParallelSPMVCudaTask;
    template<int M, class TT>
      friend class CSpMatrix;

    template< class TT>
    friend class CVector;

    Vector(ulong size=0);
    Vector(const Vector& v);
    virtual ~Vector();
    
    /*Single threaded sum*/
    T sum()const;
    
    /*Multithreaded sum*/
    T sum(T* sharedBuffer, const Thread* caller, 
	  const VectorRange* rg)const;

    /*Partial sum (used in parallel version)*/
    T sump(const VectorRange r)const;

    T& operator[](ulong i){
      cgfassert(i < size);
      return data[i];
    }
    const T& operator[](ulong i)const{
      cgfassert(i < size);
      return data[i];
    }

    void clear(){
      memset(data, 0, sizeof(T)*size);
    }

    void enableReduction(){
      if(tmp_buffer == 0){
	tmp_buffer = new T[size];
      }
    }

    Vector<T>& operator=(const Vector<T>& v);

    Vector<T>& operator=(const T f){
      for(uint i=0;i<origSize;i++){
	data[i] = f;
      }
      return *this;
    }

    Vector<T>& set(const T[], ulong size);

    template<class U>
    Vector<T>& operator*=(U n){
      Vector<T>::mulf(*this, *this, (T)n);
      return *this;
    }

    template<class U>
    Vector<T>& operator/=(U n){
      Vector<T>::mulf(*this, *this, 1.0/(T)n);
      return *this;
    }

    template<class U>
      Vector<T>& operator+=(U n){
      Vector<T>::adds(*this, *this, (T)n);
      return *this;
    }

    template<class U>
    Vector<T>& operator-=(U n){
      Vector<T>::subs(*this, *this, (T)n);
      return *this;
    }

    //Vector<T>& operator*(const Vector<T>& v)const;

    //Vector<T>& operator/(const Vector<T>& v)const;

    Vector<T>& operator+=(const Vector<T>& v);

    Vector<T>& operator-=(const Vector<T>& v);

    Vector<T>& operator*=(const Vector<T>& v);

    Vector<T>& operator/=(const Vector<T>& v);

    /*Conversion*/
    operator T*(){return (T*)data;}
    operator const T*()const {return (T*)data;}
    
    /*Unary*/
    Vector<T> operator+() const{
      Vector<T> r(*this);
      return r;
    }

    Vector<T> operator-() const{
      Vector<T> r(*this);
      r*=-1.0f;
      return r;
    }

    /*Vector vector*/
    Vector<T> operator+(const Vector<T>& v) const{
      cgfassert(size == v.size);
      Vector<T> r(*this);
      r+=v;
      return r;
    }

    Vector<T> operator-(const Vector<T>& v) const{
      cgfassert(size == v.size);
      Vector<T> r(*this);
      r-=v;
      return r;
    }

    /*Vector scalar*/
    template<class U>
    Vector<T> operator+(U f)const{
      Vector<T> r(*this);
      r+=(T)f;
      return r;
    }

    template<class U>
    Vector<T> operator-(U f)const{
      Vector<T> r(*this);
      r-=(T)f;
      return r;
    }
    
    template<class U, class V>
      friend Vector<U> operator*(const Vector<U>& a, V n);
    template<class U, class V>
      friend Vector<U> operator*(V n, const Vector<U>& a);
    template<class U, class V>
      friend Vector<U> operator/(const Vector<U>& a, V n);
    template<class U, class V>
      friend Vector<U> operator/(V n, const Vector<U>& a);
    
    /*Dot product*/
    T operator*(const Vector<T>& v) const;

    /*Cross product*/
    Vector<T> operator^(const Vector<T>& v) const{
      Vector<T> r(*this);
      error("Not implemented yet");
      return r;
    }

    /*Vector matrix multiplication*/
    template<int M, class U>
      friend Vector<U> operator*(const SpMatrix<M, U>& m, const Vector<U>& v);
    template<int M, class U>
      friend void spmv(Vector<U>& r, const SpMatrix<M, U>& m, const Vector<U>& v);
    template<int M, class U>
      friend void spmv_t(Vector<U>& r, const SpMatrix<M, U>& m, const Vector<U>& v);

    template<class U>
      friend void spmv(Vector<U>& r, const BlockDiagMatrix& m, const Vector<U>& v);
    template<int M, class U>
      friend void spmv_partial(Vector<U>& r, const SpMatrix<M, U>& m, 
			       const Vector<U>& v, const MatrixRange mr);

    /*Zero test*/
    bool operator!() const{
      error("Not implemented yet");
      return false;
    }

    /*Equality tests*/
    bool operator==(const Vector<T>& v) const{
      error("Not implemented yet");
      return false;
    }
    
    bool operator!=(const Vector<T>& v) const{
      error("Not implemented yet");
      return false;
    }
    
    template<class U>
      friend inline bool operator==(const Vector<U>& a, U n);
    template<class U>
      friend inline bool operator!=(const Vector<U>& a, U n);
    template<class U>
      friend inline bool operator==(U n, const Vector<U>& a);
    template<class U>
      friend inline bool operator!=(U n, const Vector<U>& a);

    /*Inequality tests*/
    bool operator<(const Vector<T>& v)const{
      error("Not implemented yet");
      return false;
    }

    bool operator<=(const Vector<T>& v)const{
      error("Not implemented yet");
      return false;
    }

    bool operator>(const Vector<T>& v)const{
      error("Not implemented yet");
      return false;
    }
    
    bool operator>=(const Vector<T>& v)const{
      error("Not implemented yet");
      return false;
    }
    
    template<class U>
      friend bool operator<(const Vector<U>& a, U n);
    template<class U>
      friend bool operator<=(const Vector<U>& a, U n);
    template<class U>
      friend bool operator>(const Vector<U>& a, U n);
    template<class U>
      friend bool operator>=(const Vector<U>& a, U n);
    
    template<class U>
      friend bool operator<(U n, const Vector<U>& a);
    template<class U>
      friend bool operator<=(U n, const Vector<U>& a);
    template<class U>
      friend bool operator>(U n, const Vector<U>& a);
    template<class U>
      friend bool operator>=(U n, const Vector<U>& a);
    
    /*Vector length*/
    T length2() const;
    
    T length()const{
      return sqrt(length2());
    }
    
    /*Clamp*/
    Vector<T>& clamp(T lo, T hi){
      error("Not implemented yet");
      return *this;
    }
    
    /*Lo hi*/
    template<class U>
      friend Vector<U> lo(const Vector<U>& a, const Vector<U>& b);
    template<class U>
      friend Vector<U> hi(const Vector<U>& a, const Vector<U>& b);
    
    /*Normalize vector*/
    template<class U>
      friend Vector<U> normalize(const Vector<U>& v);
    
    template<class U>
      friend std::ostream& operator<<(std::ostream& os, 
				      const Vector<U>& v);

    ulong getSize()const{return origSize;}
    ulong getPaddedSize()const{return size;}
    
    //  protected:
    /*Functions which should be used in iterative methods because
      these functions do not allocate memory themself. When executing
      iterative methods, a lot of vector calculations are
      performed. Since most standard operators will create new
      objects, with a large size of allocated memory, and return these
      temporary vectors as result. Eventually this result is assigned
      to another vector object. In normal situations this is generally
      not a problem, however, in iterative methods we suggest to reuse
      allocated memory and avoid the use of unneeded memory
      allocations and memory copies. */
    /*
    float sumReduction(float* tmpStorage)const;
    float sumReductionPartial(float* tmpStorage, 
			      const VectorRange r)const;
    */

   
    static void sub  (Vector<T>& r, const Vector<T>& a, const Vector<T>& b);
    static void add  (Vector<T>& r, const Vector<T>& a, const Vector<T>& b);
    static void subs (Vector<T>& r, const Vector<T>& a, T f);
    static void adds (Vector<T>& r, const Vector<T>& a, T f);

    static void madd (Vector<T>& r, const Vector<T>& a, const Vector<T>& b,
		      const Vector<T>& c);
    static void mfadd(Vector<T>& r, T f, const Vector<T>& b, 
		      const Vector<T>& c);
    static void mul  (Vector<T>& r, const Vector<T>& a, const Vector<T>& b);
    static void mulf (Vector<T>& r, const Vector<T>& a, T f);
    static void div  (Vector<T>& r, const Vector<T>& a, const Vector<T>& b);
    
    /*Partial operations*/
    static void subp  (Vector<T>& r, const Vector<T>& a, const Vector<T>& b,
		       const VectorRange rg);
    static void addp  (Vector<T>& r, const Vector<T>& a, const Vector<T>& b,
		       const VectorRange rg);
    static void subsp (Vector<T>& r, const Vector<T>& a, T f,
		       const VectorRange rg);
    static void addsp (Vector<T>& r, const Vector<T>& a, T f,
		       const VectorRange rg);
    static void maddp (Vector<T>& r, const Vector<T>& a, const Vector<T>& b, 
		       const Vector<T>& c, const VectorRange rg);
    static void mfaddp(Vector<T>& r, T f, const Vector<T>& b, 
		       const Vector<T>& c, const VectorRange rg);
    static void mulp  (Vector<T>& r, const Vector<T>& a, const Vector<T>& b, 
		       const VectorRange rg);
    static void mulfp (Vector<T>& r, const Vector<T>& a, T f, 
		       const VectorRange rg);
  protected:
    ulong size;
    ulong origSize;
    T* data;               /*Data points to memory containing a
			     multiple of 16 elements*/
    mutable T* tmp_buffer; /*Used for reductions*/
  };
  
  template<class U>
  extern Vector<U> normalize(const Vector<U>& a);
  
  template<class U>
  extern std::ostream& operator<<(std::ostream& os, const Vector<U>& v);

  template<class T, class U>
  Vector<T> operator*(const Vector<T>& a, U n){
    Vector<T> r = a;
    r*=(T)n;
    return r;
  }

  template<class T, class U>
  Vector<T> operator*(U n, const Vector<T>& a){
    Vector<T> r = a;
    r*=(T)n;
    return r;
  }
  
  template<class T, class U>
  Vector<T> operator/(const Vector<T>& a, U n){
    Vector<T> r = a;
    r/=(T)n;
    return r;
  }

  template<class T, class U>
  Vector<T> operator/(U n, const Vector<T>& a){
    Vector<T> r;
    error("not implemented yet");
    return r;
  }
}

#endif/*VECTOR_HPP*/

