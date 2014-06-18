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

#ifndef LINSOLVE_HPP
#define LINSOLVE_HPP

#include "math/SpMatrix.hpp"
#include "math/Vector.hpp"

namespace CGF{
  template<int N, class T>
  class LinSolve{
  public:
    LinSolve(int d){
      dim = d;
      mat = new SpMatrix<N, T>(dim, dim);

      b   = new Vector<T>(dim);
      x   = new Vector<T>(dim);

      /*Reset result vector*/
      Vector<T>::mulf(*x, *x, 0);

      externalAllocatedMat = false;
      externalAllocatedb   = false;
      externalAllocatedx   = false;

      iterations = 0;
    }

    virtual ~ LinSolve(){
      if(!externalAllocatedMat)
        delete mat;
      if(!externalAllocatedb)
        delete b;
      if(!externalAllocatedx)
        delete x;
    }

    /*Fuctions for obtaining pointers to A, x, and b*/
    SpMatrix<N, T>* getMatrix(){
      return mat;
    }

    Vector<T>* getb(){
      return b;
    }

    Vector<T>* getx(){
      return x;
    }

    int getDim()const{
      return dim;
    }

    int getIterations(){
      return iterations;
    }

    /*Functions for replacing A, x, b by versions allocated elsewhere.
      Not, if created elsewhere, they should be deleted elsewehere*/

    virtual void setb(Vector<T>* vec){
      cgfassert(vec->getSize() == b->getSize());
      if(!externalAllocatedb){
        delete b;
      }
      b = vec;
      externalAllocatedb = true;
    }

    virtual void setx(Vector<T>* vec){
      cgfassert(vec->getSize() == x->getSize());
      if(!externalAllocatedx){
        delete x;
      }
      x = vec;
      externalAllocatedx = true;
    }

    virtual void setMatrix(SpMatrix<N, T>* m){
      cgfassert(m->getWidth() == mat->getWidth());
      cgfassert(m->getHeight() == mat->getHeight());
      if(!externalAllocatedMat){
        delete mat;
      }
      mat = m;
      externalAllocatedMat = true;
    }

    virtual void preSolve() = 0;

    virtual void solve(int steps = 100000, T tolerance = (T)1E-6) = 0;

  protected:
    int dim;
    SpMatrix<N, T>* mat;
    Vector<T>* b;
    Vector<T>* x;

    bool externalAllocatedMat;
    bool externalAllocatedb;
    bool externalAllocatedx;
    int iterations;
  };
}

#endif/*LINSOLVE*/
