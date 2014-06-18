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

#ifndef LINSOLVECG_HPP
#define LINSOLVECG_HPP

#include "math/LinSolve.hpp"
#include "core/Exception.hpp"

namespace CGF{
  template<int N, class T>
  class LinSolveCG : public LinSolve<N, T>{
  public:
    LinSolveCG(int d):LinSolve<N, T>(d){
      r = new Vector<T>(this->dim);
      C = new Vector<T>(this->dim);
      scratch1 = new Vector<T>(this->dim);
      scratch2 = new Vector<T>(this->dim);

      w = new Vector<T>(this->dim);
      v = new Vector<T>(this->dim);
      u = new Vector<T>(this->dim);

      Vector<T>::mulf(*r, *r, 0); //r = 0
    }

    virtual ~LinSolveCG(){
      delete r;
      delete C;
      delete scratch1;
      delete scratch2;
      delete w;
      delete v;
      delete u;
    }

    virtual void preSolve(){
      this->mat->finalize();
      Vector<T>::mul(*scratch1, *this->b, *this->b);
      bnorm = Sqrt(scratch1->sum());
    }

    virtual void solve(int steps = 100000, T tolerance = (T)1e-6){
      cgfassert(this->mat->getWidth() == this->b->getSize());
      cgfassert(this->mat->getWidth() == this->mat->getHeight());
      cgfassert(this->mat->getWidth() == this->x->getSize());

      this->iterations = 0;

      for(int i=0;i<this->mat->getHeight();i++){
        (*C)[i] = (T)1.0/Sqrt(Fabs((*this->mat)[i][i]));
      }

      /*r = Ax*/
      spmv(*r, *(this->mat), *(this->x));

      /*r = b - r*/;
      Vector<T>::sub(*r, *(this->b), *r);

      /*w = C * r*/
      Vector<T>::mul(*w, *C, *r);

      /*v = C * w*/
      Vector<T>::mul(*v, *C, *w);

      /*s1 = w * w*/
      Vector<T>::mul(*scratch1, *w, *w);
      T alpha;

      /*alpha = sum(s1)*/
      alpha = scratch1->sum();

      int k=0;

      while(k<steps){
        /*s1 = v * v*/
        this->iterations++;
        Vector<T>::mul(*scratch1, *v, *v);
        T residual;

        residual = scratch1->sum();
        if(Sqrt(Fabs(residual)) < (tolerance*bnorm /*+ tolerance*/)){
          warning("CG::Success in %d iterations, %10.10e, %10.10e", k, residual, Sqrt(residual));
          return;
        }

        /*u = A*v*/
        spmv(*u, *(this->mat), *v);

        /*s1 = v * u*/
        Vector<T>::mul(*scratch1, *v, *u);
        T divider;

        /*divider = sum(s1)*/
        divider = scratch1->sum();

        T t = alpha/divider;

        /*x = x + t*v*/
        /*r = r - t*u*/
        /*w = C * r*/
        Vector<T>::mfadd(*(this->x),  t, *v, *(this->x));
        Vector<T>::mfadd(*r, -t, *u, *r);
        Vector<T>::mul  (*w, *C, *r);

        /*s1 = w*w*/
        Vector<T>::mul(*scratch1, *w, *w);

        /*beta = sum(s1)*/
        T beta = scratch1->sum();
#if 1
        if(beta < (tolerance*bnorm + tolerance)){
          T rl = r->length2();
          if(Sqrt(rl)<(tolerance*bnorm + tolerance)){
            warning("CG::Success in %d iterations, %10.10e, %10.10e", k, rl,
                    Sqrt(Fabs(rl)));
            return;
          }
        }
#endif

        T s = beta/alpha;

        /*s1 = C * w*/
        Vector<T>::mul(*scratch1, *C, *w);
        /*v = s1 + s * v*/
        Vector<T>::mfadd(*v, s, *v, *scratch1);
        alpha = beta;
        k++;
      }
      message("Unsuccesfull");
      throw new SolutionNotFoundException(__LINE__, __FILE__,
                                          "Number of iterations exceeded.");

    }
  protected:
    Vector<T>* r;
    Vector<T>* C;
    Vector<T>* scratch1;
    Vector<T>* scratch2;
    Vector<T>* w;
    Vector<T>* v;
    Vector<T>* u;
    T bnorm;
  };
}

#endif/*LINSOLVECG*/
