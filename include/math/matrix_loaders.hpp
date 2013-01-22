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

#ifndef MATRIX_LOADERS_HPP
#define MATRIX_LOADERS_HPP

#include "core/cgfdefs.hpp"

namespace CGF{
  template<int N, class T>
  class SpMatrix;

  template<class T>
  class Vector;
  
  template<int N, class T>
  SpMatrix<N, T>* load_matrix_market_exchange(const char* filename);

  template<int N, class T>
  void save_matrix_market_exchange(const char* filename, 
				   const SpMatrix<N, T>*const mat);

  template<int N, class T>
  Vector<T>* load_matrix_market_exchange_vector(const char* filename);

  template<int N, class T>
  bool save_matrix_matlab(const char* filename, const SpMatrix<N, T>* const mat);
  
  template<int N, class T>
  SpMatrix<N, T>* load_matrix_matlab(const char* filename);
  
};

#endif/*MATRIX_LOADERS_HPP*/
