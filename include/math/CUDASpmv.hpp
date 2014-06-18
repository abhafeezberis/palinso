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

#ifndef CUDASPMV_HPP
#define CUDASPMV_HPP

#ifdef CUDA

namespace CGF{
  enum TextureOperation{TexNone = 0, TexVector, TexVectorAndIndices};

  /*N = block_size, T = float|doublem N_THR = 128, 256, 512, TEX=0, 1, 2*/
  template<int N, class T, int N_THR>
  class CUDASPMV{
  public:
    static void spmv_ordered(T* d_blocks, 
                             int* d_col_indices, 
                             int* d_row_lengths, int* d_row_indices,
                             int* d_row_map,
                             const T* d_b, T* d_x, int dim, 
                             int n_blocks, TextureOperation tex_op);
  };
}
#endif/*CUDA*/
#endif/*CUDASPMV_HPP*/
