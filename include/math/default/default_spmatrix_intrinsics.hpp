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

#ifndef DEFAULT_SPMATRIX_INTRINSICS
#define DEFAULT_SPMATRIX_INTRINSICS

#include <string.h>

namespace CGF{
  namespace default_proc{
    template<int N, class T>
    inline void spmatrix_block_load(T r[N*N], const T a[N*N]){
      memcpy(r, a, sizeof(T) * N * N);
    }

    template<int N, class T>
    inline void spmatrix_block_muls(T r[N*N], const T a[N*N], T f){
      for(uint j=0;j<N*N;j++){
        r[j] = a[j]*f;
      }
    }

    template<int N, class T>
    inline void spmatrix_block_divs(T r[N*N], const T a[N*N], T f){
      for(uint j=0;j<N*N;j++){
        r[j] = a[j]/f;
      }
    }

    template<int N, class T>
    inline void spmatrix_block_adds(T r[N*N], const T a[N*N], T f){
      for(uint j=0;j<N*N;j++){
        r[j] = a[j]+f;
      }
    }

    template<int N, class T>
    inline void spmatrix_block_subs(T r[N*N], const T a[N*N], T f){
      for(uint j=0;j<N*N;j++){
        r[j] = a[j]-f;
      }
    }

    template<int N, class T>
    inline void spmatrix_block_cmpeq(T r[N*N], const T a[N*N], T f){
      for(uint j=0;j<N*N;j++){
        r[j] = (a[j] == f);
      }
    }


    template<int N, class T>
    inline void spmatrix_block_cmpneq(T r[N*N], const T a[N*N], T f){
      for(uint j=0;j<N*N;j++){
        r[j] = (a[j] != f);
      }
    }

    template<int N, class T>
    inline void spmatrix_block_cmplt(T r[N*N], const T a[N*N], T f){
      for(uint j=0;j<N*N;j++){
        r[j] = (a[j] < f);
      }
    }

    template<int N, class T>
    inline void spmatrix_block_cmple(T r[N*N], const T a[N*N], T f){
      for(uint j=0;j<N*N;j++){
        r[j] = (a[j] <= f);
      }
    }

    template<int N, class T>
    inline void spmatrix_block_cmpgt(T r[N*N], const T a[N*N], T f){
      for(uint j=0;j<N*N;j++){
        r[j] = (a[j] > f);
      }
    }

    template<int N, class T>
    inline void spmatrix_block_cmpge(T r[N*N], const T a[N*N], T f){
      for(uint j=0;j<N*N;j++){
        r[j] = (a[j] >= f);
      }
    }

    template<int N, class T>
    inline void spmatrix_block_mul(T r[N*N], const T a[N*N], const T b[N*N]){
      for(uint j=0;j<N*N;j++){
        r[j] = a[j] * b[j];
      }
    }

    template<int N, class T>
    inline void spmatrix_block_div(T r[N*N], const T a[N*N], const T b[N*N]){
      for(uint j=0;j<N*N;j++){
        r[j] = a[j] / b[j];
      }
    }

    template<int N, class T>
    inline void spmatrix_block_cmpeq(T r[N*N], const T a[N*N], const T b[N*N]){
      for(uint j=0;j<N*N;j++){
        r[j] = (a[j] == b[j]);
      }
    }

    template<int N, class T>
    inline void spmatrix_block_cmpneq(T r[N*N], const T a[N*N], const T b[N*N]){
      for(uint j=0;j<N*N;j++){
        r[j] = (a[j] != b[j]);
      }
    }

    template<int N, class T>
    inline void spmatrix_block_cmplt(T r[N*N], const T a[N*N], const T b[N*N]){
      for(uint j=0;j<N*N;j++){
        r[j] = (a[j] < b[j]);
      }
    }

    template<int N, class T>
    inline void spmatrix_block_cmple(T r[N*N], const T a[N*N], const T b[N*N]){
      for(uint j=0;j<N*N;j++){
        r[j] = (a[j] <= b[j]);
      }
    }

    template<int N, class T>
    inline void spmatrix_block_cmpgt(T r[N*N], const T a[N*N], const T b[N*N]){
      for(uint j=0;j<N*N;j++){
        r[j] = (a[j] > b[j]);
      }
    }

    template<int N, class T>
    inline void spmatrix_block_cmpge(T r[N*N], const T a[N*N], const T b[N*N]){
      for(uint j=0;j<N*N;j++){
        r[j] = (a[j] >= b[j]);
      }
    }

    template<int N, class T>
    inline void spmatrix_block_row_sum_reduce(T r[N*N], const T a[N*N]){
      for(uint i=0;i<N;i++){
        T sum = 0;
        for(uint j=0;j<N;j++){
          sum += a[i*N + j];
          r[i*N + j] = 0;
        }

        r[i*N] = sum;
      }
    }

    template<int N, class T>
    inline void spmatrix_block_column_sum_reduce(T r[N*N], const T a[N*N]){
      for(uint j=0;j<N;j++){
        T sum = 0;
        for(uint i=0;i<N;i++){
          sum += a[i*N + j];
          r[i*N + j] = 0;
        }
        r[N*(N-1)+j] = sum;
      }
    }

    template<int N, class T>
    inline void spmatrix_block_vector_mul(T r[N*N], const T a[N*N],
                                          const T v[N]){
      for(uint j=0;j<N;j++){
        for(uint i=0;i<N;i++){
          r[j*N + i] = a[j*N + i] * v[i];
        }
      }
    }

    template<int N, class T>
    inline void spmatrix_block_vector_mul_transpose(T r[N*N],
                                                    const T a[N*N],
                                                    const T v[N]){
      for(uint j=0;j<N;j++){
        for(uint i=0;i<N;i++){
          r[j*N + i] = a[j*N + i] * v[j];
        }
      }
    }

    template<int N, class T>
    inline void spmatrix_block_vmadd(T r[N*N], const T a[N*N],
                                     const T b[N*N],
                                     const T v[N]){
      for(uint j=0;j<N;j++){
        for(uint i=0;i<N;i++){
          r[j*N + i] = a[j*N + i] + b[j*N + i] * v[i];
        }
      }
    }

    template<int N, class T>
    inline void spmatrix_block_vmadd_p(T r[N*N], const T a[N*N],
                                       const T b[N*N],
                                       const T v[N]){
      for(uint j=0;j<N;j++){
        for(uint i=0;i<N;i++){
          T res = b[j*N + i] * v[i];
          res = res>0?res:0;
          r[j*N + i] = a[j*N + i] + res;
        }
      }
    }

    template<int N, class T>
    inline void spmatrix_block_vmadd_n(T r[N*N], const T a[N*N],
                                       const T b[N*N],
                                       const T v[N]){
      for(uint j=0;j<N;j++){
        for(uint i=0;i<N;i++){
          T res = b[j*N + i] * v[i];
          res = res<0?res:0;
          r[j*N + i] = a[j*N + i] + res;
        }
      }
    }

    template<int N, class T>
    inline void spmatrix_block_vmadd_transpose(T r[N*N], const T a[N*N],
                                               const T b[N*N],
                                               const T v[N]){
      for(uint j=0;j<N;j++){
        for(uint i=0;i<N;i++){
          r[j*N + i] = a[j*N + i] + b[j*N + i] * v[j];
        }
      }
    }

    template<int N, class T>
    inline void spmatrix_block_vmadd_jacobi(T r[N*N], const T a[N*N],
                                            const T b[N*N],
                                            const T v[N]){
      for(uint j=0;j<N;j++){
        for(uint i=0;i<N;i++){
          if(i == j){
            r[j*N + i] = a[j*N + i];
          }else{
            r[j*N + i] = a[j*N + i] + b[j*N + i] * v[i];
          }
        }
      }
    }

    template<int N, class T>
    inline void spmatrix_block_add(T r[N*N], const T a[N*N], const T b[N*N]){
      for(uint j=0;j<N*N;j++){
        r[j] = a[j] + b[j];
      }
    }

    template<int N, class T>
    inline void spmatrix_block_sub(T r[N*N], const T a[N*N], const T b[N*N]){
      for(uint j=0;j<N*N;j++){
        r[j] = a[j] - b[j];
      }
    }

    template<int N, class T>
    inline void spmatrix_block_madd(T r[N*N], const T a[N*N], const T b[N*N],
                                    const T c[N*N]){
      for(uint j=0;j<N*N;j++){
        r[j] = a[j] * b[j] + c[j];
      }
    }

    template<int N, class T>
    inline void spmatrix_block_msadd(T r[N*N], T f,
                                     const T a[N*N], const T b[N*N]){
      for(uint j=0;j<N*N;j++){
        r[j] = f * a[j] + b[j];
      }
    }

    template<int N, class T>
    inline void spmatrix_block_msadd2(T r[N*N], T f,
                                      const T a[N*N], const T b[N*N],
                                      const T d[N*N]){
      for(uint j=0;j<N*N;j++){
        r[j] = f * a[j] + b[j] + d[j];
      }
    }

    template<int N, class T>
    inline void spmatrix_block_clear(T r[N*N]){
      memset(r, 0, sizeof(T)*N*N);
    }

    template<int N, class T>
    inline void spmatrix_block_set(T r[N*N], T f){
      for(uint j=0;j<N*N;j++){
        r[j] = f;
      }
    }
  }
}

#endif/*DEFAULT_SPMATRIX_INTRINSICS*/
