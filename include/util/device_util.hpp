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

#ifndef DEVICE_UTIL_HPP
#define DEVICE_UTIL_HPP

/*Some standard macros which are used in most kernels*/
#define TX    threadIdx.x
#define TY    threadIdx.y
#define BX    blockIdx.x
#define BY    blockIdx.y
#define THRDS blockDim.x * blockDim.y
#define BIDX (BY * gridDim.x  + BX)
#define TIDX (TY * blockDim.x + TX)

/*Macros for reducing shared memory vectors within one warp*/
#define REDUCE_1(vec_out, index_out, vec_in, index_in)	\
  (vec_out)[(index_out)] = (vec_in)[(index_in)];	

#ifdef __DEVICE_EMULATION__

#error

#define REDUCE_2(vec, index, op, tx, TP)                    \
  if((tx) < 1){                                             \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+1];	\
  }                                                         \
  __syncthreads();

  
#define REDUCE_4(vec, index, op, tx, TP)                    \
  if((tx) < 2){                                             \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+2];	\
  }                                                         \
  __syncthreads();                                          \
  if((tx) < 2){                                             \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+1];	\
  }                                                         \
  __syncthreads();

#define REDUCE_8(vec, index, op, tx, TP)                    \
  if((tx) < 4){                                             \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+4];	\
  }                                                         \
  __syncthreads();                                          \
  if((tx) < 4){                                             \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+2];	\
  }                                                         \
  __syncthreads();                                          \
  if((tx) < 4){                                             \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+1];	\
  }                                                         \
  __syncthreads();

#define REDUCE_16(vec, index, op, tx, TP)                   \
  if((tx) < 8){                                             \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+8];	\
  }                                                         \
  __syncthreads();                                          \
  if((tx) < 8){                                             \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+4];	\
  }                                                         \
  __syncthreads();                                          \
  if((tx) < 8){                                             \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+2];	\
  }                                                         \
  __syncthreads();                                          \
  if((tx) < 8){                                             \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+1];	\
  }                                                         \
  __syncthreads();

#define REDUCE_32(vec, index, op, tx, TP)                   \
  if((tx) < 16){                                            \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+16];   \
  }                                                         \
  __syncthreads();                                          \
  if((tx) < 16){                                            \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+8];    \
  }                                                         \
  __syncthreads();                                          \
  if((tx) < 16){                                            \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+4];    \
  }                                                         \
  __syncthreads();                                          \
  if((tx) < 16){                                            \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+2];    \
  }                                                         \
  __syncthreads();                                          \
  if((tx) < 16){                                            \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+1];    \
  }__syncthreads();

#define REDUCE_64(vec, index, op, tx, TP)                   \
  if((tx) < 32){                                            \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+32];   \
  }                                                         \
  __syncthreads();                                          \
  if((tx) < 32){                                            \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+16];   \
  }                                                         \
  __syncthreads();                                          \
  if((tx) < 32){                                            \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+8];    \
  }                                                         \
  __syncthreads();                                          \
  if((tx) < 32){                                            \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+4];    \
  }                                                         \
  __syncthreads();                                          \
  if((tx) < 32){                                            \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+2];    \
  }                                                         \
  __syncthreads();                                          \
  if((tx) < 32){                                            \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+1];    \
  }__syncthreads();

#else

#define REDUCE_2(vec, index, op, tx, TP)                        \
  if((tx) < 1){                                                 \
    volatile TP* s_vec = vec;                                   \
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+1];	\
  }

#define REDUCE_4(vec, index, op, tx, TP)                        \
  if((tx) < 2){                                                 \
    volatile TP* s_vec = vec;                                   \
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+2];	\
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+1];	\
  }

#define REDUCE_8(vec, index, op, tx, TP)                        \
  if((tx) < 4){                                                 \
    volatile TP* s_vec = vec;                                   \
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+4];	\
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+2];	\
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+1];	\
  }

#define REDUCE_16(vec, index, op, tx, TP)                       \
  if((tx) < 8){                                                 \
    volatile TP* s_vec = vec;                                   \
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+8];	\
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+4];	\
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+2];	\
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+1];	\
  }

#define REDUCE_32(vec, index, op, tx, TP)                       \
  if((tx) < 16){                                                \
    volatile TP* s_vec = vec;                                   \
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+16];	\
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+8];	\
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+4];	\
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+2];	\
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+1];	\
  }

#define REDUCE_64(vec, index, op, tx, TP)                       \
  if((tx) < 32){                                                \
    volatile TP* s_vec = vec;                                   \
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+32];	\
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+16];	\
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+8];	\
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+4];	\
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+2];	\
    (s_vec)[(index)] = (s_vec)[(index)] op (s_vec)[(index)+1];	\
  }
#endif

#define REDUCE_128(vec, index, op, tx, TP)                  \
  if((tx) < 64){                                            \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+64];	\
  }                                                         \
  __syncthreads();                                          \
  REDUCE_64(vec, index, op, tx, TP);

#define REDUCE_256(vec, index, op, tx, TP)                  \
  if((tx) < 128){                                           \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+128];	\
  }                                                         \
  __syncthreads();                                          \
  REDUCE_128(vec, index, op, tx, TP);

#define REDUCE_512(vec, index, op, tx, TP)                  \
  if((tx) < 256){                                           \
    (vec)[(index)] = (vec)[(index)] op (vec)[(index)+256];	\
  }                                                         \
  __syncthreads();                                          \
  REDUCE_256(vec, index, op, tx, TP);

//////
template<int N, char op, class T>
__device__ void vector_reduce(T* vec, uint index);
//////

template<int N, class T>
__device__ void vector_sum(T* vec, uint index){
  vector_reduce<N, '+', T>(vec, index);
}

template<>
inline __device__ void vector_reduce<1, '+', float>(float* vec, uint index){
}

template<>
inline __device__ void vector_reduce<2, '+', float>(float* vec, uint index){
  REDUCE_2(vec, index, +, TX, float);
}

template<>
inline __device__ void vector_reduce<4, '+', float>(float* vec, uint index){
  REDUCE_4(vec, index, +, TX, float);
}

template<>
inline __device__ void vector_reduce<8, '+', float>(float* vec, uint index){
  REDUCE_8(vec, index, +, TX, float);
}

template<>
inline __device__ void vector_reduce<16, '+', float>(float* vec, uint index){
  REDUCE_16(vec, index, +, TX, float);
}

template<>
inline __device__ void vector_reduce<32, '+', float>(float* vec, uint index){
  REDUCE_32(vec, index, +, TX, float);
}

template<>
inline __device__ void vector_reduce<64, '+', float>(float* vec, uint index){
  REDUCE_64(vec, index, +, TX, float);
}

template<>
inline __device__ void vector_reduce<128, '+', float>(float* vec, uint index){
  REDUCE_128(vec, index, +, TX, float);
}

template<>
inline __device__ void vector_reduce<256, '+', float>(float* vec, uint index){
  REDUCE_256(vec, index, +, TX, float);
}

template<>
inline __device__ void vector_reduce<512, '+', float>(float* vec, uint index){
  REDUCE_512(vec, index, +, TX, float);
}

/////

template<>
inline __device__ void vector_reduce<1, '+', double>(double* vec, uint index){
}

template<>
inline __device__ void vector_reduce<2, '+', double>(double* vec, uint index){
  REDUCE_2(vec, index, +, TX, double);
}

template<>
inline __device__ void vector_reduce<4, '+', double>(double* vec, uint index){
  REDUCE_4(vec, index, +, TX, double);
}

template<>
inline __device__ void vector_reduce<8, '+', double>(double* vec, uint index){
  REDUCE_8(vec, index, +, TX, double);
}

template<>
inline __device__ void vector_reduce<16, '+', double>(double* vec, uint index){
  REDUCE_16(vec, index, +, TX, double);
}

template<>
inline __device__ void vector_reduce<32, '+', double>(double* vec, uint index){
  REDUCE_32(vec, index, +, TX, double);
}

template<>
inline __device__ void vector_reduce<64, '+', double>(double* vec, uint index){
  REDUCE_64(vec, index, +, TX, double);
}

template<>
inline __device__ void vector_reduce<128, '+', double>(double* vec, uint index){
  REDUCE_128(vec, index, +, TX, double);
}

template<>
inline __device__ void vector_reduce<256, '+', double>(double* vec, uint index){
  REDUCE_256(vec, index, +, TX, double);
}

template<>
inline __device__ void vector_reduce<512, '+', double>(double* vec, uint index){
  REDUCE_512(vec, index, +, TX, double);
}

///////////

/*Templates for loading N values from memory and sum them*/
template<int N, class T>
inline __device__ T sum_load_vector(const T* vec, uint index, uint offset){
  return vec[index + (N-1) * offset] + sum_load_vector<N-1, T>(vec, index, offset);
}

template<>
inline __device__ float sum_load_vector<1, float>(const float* vec, uint index, uint offset){
  return vec[index];
}

template<>
inline __device__ double sum_load_vector<1, double>(const double* vec, uint index, uint offset){
  return vec[index];
}

#endif/*DEVICE_UTIL_HPP*/
