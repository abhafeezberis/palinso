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

#include "util/cuda_util.hpp"
#include "util/device_util.hpp"
#include "math/CUDASpmv.hpp"
#include "core/Thread.hpp"


namespace CGF{
  static texture<float, 1, cudaReadModeElementType> texf2;
  static texture<uint,  1, cudaReadModeElementType> texui2;
  static texture<int,   1, cudaReadModeElementType> texi2;
  static texture<int2,  1, cudaReadModeElementType> texd2;

  template<class T>
  void bindTexture2(T* d_array, int size);

  template<>
  void bindTexture2<float>(float* d_array, int size){
    //cudaChannelFormatDesc channelDescr = 
    //cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
    
    size_t offset = size_t(-1);
    cudaSafeCall(cudaBindTexture(&offset, texf2, d_array));
  }

  template<>
  void bindTexture2<double>(double* d_array, int size){
    //cudaChannelFormatDesc channelDescr = 
    //cudaCreateChannelDesc(64,0,0,0,cudaChannelFormatKindFloat);
    
    size_t offset = size_t(-1);
    cudaSafeCall(cudaBindTexture(&offset, texd2, d_array));
  }

  void bindUIntTexture2(uint* d_array, int size){
    cudaChannelFormatDesc channelDescr = 
      cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindUnsigned);

    cudaSafeCall(cudaBindTexture(0, &texui2, d_array, &channelDescr, 
                                 size*sizeof(uint)));
  }

  void bindIntTexture2(int* d_array, int size){
    cudaChannelFormatDesc channelDescr = 
      cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindUnsigned);

    cudaSafeCall(cudaBindTexture(0, &texi2, d_array, &channelDescr, 
                                 size*sizeof(int)));
  }


  void unbindTexture2(){
    cudaSafeCall(cudaUnbindTexture(&texf2));
    cudaSafeCall(cudaUnbindTexture(&texui2));
    cudaSafeCall(cudaUnbindTexture(&texi2));
    cudaSafeCall(cudaUnbindTexture(&texd2));
  }

  template<class T, int TEX> class TEXTURE{
  public:
    static __inline__ __device__ T fetch_vec2(const int& index, 
                                              const int& offset, 
                                              const T* array,
                                              const int* column_indices);
  };
  
  template<int TEX> class TEXTURE<double, TEX>{
  public:
    static __inline__ __device__ double fetch_vec2(const int& index,
                                                   const int& offset,
                                                   const double* array,
                                                   const int* column_indices){
      if(TEX == 0){
        return array[column_indices[index]+offset];
      }else if(TEX == 1){
#if __CUDA_ARCH__ >= 130
        int2 dd = tex1Dfetch(texd2, column_indices[index]+offset);
        return __hiloint2double(dd.y, dd.x);
#else
        return array[column_indices[index]+offset];
#endif
      }else if(TEX == 3){
#if __CUDA_ARCH__ >= 130
        int2 dd = tex1Dfetch(texd2, tex1Dfetch(texui2, index)+offset);
        return __hiloint2double(dd.y, dd.x);
#else
        return array[tex1Dfetch(texui2, index)+offset];
#endif
      }else{
        return 0;
      }
    }
  };
  
  template<int TEX> class TEXTURE<float, TEX>{
  public:
    static __inline__ __device__ float fetch_vec2(const int& index,
                                                  const int& offset,
                                                  const float* array,
                                                  const int* column_indices){
      if(TEX==0){
        return array[column_indices[index]+offset];
      }else if(TEX == 1){
        return tex1Dfetch(texf2, column_indices[index]+offset);
      }else if(TEX == 2){
        return tex1Dfetch(texf2, tex1Dfetch(texui2, index)+offset);
      }else{
        return 0;
      }
    }
  };

  /*To allow more than one block in the x direction, define EXT*/
  //#define EXT

#define NX 1
  /*Compute maximum NX for configuration N*/
  template<int N, int N_THR>
  __inline__ __device__ int NNX2(){
    if(NX > (N_THR/(N*N))){
      return N_THR/(N*N);
    }else{
      return NX;
    }
  }
  
  /*N=4 && N=8 uses texture for vector indices. For N=1 && N=2 normal
    array lookup can be done, which is faster in these cases*/
  template<int N, class T, int N_THR, int TEX>
  __global__ void cuda_spmv_ordered2(T* d_blocks, 
                                     int* d_col_indices,
                                     int* d_row_lengths, 
                                     int* d_row_indices,
                                     int* d_row_map,
                                     const T* d_b,
                                     T* d_res,
                                     int dim, int n_blocks){
    __shared__ T     blocks[N_THR];

    int  row_lengths;
    int  row_index;

    int thread_id = TY * N + TX;
    int block_index   = (TY%N) * N + TX;    
    int cuda_block_id = (BY * gridDim.x + BX);

    if(cuda_block_id >= n_blocks)
      return;

#ifdef EXT
    int nx = NNX2<N, N_THR>();
#endif

#ifdef EXT
    int  orig_row      = d_row_map[cuda_block_id * (N_THR/(N*N))/nx + (TY/N)/nx];
#else
    int  orig_row      = d_row_map[cuda_block_id * (N_THR/(N*N)) + (TY/N)];
#endif

#ifdef EXT
    row_lengths = d_row_lengths[cuda_block_id]/nx;
#else
    row_lengths = d_row_lengths[cuda_block_id];
#endif
    row_index   = d_row_indices[cuda_block_id];

    blocks[thread_id] = 0;
    
    for(int i=0;i<row_lengths;i++){
      int block_id = row_index + (N_THR/(N*N)) * i + TY/N;
      /*GTX280 N==1 -> Vector texture
        N==2 -> Vector texture
        N==4 -> Double texture
        N==8 -> Double texture
	       
        GTX570 N==1 -> Vector texture
        N==2 -> Vector texture
        N==4 -> Vector texture
        N==8 -> Vector texture
      */
      blocks[thread_id] += 
        d_blocks[block_id*N*N + block_index]* 
        TEXTURE<T, TEX>::fetch_vec2(block_id, TX, d_b, d_col_indices);
    }

    //__syncthreads();


#ifdef EXT
    /*If NX == 1 keep, if NX == 2 add all odd to all even blocks,
      reduce all even blocks, if NX == 4, add all odd to even, add*/
    int row = TY/N;
    if(nx == 1){
      /*Ok*/
    }
#endif

#if 0
    else{
      volatile T* vb = blocks;
      
      int offset = N*N;
      int nn = 1;
      while(nn != nx){
        if(thread_id + offset < N_THR){
          vb[thread_id] += vb[thread_id + offset];
        }

        if(offset > 32){
          __syncthreads();
        }
	
        offset *= 2;
        nn *= 2;
      }
    }
#else
    
    
#endif
#ifdef EXT
    if(nx == 2){
      /*Add block 1 to 0, 3 to 2*/
      if(row%nx == 0){
        blocks[thread_id] += blocks[thread_id + N*N];
      }
    }else if(nx == 4){
      if(row%nx == 0){
        blocks[thread_id] += 
          blocks[thread_id + N*N*1] +
          blocks[thread_id + N*N*2] +
          blocks[thread_id + N*N*3];
      }
    }else if(nx == 8){
      if(row%nx == 0){
        blocks[thread_id] += 
          blocks[thread_id + N*N*1] +
          blocks[thread_id + N*N*2] +
          blocks[thread_id + N*N*3] +
          blocks[thread_id + N*N*4] +
          blocks[thread_id + N*N*5] +
          blocks[thread_id + N*N*6] +
          blocks[thread_id + N*N*7];
      }
    }else if(nx == 16){
      if(row%nx == 0){
        blocks[thread_id] += 
          blocks[thread_id + N*N*1] +
          blocks[thread_id + N*N*2] +
          blocks[thread_id + N*N*3] +
          blocks[thread_id + N*N*4] +
          blocks[thread_id + N*N*5] +
          blocks[thread_id + N*N*6] +
          blocks[thread_id + N*N*7] +
          blocks[thread_id + N*N*8] +
          blocks[thread_id + N*N*9] +
          blocks[thread_id + N*N*10] +
          blocks[thread_id + N*N*11] +
          blocks[thread_id + N*N*12] +
          blocks[thread_id + N*N*13] +
          blocks[thread_id + N*N*14] +
          blocks[thread_id + N*N*15];
      }
    }else if(nx == 32){
      if(row%nx == 0){
        blocks[thread_id] += 
          blocks[thread_id + N*N*1] +
          blocks[thread_id + N*N*2] +
          blocks[thread_id + N*N*3] +
          blocks[thread_id + N*N*4] +
          blocks[thread_id + N*N*5] +
          blocks[thread_id + N*N*6] +
          blocks[thread_id + N*N*7] +
          blocks[thread_id + N*N*8] +
          blocks[thread_id + N*N*9] +
          blocks[thread_id + N*N*10] +
          blocks[thread_id + N*N*11] +
          blocks[thread_id + N*N*12] +
          blocks[thread_id + N*N*13] +
          blocks[thread_id + N*N*14] +
          blocks[thread_id + N*N*15] +
          blocks[thread_id + N*N*16] +
          blocks[thread_id + N*N*17] +
          blocks[thread_id + N*N*18] +
          blocks[thread_id + N*N*19] +
          blocks[thread_id + N*N*20] +
          blocks[thread_id + N*N*21] +
          blocks[thread_id + N*N*22] +
          blocks[thread_id + N*N*23] +
          blocks[thread_id + N*N*24] +
          blocks[thread_id + N*N*25] +
          blocks[thread_id + N*N*26] +
          blocks[thread_id + N*N*27] +
          blocks[thread_id + N*N*28] +
          blocks[thread_id + N*N*29] +
          blocks[thread_id + N*N*30] +
          blocks[thread_id + N*N*31];
      }
    }
#endif
#if 1

    __syncthreads();
#ifdef EXT
    if(row %nx == 0){
#endif
      vector_sum<N, T>(blocks, thread_id);
      
      if((orig_row != -1) && (TX == 0)){
      	orig_row = orig_row * N + TY % N;
      	REDUCE_1(d_res, orig_row, blocks, thread_id);
      }
#ifdef EXT
    }
#endif
    //__syncthreads();
    //if(thread_id < (SIMB/NX)*N){
    //orig_row = orig_row * N + thread_id;
    //d_res[orig_row] = blocks[TY/N];
    //}
#endif
  }

  template<int N, class T, int N_THR>
  void CUDASPMV<N, T, N_THR>::spmv_ordered(T* d_blocks, 
                                           int* d_col_indices, 
                                           int* d_row_lengths, 
                                           int* d_row_indices,
                                           int* d_row_map,
                                           const T* d_b, T* d_x, int dim, 
                                           int n_blocks, 
                                           TextureOperation tex_op){
    int n_cuda_blocks;
    int nx = (NX>(N_THR/(N*N)))?(N_THR/(N*N)):NX;
    dim *= N;
    if((dim/N)%((N_THR/(N*N))/nx) != 0){
      n_cuda_blocks = 1+(dim/N)/((N_THR/(N*N))/nx);
    }else{
      /*If NTHREADS=32, N=2, NX=2, dim = 16 -> dim/N = 8, SIMB=8,
        SIMB/NX=4, n_cuda_blocks = 2*/
      n_cuda_blocks = (dim/N)/((N_THR/(N*N))/nx);
    }

    dim3 threads(N, N*(N_THR/(N*N)));
    dim3 grid(n_cuda_blocks);
    if(n_cuda_blocks > 65535){
      grid.x = (int)ceil(sqrt((float)n_cuda_blocks));
      grid.y = (int)ceil(sqrt((float)n_cuda_blocks));
    }

    if(tex_op == TexNone){
      cuda_spmv_ordered2<N, T, N_THR, 0><<<grid, threads>>>(d_blocks, 
                                                            d_col_indices, 
                                                            d_row_lengths, 
                                                            d_row_indices, 
                                                            d_row_map,
                                                            d_b, 
                                                            d_x, 
                                                            dim, 
                                                            n_cuda_blocks);
    }else if(tex_op == TexVector){
      cuda_spmv_ordered2<N, T, N_THR, 1><<<grid, threads>>>(d_blocks, 
                                                            d_col_indices, 
                                                            d_row_lengths, 
                                                            d_row_indices, 
                                                            d_row_map,
                                                            d_b, 
                                                            d_x, 
                                                            dim, 
                                                            n_cuda_blocks);
    }else if(tex_op == TexVectorAndIndices){
      cuda_spmv_ordered2<N, T, N_THR, 2><<<grid, threads>>>(d_blocks, 
                                                            d_col_indices, 
                                                            d_row_lengths, 
                                                            d_row_indices, 
                                                            d_row_map,
                                                            d_b, 
                                                            d_x, 
                                                            dim, 
                                                            n_cuda_blocks);
    }
    
    cudaCheckError("Kernel launch cuda_spmv_ordered");
  }

  template class CUDASPMV<1, float, 128>;
  template class CUDASPMV<2, float, 128>;
  template class CUDASPMV<4, float, 128>;
  template class CUDASPMV<8, float, 128>;

  template class CUDASPMV<1, double, 128>;
  template class CUDASPMV<2, double, 128>;
  template class CUDASPMV<4, double, 128>;
  template class CUDASPMV<8, double, 128>;

  template class CUDASPMV<1, float, 256>;
  template class CUDASPMV<2, float, 256>;
  template class CUDASPMV<4, float, 256>;
  template class CUDASPMV<8, float, 256>;

  template class CUDASPMV<1, double, 256>;
  template class CUDASPMV<2, double, 256>;
  template class CUDASPMV<4, double, 256>;
  template class CUDASPMV<8, double, 256>;

  template class CUDASPMV<1, float, 512>;
  template class CUDASPMV<2, float, 512>;
  template class CUDASPMV<4, float, 512>;
  template class CUDASPMV<8, float, 512>;

  template class CUDASPMV<1, double, 512>;
  template class CUDASPMV<2, double, 512>;
  template class CUDASPMV<4, double, 512>;
  template class CUDASPMV<8, double, 512>;
}
