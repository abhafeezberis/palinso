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

#ifdef CUDA
#include "math/SpMatrix.hpp"
#include "math/CSpMatrix.hpp"
#include "util/cuda_util.hpp"
#include <cuda_runtime.h>
#include "core/Exception.hpp"
#include "math/Vector.hpp"
#include <algorithm>
#include <linux/unistd.h>
#include "math/CUDACGOp.hpp"
#include "core/ThreadPool.hpp"
#include "math/CUDASpmv.hpp"

#define NTHREADS 256
#undef NTHREADS

#define PINNED
//#undef  PINNED


namespace CGF{

  typedef struct _index_map{
    uint row;
    uint length;
  }index_map;

  bool index_map_sort_func(index_map a, index_map b){
    if(a.length > b.length){
      return true;
    }else{
      if(a.length == b.length){
	if(a.row < b.row){
	  return true;
	}
      }
    }
    return false;
  }

  template<int N, class T>
  CSpMatrix<N, T>::CSpMatrix(const SpMatrix<N, T>*matrix,
			     int n_thr, TextureOperation tex,
			     const ThreadPool* p):
    CObject(p), mat(matrix){
    n_threads = n_thr;
    texture   = tex;
#ifdef CSPMATRIX_EXT
    /*Extended arrays*/
    d_ext_blocks      = new T*[n_devices];
    d_ext_col_indices = new uint*[n_devices];
    d_ext_row_lengths = new uint*[n_devices];
    d_ext_row_indices = new uint*[n_devices];
    d_ext_row_map     = new uint*[n_devices];

    n_ext_blocks      = new uint[n_devices];
#else
    d_blocks          = new T*[n_devices];
    d_col_indices     = new uint* [n_devices];
    d_row_lengths     = new uint* [n_devices];
    d_row_indices     = new uint* [n_devices];
#endif

    for(uint i=0;i<n_devices;i++){
#ifdef CSPMATRIX_EXT
      d_ext_blocks[i]      = 0;
      d_ext_col_indices[i] = 0;
      d_ext_row_lengths[i] = 0;
      d_ext_row_indices[i] = 0;
      d_ext_row_map[i]     = 0;

      n_ext_blocks[i]      = 0;
#else
      d_blocks[i]      = 0;
      d_col_indices[i] = 0;
      d_row_lengths[i] = 0;
      d_row_indices[i] = 0;
#endif

      if(p == 0){
	computeBlockDistribution();
	allocateDevice(0);
      }
    }
  }

  template<int N, class T>
  CSpMatrix<N, T>::~CSpMatrix(){
    for(uint i=0;i<n_devices;i++){
#ifdef CSPMATRIX_EXT
      cgfassert(d_ext_blocks[i] == 0);
      cgfassert(d_ext_col_indices[i] == 0);
      cgfassert(d_ext_row_lengths[i] == 0);
      cgfassert(d_ext_row_indices[i] == 0);
      cgfassert(d_ext_row_map[i]     == 0);
#else
      cgfassert(d_blocks[i] == 0);
      cgfassert(d_col_indices[i] == 0);
      cgfassert(d_row_lengths[i] == 0);
      cgfassert(d_row_indices[i] == 0);
#endif
    }

    if(pool == 0){
      deallocateDevice(0);
    }

#ifdef CSPMATRIX_EXT
    delete[] d_ext_blocks;
    delete[] d_ext_col_indices;
    delete[] d_ext_row_lengths;
    delete[] d_ext_row_indices;
    delete[] d_ext_row_map;

    delete[] n_ext_blocks;
#else
    delete[] d_blocks;
    delete[] d_col_indices;
    delete[] d_row_lengths;
    delete[] d_row_indices;
#endif
  }

  template<int N, class T>
  void CSpMatrix<N, T>::computeBlockDistribution(){
    mat->computeBlockDistribution(mRange, vRange, n_blocks, n_devices);
  }

  template<int N, class T>
  void CSpMatrix<N, T>::allocateDevice(const Thread* caller){
    /*Set device pointers to NULL*/
    uint tid = 0;
    if(caller != 0){
      tid = caller->getId();
    }

#ifdef CSPMATRIX_EXT
    d_ext_blocks[tid]  = 0;
    d_ext_col_indices[tid]  = 0;
    d_ext_row_lengths[tid]  = 0;
    d_ext_row_indices[tid]  = 0;
    d_ext_row_map[tid]      = 0;
#else
    d_blocks[tid]      = 0;
    d_col_indices[tid] = 0;
    d_row_lengths[tid] = 0;
    d_row_indices[tid] = 0;

    n_blocks[tid]      = 0;
#endif
        
    /*Assuming that all blocks are ordered*/
    startBlock[tid] = mat->block_indices[mRange[tid].startRow][0];

    /*Compute number of matrix blocks*/
    for(uint i=mRange[tid].startRow;i<mRange[tid].endRow;i++){
      for(uint j=0;j<mat->row_lengths[i];j++){
	if(mat->block_indices[i][j] != UINT_MAX){
	  n_blocks[tid]++;  
	}
      }
    }
    
#ifndef CSPMATRIX_EXT
    /*Allocate data*/
    cudaSafeMalloc((void**)&d_row_indices[tid],
		   sizeof(int)*(mRange[tid].range+1)); /*Alloc one extra
							 for computing
							 the length*/

    cudaSafeMalloc((void**)&d_blocks[tid], 
		   sizeof(T)*N*N*n_blocks[tid]);

    cudaSafeMalloc((void**)&d_col_indices[tid], sizeof(int)*n_blocks[tid]);

    cudaSafeMalloc((void**)&d_row_lengths[tid], 
		   sizeof(int)*mRange[tid].range);

    /*Copy sparse matrix data*/
    if(((N*N*sizeof(T))%sizeof(SpMatrixBlock<N, T>)) != 0 ){
      /*Copy blocks. Copy each block individually, because each block
	is aligned using a 16 byte offset on amd64 devices. For 2x2
	blocks and larger one complete block fits exactly in one or
	miltiple 16 byte aligned memory block. 1x1 blocks only use 4
	bytes so 12 bytes are in a memory block are not used.*/
      T* tmpBlocks = new T[n_blocks[tid]*N*N];
      for(uint i=0;i<n_blocks[tid];i++){
	for(uint j=0;j<N*N;j++){
	  uint index = i*N*N + j;
	  tmpBlocks[index] = mat->blocks[startBlock[tid]+i].m[j];
	}
      }
      cudaSafeCall(cudaMemcpy(d_blocks[tid], 
			      tmpBlocks, 
			      sizeof(T)*N*N * n_blocks[tid], 
			      cudaMemcpyHostToDevice));
      delete[]tmpBlocks;
    }else{
      cudaSafeCall(cudaMemcpy(d_blocks[tid], 
			      &(mat->blocks[startBlock[tid]].m), 
			      sizeof(T)*N*N * n_blocks[tid], 
			      cudaMemcpyHostToDevice));
    }

    /*Copy column indices*/
    uint* colIndices = new uint[n_blocks[tid]];
    
    uint idx = 0;
    for(uint i=mRange[tid].startRow;i<mRange[tid].endRow;i++){
      for(uint j=0;j<mat->row_lengths[i];j++){
	/*The blocks must be sorted*/
	cgfassert((idx+startBlock[tid]) == mat->block_indices[i][j]);
	
	colIndices[idx++] = mat->col_indices[i][j] * N;
      }
    }

    cudaSafeCall(cudaMemcpy(d_col_indices[tid], 
			    colIndices, 
			    sizeof(int)*n_blocks[tid], 
			    cudaMemcpyHostToDevice));
    delete[] colIndices;

    /*Copy row lengths*/
    cudaSafeCall(cudaMemcpy(d_row_lengths[tid], 
			    mat->row_lengths + mRange[tid].startRow, 
			    sizeof(int)*mRange[tid].range, 
			    cudaMemcpyHostToDevice));

    /*Copy row indices*/
    uint* rowIndices = new uint[mRange[tid].range+1];
    for(uint i=mRange[tid].startRow;i<mRange[tid].endRow;i++){
      rowIndices[i-mRange[tid].startRow] = 
	mat->block_indices[i][0] - startBlock[tid];
      cgfassert(mat->block_indices[i][0] - startBlock[TID]>=0);
    }
    rowIndices[mRange[tid].endRow-mRange[tid].startRow] =     
      rowIndices[mRange[tid].endRow-mRange[tid].startRow -1]
      +mat->row_lengths[mRange[tid].endRow-1]; 

    cudaSafeCall(cudaMemcpy(d_row_indices[tid], 
			    rowIndices, 
			    sizeof(int)*(mRange[tid].range+1), 
			    cudaMemcpyHostToDevice));
    
    delete[] rowIndices;
#endif

#ifdef CSPMATRIX_EXT
    uint blocks_per_cuda_block_x = HNNX(N, n_threads);
    uint blocks_per_cuda_block_y = 
      ceil(n_threads / (N*N))/blocks_per_cuda_block_x;

    
    uint n_cuda_blocks;
    uint dim = mRange[tid].range * N;
    //message("tid = %d, dim = %d", tid, dim);
    if((dim/N) % blocks_per_cuda_block_y != 0){
      n_cuda_blocks = 1 + (dim/N)/blocks_per_cuda_block_y;
    }else{
      n_cuda_blocks = (dim/N)/blocks_per_cuda_block_y;
    }
    
    /*Reorder rows*/
    std::vector<index_map> index_vector;
    index_vector.clear();
    
    for(uint i=mRange[tid].startRow;i<mRange[tid].endRow;i++){
      index_map im;
      im.row = i;
      im.length = mat->row_lengths[i];
      
      index_vector.push_back(im);
    }
    
    sort(index_vector.begin(), index_vector.end(), index_map_sort_func);
    
    /*Count blocks*/
    uint found_blocks = 0;
    uint total_blocks = 0;
    //uint current_block_row = 0;
    uint cuda_block = 0;
    
    uint* ext_row_lengths = new uint[n_cuda_blocks];
    //message("n_cuda_blocks = %d", n_cuda_blocks);
    uint max_row_length_block = 0;
    uint min_row_length_block = 1000000;
    uint avg_row_length_block = 0;
    
    for(uint i=mRange[tid].startRow, j=0;i<mRange[tid].endRow;i++,j++){
      max_row_length_block = MAX(max_row_length_block, 
				 mat->row_lengths[index_vector[j].row]);
      min_row_length_block = MIN(min_row_length_block, 
				 mat->row_lengths[index_vector[j].row]); 
      avg_row_length_block += mat->row_lengths[index_vector[j].row];
      
      found_blocks += mat->row_lengths[index_vector[j].row];
      
      if((j % (blocks_per_cuda_block_y) == blocks_per_cuda_block_y - 1) ||
	 (i==mRange[tid].endRow-1)){
	
	/*Inspected last block of block row. The max row length is the
	  max for this cuda block. Extend the number of blocks for
	  each block row in this cuda block such that they are equal
	  to the maximum number of blocks per row in this cuda block.*/
	
	/*Make max_row_length a multiple of NX*/
	if(max_row_length_block % blocks_per_cuda_block_x == 0){
	  /*Max_row_length is a multiple of NX*/
	}else{
	  max_row_length_block/=blocks_per_cuda_block_x;
	  max_row_length_block++;
	  max_row_length_block*=blocks_per_cuda_block_x;
	}
	
	/*compute total number of blocks (including empty ones) in
	  this cuda block*/
	uint blocks_in_current_block = 
	  max_row_length_block * blocks_per_cuda_block_y;
	
	/*Add to total*/
	total_blocks += blocks_in_current_block;
	
	/*Store extended row_length*/
	ext_row_lengths[cuda_block] = max_row_length_block;
	
	cuda_block++;
	max_row_length_block = 0;
	min_row_length_block = 10000000;
	avg_row_length_block = 0;
      }
      //current_block_row++;
    }
    n_ext_blocks[tid] = total_blocks;
    
    /*Allocate memory for new layout*/
    T* ext_blocks          = new T[N*N*n_ext_blocks[tid]];
    uint*  ext_col_indices = new uint [n_ext_blocks[tid]];
    uint*  ext_row_indices = new uint [n_cuda_blocks];
    uint*  ext_row_map     = new uint [n_cuda_blocks*blocks_per_cuda_block_y];
    uint block_index = 0;
    
    /*Copy row mapping*/
    for(uint i=0;i<n_cuda_blocks*blocks_per_cuda_block_y;i++){
      if(i<mRange[tid].range){
	ext_row_map[i] = index_vector[i].row - mRange[tid].startRow;
      }else{
	ext_row_map[i] = -1;
      }
    }
    
    /*Fill arrays*/
    for(uint i=0;i<n_cuda_blocks;i++){
      /*Save data for cuda block i*/
      
      /*Save sorted row_indices*/
      if(i==0){
	ext_row_indices[i] = 0;
      }else{
	ext_row_indices[i] = 
	  ext_row_indices[i-1] + blocks_per_cuda_block_y * ext_row_lengths[i-1];
      }
      
      for(uint j=0;j<ext_row_lengths[i]/blocks_per_cuda_block_x;j++){
	/*Save data for cuda block i. ext_row_lengths is a multiple of
	  NX. First store the first NX blocks of the first block_row,
	  then the first two block_rows of the second block_row.
	  
	  Next, store the second two blocks of the first row. And so forth.*/
	for(uint k=0;k<blocks_per_cuda_block_y;k++){
	  for(uint l=0;l<blocks_per_cuda_block_x;l++){
	    /*Check if block exists, else store a zero block*/
	    bool outofbounds = false;
	    bool zerocopy = false;
	    
	    uint block_row_id = i * blocks_per_cuda_block_y + k;
	    if(block_row_id >= mRange[tid].range){
	      /*Block row does not exist*/
	      outofbounds = true;
	    }
	    
	    /*Check if block exists in block_row*/
	    if(!outofbounds){
	      if(j*blocks_per_cuda_block_x + l < 
		 mat->row_lengths[index_vector[block_row_id].row]){
		uint row_start_index = 
		  mat->block_indices[index_vector[block_row_id].row][0];
		/*Copy data*/
		for(uint bidx=0;bidx<N*N;bidx++){
		  ext_blocks[block_index * N * N + bidx] =
		    mat->blocks[row_start_index + 
				j*blocks_per_cuda_block_x+l].m[bidx];
		}
		
		ext_col_indices[block_index] = 
		  mat->col_indices[index_vector[block_row_id].row]
		  [j*blocks_per_cuda_block_x+l]*N;
	      }else{
		zerocopy = true;
	      }
	    }
	    
	    if(outofbounds || zerocopy){
	      for(uint bidx=0;bidx<N*N;bidx++){
		ext_blocks[block_index*N*N + bidx] = 0;
	      }
	      ext_col_indices[block_index] = 0;
	    }
	    
	    block_index++;
	  }
	}
      }
	
	


#if 0
	/*Save data for column j in cuda block i*/
	for(uint k=0;k<n_threads/32;k++){
	  /*Save data for warp k*/
	  /*K contains 32 threads*/
	  if(N<8){
	    /*One warp contains 32/(N*N) complete blocks*/
	    for(uint l=0;l<32/(N*N);l++){
	      /*If length of row is smaller than max length, add zeros*/
	      /*Or if a block row does not exists, add zeros*/
	      bool outofbounds = false;
	      bool zerocopy = false;

	      uint block_row_id = i * blocks_per_cuda_block_y + k * (32/(N*N))+l;

	      if(block_row_id >= 
		 mRange[tid].range){
		outofbounds = true;
	      }
	      
	      if(!outofbounds){
		if(j<mat->row_lengths[index_vector[block_row_id].row]){
		  uint row_start_index = 
		    mat->block_indices[index_vector[block_row_id].row][0];
		  /*Copy data*/
		  for(uint bidx=0;bidx<N*N;bidx++){
		    ext_blocks[block_index * N * N + bidx] =
		      mat->blocks[row_start_index + j].m[bidx];
		  }

		  ext_col_indices[block_index] = 
		    mat->col_indices[index_vector[block_row_id].row][j]*N;
		}else{
		  zerocopy = true;
		}
	      }

	      if(zerocopy || outofbounds){
		/*Add zeros*/
		/*Copy data*/
		for(uint bidx=0;bidx<N*N;bidx++){
		  ext_blocks[block_index * N * N + bidx] = 0;
		}
		ext_col_indices[block_index] = 0;
	      }
	    
	      block_index++;
	    }
	  }else{
	    /*N=8 -> k[0..4)*/
	    /*k 0,1 -> row 0. k 2,3 -> row 1*/
	    uint warps_per_block = N*N/32;
	    uint row = k/warps_per_block;

	    uint block_row_id = i*blocks_per_cuda_block_y + row;

	    bool outofbounds = false;
	    bool zerocopy = false;

	    if(block_row_id >= mRange[tid].range){
	      outofbounds = true;
	    }
	    if(!outofbounds){
	      if(j<mat->row_lengths[index_vector[block_row_id].row]){
		/*One block contains N*N/32 warps*/
		for(uint l=0;l<N*N/32;l++){
		  uint row_start_index = 
		    mat->block_indices[index_vector[block_row_id].row][0];
		  for(uint bidx = 0;bidx < 32;bidx++){
		    ext_blocks[block_index * N * N + l*32 + bidx] = 
		      mat->blocks[row_start_index + j].m[l*32 + bidx];
		  }
		}
	      }else{
		zerocopy = true;
	      }
	    }
	    
	    if(zerocopy || outofbounds){
	      /*One block contains N*N/32 warps*/
	      for(uint l=0;l<N*N/32;l++){
		for(uint bidx = 0;bidx < 32;bidx++){
		  ext_blocks[block_index * N * N + l*32 + bidx] = 0;
		}
	      }
	    }
	    
	    if(k%warps_per_block == warps_per_block -1){
	      if(zerocopy || outofbounds){
		ext_col_indices[block_index] = 0;
	      }else{
		ext_col_indices[block_index] = 
		  mat->col_indices[index_vector[block_row_id].row][j]*N;
	      }
	      block_index++;
	    }
	  }
	}
      }
#endif
    }

    cudaSafeMalloc((void**)&d_ext_blocks[tid],
		   sizeof(T)*N*N*n_ext_blocks[tid]);

    cudaSafeMalloc((void**)&d_ext_col_indices[tid],
		   sizeof(uint)*n_ext_blocks[tid]);

    cudaSafeMalloc((void**)&d_ext_row_indices[tid],
		   sizeof(uint)*n_cuda_blocks);

    cudaSafeMalloc((void**)&d_ext_row_lengths[tid],
		   sizeof(uint)*n_cuda_blocks);

    cudaSafeMalloc((void**)&d_ext_row_map[tid],
		   sizeof(uint)*n_cuda_blocks*blocks_per_cuda_block_y);
    
    cudaSafeCall(cudaMemcpy(d_ext_blocks[tid],
			    ext_blocks, sizeof(T)*N*N*n_ext_blocks[tid], 
			    cudaMemcpyHostToDevice));
    
    cudaSafeCall(cudaMemcpy(d_ext_col_indices[tid],
			    ext_col_indices, sizeof(uint)*n_ext_blocks[tid],
			    cudaMemcpyHostToDevice));
    
    cudaSafeCall(cudaMemcpy(d_ext_row_indices[tid], 
			    ext_row_indices, sizeof(uint)*n_cuda_blocks,
			    cudaMemcpyHostToDevice));

    cudaSafeCall(cudaMemcpy(d_ext_row_lengths[tid], 
			    ext_row_lengths, sizeof(uint)*n_cuda_blocks,
			    cudaMemcpyHostToDevice));

    cudaSafeCall(cudaMemcpy(d_ext_row_map[tid], 
			    ext_row_map, 
			    sizeof(uint)*n_cuda_blocks*blocks_per_cuda_block_y,
			    cudaMemcpyHostToDevice));

    delete [] ext_blocks;
    delete [] ext_col_indices;
    delete [] ext_row_indices;
    delete [] ext_row_lengths;
    delete [] ext_row_map;
#endif
  }

  template<int N, class T>
  void CSpMatrix<N, T>::deallocateDevice(const Thread* caller){
    /*De-allocate all used device resources*/
    uint tid = 0;
    if(caller != 0){
      tid = caller->getId();
    }
#ifdef CSPMATRIX_EXT
    cudaSafeFree(d_ext_blocks[tid]);
    cudaSafeFree(d_ext_col_indices[tid]);
    cudaSafeFree(d_ext_row_lengths[tid]);
    cudaSafeFree(d_ext_row_indices[tid]);
    cudaSafeFree(d_ext_row_map[tid]);

    n_ext_blocks[tid] = 0;
#else
    cudaSafeFree(d_row_indices[tid]);
    cudaSafeFree(d_row_lengths[tid]);
    cudaSafeFree(d_col_indices[tid]);
    cudaSafeFree(d_blocks[tid]);
#endif
  }

  template<int N, class T>
  void CSpMatrix<N, T>::preSpmv(const Thread* caller){
    //message("Binding textures");
#ifdef CSPMATRIX_EXT
    bindUIntTexture2(d_ext_col_indices[TID], n_ext_blocks[TID]);
#else
    bindUIntTexture2<T>(d_col_indices[TID], n_blocks[TID]);
#endif
  }

  template<int N, class T>
  void CSpMatrix<N, T>::spmv(CVector<T>* x, const CVector<T>* const b, 
			     const Thread* caller){
    T* res_data = 0;
    /*If x is a full vector, give offset pointer to spmv such that the
      results are stored in the proper place.*/
    if(x->copy){
      res_data = x->getData(TID) + x->vRange[TID].startBlock;
    }else{
      res_data = x->getData(TID);
    }

#ifdef CSPMATRIX_EXT
#ifdef OLD_METHOD 
    ordered_spmv_cuda<N, T>(d_ext_blocks[TID], d_ext_col_indices[TID], 
			    d_ext_row_lengths[TID], 
			    d_ext_row_indices[TID],
			    d_ext_row_map[TID],
			    b->getData(TID), res_data, 
			    mRange[TID].range, n_ext_blocks[TID]);
#else
    switch(n_threads){
    case 128:
      CUDASPMV<N, T, 128>::spmv_ordered(d_ext_blocks[TID], 
					d_ext_col_indices[TID], 
					d_ext_row_lengths[TID], 
					d_ext_row_indices[TID],
					d_ext_row_map[TID],
					b->getData(TID), res_data, 
					mRange[TID].range, 
					n_ext_blocks[TID],
					texture);
      break;
    case 256:
      CUDASPMV<N, T, 256>::spmv_ordered(d_ext_blocks[TID], 
					d_ext_col_indices[TID], 
					d_ext_row_lengths[TID], 
					d_ext_row_indices[TID],
					d_ext_row_map[TID],
					b->getData(TID), res_data, 
					mRange[TID].range, 
					n_ext_blocks[TID],
					texture);
      break;
    case 512:
      CUDASPMV<N, T, 512>::spmv_ordered(d_ext_blocks[TID], 
					d_ext_col_indices[TID], 
					d_ext_row_lengths[TID], 
					d_ext_row_indices[TID],
					d_ext_row_map[TID],
					b->getData(TID), res_data, 
					mRange[TID].range, 
					n_ext_blocks[TID],
					texture);
      break;
    default:
      error("Unsupported number of cuda threads");
    };
#endif
#else
    parallel_spmv_cuda<N, T>(d_blocks[TID], d_col_indices[TID], 
			     d_row_lengths[TID], 
			     d_row_indices[TID], b->getData(TID), res_data, 
			     mRange[TID].range, n_blocks[TID], TID);
#endif
  }

  template class CSpMatrix<1, float>;
  template class CSpMatrix<2, float>;
  template class CSpMatrix<4, float>;
  template class CSpMatrix<8, float>;
  //template class CSpMatrix<16, float>;

  template class CSpMatrix<1, double>;
  template class CSpMatrix<2, double>;
  template class CSpMatrix<4, double>;
  template class CSpMatrix<8, double>;
  //template class CSpMatrix<16, double>;

}

#endif/*CUDA*/
