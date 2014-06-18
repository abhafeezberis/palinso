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

#include "math/CUDACGOp.hpp"
#include "math/ParallelCGCudaTask.hpp"
#include "util/cuda_util.hpp"
#include "util/device_util.hpp"
#include "math/CVector.hpp"
#include "core/Thread.hpp"

namespace CGF{

#define SIMB (NTHREADS/(N*N))
#define BPW  (32/(N*N))

  void computeVecConfiguration(int dim, dim3* grid, dim3* block){
    int threads = NTHREADS;
    int blocks = 8*30*2;

    if(dim < threads){
      blocks = 1;
    }else if(dim < blocks * threads){
      blocks = (int)ceil((float)dim/(float)threads);
    }else{
      
    }

    grid->x = blocks;
    grid->y = 1;
    grid->z = 1;
    block->x = threads;
    block->y = 1;
    block->z = 1;
  }

  /*r = b - r*/
  /*w = C * r*/
  /*v = C * r*/
  /*tmp = w*w*/
  /*tmp2 = v*v*/
  template<class T>
  __global__ void cuda_cg_step_1(const T* d_b, T* d_C,
                                 T* d_r, T* d_v, T* d_w, 
                                 T* d_tmp, T* d_tmp2, unsigned int dim,
                                 unsigned int startRow, T* d_full_vec, 
                                 T* mapped_memory, unsigned int n_threads){
    unsigned int total_threads = gridDim.x * NTHREADS;
    unsigned int tx        = threadIdx.x;
    unsigned int vec_index = NTHREADS * blockIdx.x + tx;

    while(vec_index < dim){
      T r = d_r[vec_index];
      T C = d_C[vec_index];
      if(C != 0){
        C = T(1.0)/sqrt(abs(d_C[vec_index]));
      }

      T w, v;
      /*Compute*/
      r = d_b[vec_index] - r;
      w = C * r;
      v = C * w;
      
      /*Upload*/
      d_r[vec_index] = r;
      d_w[vec_index] = w;
      d_v[vec_index] = v;
      d_C[vec_index] = C;

      if(d_full_vec != 0){
        d_full_vec[vec_index + startRow] = v;
      }
      if(mapped_memory != 0){
        mapped_memory[vec_index + startRow] = v;
      }
      
      d_tmp[vec_index]  = w * w;
      d_tmp2[vec_index] = v * v;

      vec_index += total_threads;
    }
  }

  template<class T>
  void parallel_cg_step_1(const CVector<T>* b, CVector<T>* C,
                          CVector<T>* r, CVector<T>* v, CVector<T>* w, 
                          CVector<T>* tmp, CVector<T>* tmp2, 
                          CVector<T>* full_vec,
                          T* mapped_memory, int n_threads, 
                          const Thread* caller){
    int dim = b->getVRange(caller->getId())->range;
    int startRow  = b->getVRange(caller->getId())->startBlock;
    int tid = caller->getId();

    dim3 threads;
    dim3 grid;

    computeVecConfiguration(dim, &grid, &threads);

    cuda_cg_step_1<T><<<grid, threads>>>(b->getData(tid), C->getData(tid), 
                                         r->getData(tid), v->getData(tid), 
                                         w->getData(tid), tmp->getData(tid),
                                         tmp2->getData(tid),
                                         b->getVRange(caller->getId())->range, 
                                         startRow, (full_vec==0)?0:full_vec->getData(tid), 
                                         mapped_memory,
                                         n_threads);

    cudaCheckError("Kernel launch cuda_cg_step_1");
  }

  template<int LP, class T>
  __global__ void cuda_reduction(const T* d_tmp, T* d_tmp2, 
                                 T* mapped, unsigned int dim, bool sq){
    unsigned int bx            = blockIdx.x;
    unsigned int total_threads = gridDim.x * NTHREADS;
    unsigned int vec_index     = NTHREADS * bx;
    unsigned int tx            = threadIdx.x;

    __shared__ T vector[NTHREADS];

    vector[tx] = 0;
    
    unsigned int i = vec_index+tx;

    for(i=i;i+(LP-1)*total_threads<dim;i+=LP*total_threads){
      vector[tx] += sum_load_vector<LP>(d_tmp, i, total_threads);
    }
    
    for(i=i;i<dim;i+=total_threads){
      vector[tx] += d_tmp[i];
    }

    __syncthreads();

    /*reduce vector*/
    vector_sum<NTHREADS, T>(vector, tx);

    __syncthreads();

    /*Store result*/
    if(tx==0){
      d_tmp2[bx] = vector[0];
      if(mapped)
        mapped[0] = vector[0];
    }
  }

  template<int LP, class T>
  __global__ void cuda_reduction2(const T* d_tmp, T* d_tmp2, 
                                  T* mapped, unsigned int dim, 
                                  unsigned int n_blocks){
    unsigned int block_index = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int tx          = threadIdx.x;
    unsigned int vec_index   = block_index * NTHREADS * LP + tx;

    if(block_index >= n_blocks)
      return;
    
    __shared__ T vector[NTHREADS];

    vector[tx] = 0;
    if(vec_index + LP * NTHREADS < dim ){
      vector[tx] = sum_load_vector<LP>(d_tmp, vec_index, NTHREADS);
    }else{
      for(int i=0;i<LP;i++){
        if(vec_index + i * NTHREADS < dim){
          vector[tx] += d_tmp[vec_index + i * NTHREADS];
        }
      }
    }

    __syncthreads();

    /*reduce vector*/
    vector_sum<NTHREADS, T>(vector, tx);

    /*Store result*/
    if(tx==0){
      d_tmp2[block_index] = vector[0];
      if(mapped && gridDim.x == 1 && block_index == 0)
        mapped[0] = vector[0];
    }
  }

  template<class T>
  void parallel_reduction(const CVector<T>* vector, CVector<T>* tmp, T** res, 
                          T* mapped, const Thread* caller){
    /*tmp is a temporary scratch vector on the device.*/

    int tid = 0;
    if(caller)
      tid = caller->getId();

    int dim = vector->getVRange(tid)->range;
    if(vector->getVRange(tid)->startBlock == 
       vector->getVRange(tid)->endBlock){
      /*Size is zero*/
      warning("Size of reduced vector is 0");
    }
    dim3 threads, threads2;
    dim3 grid, grid2;

    T* ttmp_a = tmp->getData(tid);
    T* ttmp_b = ttmp_a + (tmp->getVRange(tid)->range / 2);

    /*Pass 1*/
    computeVecConfiguration((int)ceil((float)dim/(float)R_LOOP), 
                            &grid, &threads);

    cuda_reduction<R_LOOP, T><<<grid, threads>>>(vector->getData(tid), 
                                                 ttmp_b, 0, dim, true);
    cudaCheckError("Kernel launch cuda_reduction pass 1");

    threads2.x = NTHREADS;
    grid2.x = 1;
    *res = ttmp_b;

    /*Pass 2*/
    cuda_reduction<R_LOOP, T><<<grid2, threads2>>>(ttmp_b, ttmp_a, 
                                                   mapped, grid.x, false);
    cudaCheckError("Kernel launch cuda_reduction pass 2");
    *res = ttmp_a;

    /*Check for pass 3*/
    if(grid.x < NTHREADS){
      return;
    }
    return;
#if 0
    /*Pass 3*/
    grid.x = 1;
    cuda_reduction<R_LOOP, T><<<grid, threads2>>>(ttmp_a, ttmp_b, 
                                                  mapped, grid2.x, false);
    *res = ttmp_b;
#endif
  }
  
  template<class T>
  void parallel_reduction2(const CVector<T>* vector, CVector<T>* tmp, T** res, 
                           T* mapped, const Thread* caller){
    /*tmp is a temporary scratch vector on the device.*/

    int tid = 0;
    if(caller)
      tid = caller->getId();

    int dim = vector->getVRange(tid)->range;
    dim3 threads;
    dim3 grid;

    T* ttmp_a = tmp->getData(tid);
    T* ttmp_b = ttmp_a + (tmp->getVRange(tid)->range / 2);
    int n_blocks = (int)ceil((float)dim/(float)R_LOOP/(float)NTHREADS);
    int pass = 0;
    T* m = 0;
    while(dim != 1){
      if(n_blocks >= 65536){
        grid.x = (int)ceil(Sqrt((float)n_blocks));
        grid.y = (int)ceil(Sqrt((float)n_blocks));
      }else{
        grid.x = n_blocks;
      }

      if(n_blocks == 1){
        m = mapped;
      }else{
        m = 0;
      }

      threads.x = NTHREADS;
      if(pass == 0){
        cuda_reduction2<R_LOOP, T><<<grid, threads>>>(vector->getData(tid), 
                                                      ttmp_b, 
                                                      m, dim, n_blocks);
        cudaCheckError("Kernel launch cuda_reduction pass 1");
      }else{
        cuda_reduction2<R_LOOP, T><<<grid, threads>>>(ttmp_a, ttmp_b, m, 
                                                      dim, n_blocks);
        cudaCheckError("Kernel launch cuda_reduction pass 2");
      }
      T* tmp = ttmp_b;
      ttmp_b = ttmp_a;
      ttmp_a = tmp;
      dim = n_blocks;
      n_blocks = (int)ceil((float)dim/(float)R_LOOP/(float)NTHREADS);
      pass++;
    }

    *res = ttmp_a;
  }

  /*Compute tmp = v * v*/
  template<class T>
  __global__ void cuda_cg_step_2(const T* d_v, 
                                 T* d_tmp, unsigned int dim){
    unsigned int total_threads = gridDim.x * NTHREADS;
    unsigned int tx        = threadIdx.x;
    unsigned int vec_index = NTHREADS * blockIdx.x + tx;

    while(vec_index < dim){
      T v = d_v[vec_index];
      
      /*Compute*/
      v = v * v;
      
      /*Upload*/
      d_tmp[vec_index] = v;

      vec_index += total_threads;
    }
  }

  template<class T>
  void parallel_cg_step_2(const CVector<T>* v, CVector<T>* tmp, 
                          const Thread* caller){ 
    int dim = v->getVRange(caller->getId())->range;
    dim3 threads;
    dim3 grid;

    computeVecConfiguration(dim, &grid, &threads);

    cuda_cg_step_2<T><<<grid, threads>>>(v->getData(caller->getId()), 
                                         tmp->getData(caller->getId()), dim);
    cudaCheckError("Kernel launch cuda_cg_step_2");
  }

  /*Compute tmp = u * v*/
  template<class T>
  __global__ void cuda_cg_step_3(const T* d_v, const T* d_u, 
                                 T* d_tmp, unsigned int dim){
    unsigned int total_threads = gridDim.x * NTHREADS;
    unsigned int tx        = threadIdx.x;
    unsigned int vec_index = NTHREADS * blockIdx.x + tx;

    while(vec_index < dim){
      /*Load & Compute*/
      T v = d_v[vec_index] * d_u[vec_index];
      
      /*Upload*/
      d_tmp[vec_index] = v;
      vec_index += total_threads;
    }
  }

  template<class T>
  void parallel_cg_step_3(const CVector<T>* v, const CVector<T>* u, 
                          CVector<T>* tmp, const Thread* caller){
    int dim = v->getVRange(caller->getId())->range;
    dim3 threads;
    dim3 grid;

    computeVecConfiguration(dim, &grid, &threads);

    cuda_cg_step_3<T><<<grid, threads>>>(v->getData(caller->getId()), 
                                         u->getData(caller->getId()), 
                                         tmp->getData(caller->getId()), dim);
    cudaCheckError("Kernel launch cuda_cg_step_3");
  }

  /*x = x  + tv*/
  /*r = r  - tu*/
  /*w = C * r*/
  /*tmp = w * w*/
  template<class T>
  __global__ void cuda_cg_step_4(const T* d_v, const T* d_u,
                                 const T* d_C, const T t, 
                                 T* d_w, T* d_x, T* d_r, 
                                 T* d_tmp, unsigned int dim){
    unsigned int total_threads = gridDim.x * NTHREADS;
    unsigned int tx        = threadIdx.x;
    unsigned int vec_index = NTHREADS * blockIdx.x + tx;

    while(vec_index < dim){
      T w, x, r;
      x = d_x[vec_index];
      r = d_r[vec_index];
      
      /*Compute*/
      x += t * d_v[vec_index];
      r -= t * d_u[vec_index];
      w  = d_C[vec_index] * r;
      
      /*Upload*/
      d_x[vec_index]   = x;
      d_r[vec_index]   = r;
      d_w[vec_index]   = w;
      d_tmp[vec_index] = w * w;
      vec_index += total_threads;
    }
  }

  template<class T>
  void parallel_cg_step_4(const CVector<T>* v, const CVector<T>* u, 
                          const CVector<T>* C, const T t, 
                          CVector<T>* w, CVector<T>* x, CVector<T>* r, 
                          CVector<T>* tmp, const Thread* caller){
    int dim = x->getVRange(caller->getId())->range;
    dim3 threads;
    dim3 grid;

    computeVecConfiguration(dim, &grid, &threads);

    cuda_cg_step_4<T><<<grid, threads>>>(v->getData(caller->getId()), 
                                         u->getData(caller->getId()), 
                                         C->getData(caller->getId()),
                                         t,
                                         w->getData(caller->getId()), 
                                         x->getData(caller->getId()), 
                                         r->getData(caller->getId()), 
                                         tmp->getData(caller->getId()), 
                                         dim);

    cudaCheckError("Kernel launch cuda_cg_step_4");
  }

  /*v = C * w  + sv*/
  /*tmp = v * v*/
  /*Fullvec = v*/
  template<class T>
  __global__ void cuda_cg_step_5(const T* d_w, const T* d_C,
                                 const T s, T* d_v, T* d_tmp,
                                 T* d_full_vec, T* mapped_memory,
                                 unsigned int dim, unsigned int startRow,
                                 unsigned int n_threads){

    unsigned int total_threads = gridDim.x * NTHREADS;
    unsigned int tx        = threadIdx.x;
    unsigned int vec_index = NTHREADS * blockIdx.x + tx;

    while(vec_index < dim){
      /*Compute*/
      T v = d_C[vec_index] * d_w[vec_index] + s * d_v[vec_index];
      
      /*Upload*/
      d_v[vec_index] = v;
      d_tmp[vec_index] = v * v;
      if(d_full_vec != 0){
        d_full_vec[vec_index + startRow] = v;
      }
      if(mapped_memory != 0){
        mapped_memory[vec_index + startRow] = v;
      }
      vec_index += total_threads;
    }
  }

  template<class T>
  void parallel_cg_step_5(const CVector<T>* w, const CVector<T>* C,
                          const T s, CVector<T>* v, CVector<T>* tmp,
                          CVector<T>* full_vec, T* mapped_memory, 
                          int n_threads, const Thread* caller){
    int dim = v->getVRange(caller->getId())->range;
    int startRow = v->getVRange(caller->getId())->startBlock;
    dim3 threads;
    dim3 grid;

    computeVecConfiguration(dim, &grid, &threads);

    cuda_cg_step_5<T><<<grid, threads>>>(w->getData(caller->getId()), 
                                         C->getData(caller->getId()),
                                         s,
                                         v->getData(caller->getId()),
                                         tmp->getData(caller->getId()),
                                         (full_vec == 0)?0:full_vec->getData(caller->getId()),
                                         mapped_memory, 
                                         dim, startRow, 
                                         n_threads);
    cudaCheckError("Kernel launch cuda_cg_step_5");
  }

  template
  void parallel_reduction<float>(const CVector<float>* vector, 
                                 CVector<float>* tmp, 
                                 float** res, 
                                 float* mapped, const Thread* caller);

  template
  void parallel_reduction2<float>(const CVector<float>* vector, 
                                  CVector<float>* tmp, 
                                  float** res, 
                                  float* mapped, const Thread* caller);


  template
  void parallel_cg_step_1<float>(const CVector<float>* b, 
                                 CVector<float>* C,
                                 CVector<float>* r, CVector<float>* v, 
                                 CVector<float>* w, 
                                 CVector<float>* tmp, 
                                 CVector<float>* tmp2, 
                                 CVector<float>* full_vec,
                                 float* mapped_memory, int n_threads, 
                                 const Thread* caller);

  template
  void parallel_cg_step_2<float>(const CVector<float>* v, 
                                 CVector<float>* tmp, 
                                 const Thread* caller);

  template
  void parallel_cg_step_3<float>(const CVector<float>* v, 
                                 const CVector<float>* u, 
                                 CVector<float>* tmp, const Thread* caller);

  template
  void parallel_cg_step_4<float>(const CVector<float>* v, 
                                 const CVector<float>* u, 
                                 const CVector<float>* C, 
                                 const float t, 
                                 CVector<float>* w, 
                                 CVector<float>* x, 
                                 CVector<float>* r, 
                                 CVector<float>* tmp,
                                 const Thread* caller);

  template
  void parallel_cg_step_5<float>(const CVector<float>* w, 
                                 const CVector<float>* C,
                                 const float s, 
                                 CVector<float>* v, 
                                 CVector<float>* tmp,
                                 CVector<float>* full_vec, 
                                 float* mapped_memory, 
                                 int n_threads, const Thread* caller);

  template
  void parallel_reduction<double>(const CVector<double>* vector, 
                                  CVector<double>* tmp, 
                                  double** res, 
                                  double* mapped, const Thread* caller);

  template
  void parallel_reduction2<double>(const CVector<double>* vector, 
                                   CVector<double>* tmp, 
                                   double** res, 
                                   double* mapped, const Thread* caller);

  template
  void parallel_cg_step_1<double>(const CVector<double>* b, 
                                  CVector<double>* C,
                                  CVector<double>* r, CVector<double>* v, 
                                  CVector<double>* w, 
                                  CVector<double>* tmp, 
                                  CVector<double>* tmp2, 
                                  CVector<double>* full_vec,
                                  double* mapped_memory, int n_threads, 
                                  const Thread* caller);

  template
  void parallel_cg_step_2<double>(const CVector<double>* v, 
                                  CVector<double>* tmp, 
                                  const Thread* caller);

  template
  void parallel_cg_step_3<double>(const CVector<double>* v, 
                                  const CVector<double>* u, 
                                  CVector<double>* tmp, const Thread* caller);

  template
  void parallel_cg_step_4<double>(const CVector<double>* v, 
                                  const CVector<double>* u, 
                                  const CVector<double>* C, 
                                  const double t, 
                                  CVector<double>* w, 
                                  CVector<double>* x, 
                                  CVector<double>* r, 
                                  CVector<double>* tmp,
                                  const Thread* caller);

  template
  void parallel_cg_step_5<double>(const CVector<double>* w, 
                                  const CVector<double>* C,
                                  const double s, 
                                  CVector<double>* v, 
                                  CVector<double>* tmp,
                                  CVector<double>* full_vec, 
                                  double* mapped_memory, 
                                  int n_threads, const Thread* caller);

}

