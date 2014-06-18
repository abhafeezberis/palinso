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

#ifndef SPMATRIX_HPP
#define SPMATRIX_HPP

#include "core/cgfdefs.hpp"
#include "math/Math.hpp"
#include "math/SpMatrixBlock.hpp"
#include "datastructures/Tree.hpp"

#include <ostream>
#include <map>

#ifdef CUDA
/*Number of blocks processed simultaneously*/
#define SIM_BLOCKS  2

/*A block row is divided in a number of segment of size SEG_LENGTH*/
#define SEG_LENGTH  8

#define THREADS 16
#endif

namespace CGF{

  template<class T>
  class CGFAPI Vector;

  /*typedef struct _block_index block_index_t;
    struct _block_index{
    int col;
    int index;
    };*/

  typedef struct _reverse_index reverse_index_t;
  struct _reverse_index{
    int blockRow;
    int blockCol;
  };

  template<int N, class T>
  class CGFAPI SpMatrix;

#ifdef USE_THREADS
  class CGFAPI ThreadPool;

  template<int N, class T>
  class CGFAPI ParallelSPMVTask;
#endif

  template<int N, class T>
  class CGFAPI SpRowProxy{
  public:
    SpRowProxy(int r, const SpMatrix<N, T>* const cm){
      row = r;
      cmatrix = cm;
      matrix = 0;
    }

    SpRowProxy(int r, SpMatrix<N, T>* m){
      row = r;
      matrix = m;
      cmatrix = 0;
    }

    T& operator[](int i);
    T operator[](int i) const;

  protected:
    template <int, class>
      friend class SpMatrix;

    SpRowProxy(){
    }

    SpMatrix<N, T>* matrix;
    const SpMatrix<N, T>* cmatrix;
    int row;
  };

  template<int N, class T>
  class CGFAPI SpMatrix{
  public:
    /*Constructors*/
    SpMatrix();
    SpMatrix(int width, int height);

    /*Copy constructor*/
    SpMatrix(const SpMatrix<N, T>& m);

    /*Destructor*/
    virtual ~SpMatrix();

    /*Assignment operator*/
    SpMatrix<N, T>& operator=(const SpMatrix<N, T>& m);

    /*Enables direct manipulation of the data items. If the element
      does not exists then this function will create the element for
      you!*/
    SpRowProxy<N, T> operator[](int i);

    /*If an element does not exist, this function will not create the
      memory block but returns the zero element*/
    const SpRowProxy<N, T> operator[](int i) const;

    /*Sparse matrix sparse matrix multiplication*/
#if 0
	/*Not available yet*/
    SpMatrix<N, T>  operator* (const SpMatrix<N, T>& m)const;
    SpMatrix<N, T>& operator*=(const SpMatrix<N, T>& m);
#endif

    /*Add similar sparse matrices together*/
    SpMatrix<N, T>  operator+ (const SpMatrix<N, T>& m) const;
    SpMatrix<N, T>& operator+=(const SpMatrix<N, T>& m);
    SpMatrix<N, T>  operator- (const SpMatrix<N, T>& m) const;
    SpMatrix<N, T>& operator-=(const SpMatrix<N, T>& m);

#if 0
    /*Turns the sparse matrix into a full matrix! Perhaps not a good
      idea*/
    SpMatrix<N, T>  operator+ (T f) const;
    SpMatrix<N, T>& operator+=(T f);
    SpMatrix<N, T>  operator- (T f) const;
    SpMatrix<N, T>& operator-=(T f);
#endif

    /*Multiply/divide each non zero value*/
    SpMatrix<N, T>  operator* (T f) const;
    SpMatrix<N, T>& operator*=(T f);
    SpMatrix<N, T>  operator/ (T f) const;
    SpMatrix<N, T>& operator/=(T f);

    /*Test binary relations. The result of an (in)equality test is a
      binary matrix.*/

    /*Binary equality tests*/
    template<int M, class TT>
      friend SpMatrix<M, TT> operator==(const SpMatrix<M, TT>& m, TT n);
    template<int M, class TT>
      friend SpMatrix<M, TT> operator!=(const SpMatrix<M, TT>& m, TT n);
    template<int M, class TT>
      friend SpMatrix<M, TT> operator==(TT n, const SpMatrix<M, TT>& m);
    template<int M, class TT>
      friend SpMatrix<M, TT> operator!=(TT n, const SpMatrix<M, TT>& m);

    /*Binary inequality tests*/
    template<int M, class TT>
      friend SpMatrix<M, TT> operator< (const SpMatrix<M, TT>& m, TT n);
    template<int M, class TT>
      friend SpMatrix<M, TT> operator<=(const SpMatrix<M, TT>& m, TT n);
    template<int M, class TT>
      friend SpMatrix<M, TT> operator> (const SpMatrix<M, TT>& m, TT n);
    template<int M, class TT>
      friend SpMatrix<M, TT> operator>=(const SpMatrix<M, TT>& m, TT n);

    template<int M, class TT>
      friend SpMatrix<M, TT> operator< (TT n, const SpMatrix<M, TT>& m);
    template<int M, class TT>
      friend SpMatrix<M, TT> operator<=(TT n, const SpMatrix<M, TT>& m);
    template<int M, class TT>
      friend SpMatrix<M, TT> operator> (TT n, const SpMatrix<M, TT>& m);
    template<int M, class TT>
      friend SpMatrix<M, TT> operator>=(TT n, const SpMatrix<M, TT>& m);

    /*Set one singe value and increment the number of stored
      elements*/
    void set(int row, int col, T value){
      (*this)[row][col] = value;
      n_elements++;
    }

    //SpMatrix<N, T>* reorderRCM()const;
    //SpMatrix<N, T>* reorderKing()const;
    //SpMatrix<N, T>* reorderMMD()const;
    //SpMatrix<N, T>* reorderMy()const;

    void clear();

    void analyse()const;
    int getNElements()const;
    int getNBlocks()const;
    int getShortestRow()const;
    int getLongestRow()const;
    float getAverageRow()const;
    float getAverageBlockFill()const;

    /*Candidate for removal, use LinSolve classes instead*/
#if 0
    Vector<T>* solveSystemCG(const Vector<T>* b);
    void       solveSystemCG(const Vector<T>* b, Vector<T>* x);
    Vector<T>* solveSystemBICGSTAB(const Vector<T>* b);
    void       solveSystemBICGSTAB(const Vector<T>* b, Vector<T>* x);
    Vector<T>* solveSystemBICGStabl(const Vector<T>* b, int l);
    void       solveSystemBICGStabl(const Vector<T>* b, int l, Vector<T>* x,
                                    int iter=1000);
    Vector<T>* solveSystemCGParallel(const Vector<T>* b, ThreadPool* pool);
    Vector<T>* solveSystemBICGSTABParallel(const Vector<T>* b,
                                           ThreadPool* pool);
#endif
#ifndef NO_CUDA
    Vector<T>* solveSystemCGCuda(const Vector<T>* b, ThreadPool* pool);
#endif

    /*Multiplies matrix M = (D+U+L) with v as D^{-1}(U+L)v*/
    template<int M, class TT>
      friend void multiplyJacobi(Vector<TT>& r, const SpMatrix<M, TT>& m,
                                 const Vector<TT>& v);

    template<int M, class TT>
      friend void multiplyGaussSeidel(Vector<TT>& r, const SpMatrix<M, TT>& m,
                                      const Vector<TT>& v,
                                      const Vector<TT>* b);

    template<int M, class TT>
      friend void spmv(Vector<TT>& r, const SpMatrix<M, TT>& m,
                       const Vector<TT>& v);

    template<int M, class TT>
      friend void spmv_t(Vector<TT>& r, const SpMatrix<M, TT>& m,
                         const Vector<TT>& v);

    template<int M, class TT>
      friend void spmv_parallel(Vector<TT>& r, const SpMatrix<M, TT>& m,
                                const Vector<TT>& v);

    template<int M, class TT>
      friend void spmv_partial(Vector<TT>& r, const SpMatrix<M, TT>& m,
                               const Vector<TT>& v, const MatrixRange rg);

    template<int M, class TT>
      friend Vector<TT> operator*(const SpMatrix<M, TT>&m, const Vector<TT>&v);

    template<int M, int MM, class TPA, class TPB>
      friend void convertMatrix(SpMatrix<M, TPA> & a,
                                const SpMatrix<MM, TPB>& b);

    int getWidth()const {return origWidth;}
    int getHeight()const {return origHeight;}

    void printRow(int row)const;

    template<int M, class TT>
      friend std::ostream& operator<<(std::ostream& stream, const SpMatrix<M, TT>& m);

    void finalize(){
      if(!finalized){
        reorderBlocks();

        /*Allocate new temp buffer for spmv_t*/
        if(tBlocks){
          delete[]tBlocks;
        }

        tBlocks = new SpMatrixBlock<N, T>[width/N];

        finalized = true;
      }
    }

    bool isFinalized()const{
      return finalized;
    }

    //void finalizeParallel(ThreadPool* pool);

    void computeBlockDistribution(MatrixRange* mRange,
                                  VectorRange* vRange,
                                  int* n_blocks,
                                  int n_segments) const;

#if 0
    void setBandwidth(int bw){
      bandwidth = bw;
    }

    int getBandwidth(){
      return bandwidth;
    }
#endif

  protected:
    template<int M, class TT>
      friend class LinSolveGS;

    template<int M, class TT>
      friend class LinSolveBGS;

    template<int M, class TT>
      friend class IECLinSolveGS;

    template<int M, class TT>
      friend class IECLinSolveBGS;

    template<int M, class TT>
      friend class SpRowProxy;

    template<class TT>
      friend class Vector;

    template<int M, class TT>
      friend class ParallelCGCudaTask;

    template<int M, class TT>
      friend class ParallelSPMVCudaTask;

    template<int M, class TT>
      friend class ParallelCGTask;

    template<int M, class TT>
      friend class ParallelSPMVTask;

    template<int M, class TT>
      friend class CSpMatrix;

    template<int M, class TT>
      friend bool save_matrix_matlab(const char* filename,
                                     const SpMatrix<M, TT>* const mat);

    template<int M, class TT>
      friend void save_matrix_market_exchange(const char* filename,
                                              const SpMatrix<M, TT>* const mat);

    int width;
    int height;
    int origWidth;
    int origHeight;

  public:
    //std::map<int, int>* block_map; /*Block lookup map*/
    Tree<int>* block_map;
    int** col_indices;      /*Indices to blocks for each row*/
    int** block_indices;    /*Indices to blocks for each row*/
    int* comp_col_indices;  /*Compressed column indices multiplied with N*/
    int* row_lengths;       /*Stores for each row the length*/
    int* allocated_length;  /*Stores for each row the allocated length */
    SpMatrixBlock<N, T>* blocks; /*Blocks*/
    int n_allocated_blocks; /*Stores the number of allocated blocks*/
    int n_blocks;           /*Number of stored blocks*/
    int n_elements;         /*Number of elements*/
  protected:
    void reorderBlocks();    /*Reorder blocks such that all blocks are
                               stored in a cache coherent order*/
    void grow_row(int row); /*Extends the storage of a row*/
    void grow_blocks();      /*Extends the storafe of for the blocks*/
    SpRowProxy<N, T> proxy;  /*Proxy object for [][] operator*/
    //ParallelSPMVTask<N, T>*task;/*Task descriptor*/
    //ThreadPool* pool;        /*Pool associated with task descriptor*/
    bool finalized;          /*True if the matrix is finalized. If the
                               matrix is altered after finalization,
                               this value becomes false*/

    SpMatrixBlock<N, T>* tBlocks; /*Temporary buffer for storing the
                                    intermediate results for a
                                    transposed multiplication*/
#if 0
    CSVExporter* exporter;   /*Exporter*/
    int bandwidth;
#endif
  };

#ifdef CUDA
  void cuda_sum();
#endif


  template<int N, class T>
  void spmv(Vector<T>& r, const SpMatrix<N, T>& m, const Vector<T>& v);

  template<int N, class T>
  void spmv_t(Vector<T>& r, const SpMatrix<N, T>& m, const Vector<T>& v);

  template<int N, class T>
  void spmv_partial(Vector<T>& r, const SpMatrix<N, T>& m,
                    const Vector<T>& v, const MatrixRange rg);

  template<int N, class T>
  void spmv_parallel(Vector<T>& r, const SpMatrix<N, T>& m,
                     const Vector<T>& v);

  /*Performs a multiplication x^{k+1}' = D^{-1}(L+U)x^{k} and is used
    in Jacobi's method as x^{k+1} = D^{-1}b - x^{k+1}'
  */
  template<int N, class T>
  void multiplyJacobi(Vector<T>& r, const SpMatrix<N, T>& m,
                      const Vector<T>& v);

  /*Performs one Gauss-Seidel step*/
  template<int N, class T>
  void multiplyGaussSeidel(Vector<T>& r, const SpMatrix<N, T>& m,
                           const Vector<T>& v, const Vector<T>* b = 0);

  template<int N, int M, class TPA, class TPB>
  void convertMatrix(SpMatrix<N, TPA> & a, const SpMatrix<M, TPB>& b){
    /*Create new empty matrix*/
    SpMatrix<N, TPA> na(b.getWidth(), b.getHeight());

    /*Assign empty matrix*/
    a = na;
    for(int i=0;i<b.height/M;i++){
      for(int j=0;j<b.row_lengths[i];j++){
        int block_index = b.block_indices[i][j];
        int col_index   = b.col_indices[i][j] * M;
        int idx = 0;
        for(int k=0;k<M;k++){
          for(int l=0;l<M;l++){
            TPB bval = b.blocks[block_index].m[idx++];
            if(bval != (TPB)0.0){
              a[i*M + k][col_index + l] = (TPA)bval;
            }
          }
        }
      }
    }
  }
}

#endif/*SPMATRIX_HPP*/

