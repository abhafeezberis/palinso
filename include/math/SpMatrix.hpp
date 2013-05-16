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

#include <ostream>
#include <map>

/*Number of blocks processed simultaneously*/
#define SIM_BLOCKS  2

/*A block row is divided in a number of segment of size SEG_LENGTH*/
#define SEG_LENGTH  8

#define THREADS 16

namespace CGF{

  template<class T>
  class CGFAPI Vector;

  /*typedef struct _block_index block_index_t;  
  struct _block_index{
    uint col;
    uint index;
    };*/

  typedef struct _reverse_index reverse_index_t;
  struct _reverse_index{
    int blockRow;
    int blockCol;
  };

  template<int N, class T>
  class CGFAPI SpMatrix;
  class CGFAPI ThreadPool;

  template<int N, class T>
  class CGFAPI ParallelSPMVTask;

  template <int N, class T>
  class CGFAPI SpRowProxy{
  public:
    SpRowProxy(uint r, const SpMatrix<N, T>* const cm){
      row = r;
      cmatrix = cm;
      matrix = 0;
    }

    SpRowProxy(uint r, SpMatrix<N, T>* m){
      row = r;
      matrix = m;
      cmatrix = 0;
    }

    T& operator[](uint i);
    T operator[](uint i) const;

  protected:
    template <int, class>
      friend class SpMatrix;
    
    SpRowProxy(){
    }
    
    SpMatrix<N, T>* matrix;
    const SpMatrix<N, T>* cmatrix;
    uint row;
  };
  


  template<int N, class T>
  class CGFAPI SpMatrix{
  public:
    /*Constructors*/
    SpMatrix();
    SpMatrix(uint width, uint height);
    
    /*Copy constructor*/
    SpMatrix(const SpMatrix<N, T>& m);

    /*Destructor*/
    virtual ~SpMatrix();

    /*Assignment operator*/
    SpMatrix<N, T>& operator=(const SpMatrix<N, T>& m);

    /*Enables direct manipulation of the data items. If the element
      does not exists then this function will create the element for
      you!*/
    SpRowProxy<N, T> operator[](uint i);

    /*If an element does not exist, this function will not create the
      memory block but returns the zero element*/
    const SpRowProxy<N, T> operator[](uint i) const;

    /*Sparse matrix sparse matrix multiplication*/
    SpMatrix<N, T>  operator* (const SpMatrix<N, T>& m)const;
    SpMatrix<N, T>& operator*=(const SpMatrix<N, T>& m);

    /*Add similar sparse matrices together*/
    SpMatrix<N, T>  operator+ (const SpMatrix<N, T>& m) const;
    SpMatrix<N, T>& operator+=(const SpMatrix<N, T>& m);
    SpMatrix<N, T>  operator- (const SpMatrix<N, T>& m) const;
    SpMatrix<N, T>& operator-=(const SpMatrix<N, T>& m);

    /*Turns the sparse matrix into a full matrix! Perhaps not a good
      idea*/
    SpMatrix<N, T>  operator+ (T f) const;
    SpMatrix<N, T>& operator+=(T f);
    SpMatrix<N, T>  operator- (T f) const;
    SpMatrix<N, T>& operator-=(T f);

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
    void set(uint row, uint col, T value){
      (*this)[row][col] = value;
      n_elements++;
    }

    SpMatrix<N, T>* reorderRCM()const;
    SpMatrix<N, T>* reorderKing()const;
    SpMatrix<N, T>* reorderMMD()const;
    SpMatrix<N, T>* reorderMy()const;

    void clear();

    void analyse()const;
    uint getNElements()const;
    uint getNBlocks()const;
    uint getShortestRow()const;
    uint getLongestRow()const;
    float getAverageRow()const;
    float getAverageBlockFill()const;

    /*Candidate for removal, use LinSolve classes instead*/
#if 0
    Vector<T>* solveSystemCG(const Vector<T>* b); 
    void       solveSystemCG(const Vector<T>* b, Vector<T>* x); 
    Vector<T>* solveSystemBICGSTAB(const Vector<T>* b); 
    void       solveSystemBICGSTAB(const Vector<T>* b, Vector<T>* x); 
    Vector<T>* solveSystemBICGStabl(const Vector<T>* b, uint l); 
    void       solveSystemBICGStabl(const Vector<T>* b, uint l, Vector<T>* x, 
				    uint iter=1000); 
    Vector<T>* solveSystemCGParallel(const Vector<T>* b, ThreadPool* pool);
    Vector<T>* solveSystemBICGSTABParallel(const Vector<T>* b, 
					   ThreadPool* pool);
#endif
#ifndef NO_CUDA
    Vector<T>* solveSystemCGCuda(const Vector<T>* b, ThreadPool* pool); 
#endif

    template<int M, class TT>
      friend void spmv(Vector<TT>& r, const SpMatrix<M, TT>& m, 
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

    uint getWidth()const {return origWidth;}
    uint getHeight()const {return origHeight;}

    void printRow(uint row)const;

    template<int M, class TT>
      friend std::ostream& operator<<(std::ostream& stream, const SpMatrix<M, TT>& m);

    void finalize(){
      if(!finalized){
	reorderBlocks();
	finalized = true;
      }
    }
    
    bool isFinalized()const{
      return finalized;
    }
    
    void finalizeParallel(ThreadPool* pool);

    void computeBlockDistribution(MatrixRange* mRange,
				  VectorRange* vRange,
				  uint* n_blocks,
				  uint n_segments) const;

#if 0
    void setBandwidth(uint bw){
      bandwidth = bw;
    }

    uint getBandwidth(){
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

    uint width;
    uint height;
    uint origWidth;
    uint origHeight;

  public:
    std::map<uint, uint>* block_map; /*Block lookup map*/
    uint** col_indices;      /*Indices to blocks for each row*/
    uint** block_indices;    /*Indices to blocks for each row*/
    uint* comp_col_indices;  /*Compressed column indices multiplied with N*/
    uint* row_lengths;       /*Stores for each row the length*/
    uint* allocated_length;  /*Stores for each row the allocated length */
    SpMatrixBlock<N, T>* blocks; /*Blocks*/
    uint n_allocated_blocks; /*Stores the number of allocated blocks*/
    uint n_blocks;           /*Number of stored blocks*/
    uint n_elements;         /*Number of elements*/
  protected:
    void reorderBlocks();    /*Reorder blocks such that all blocks are
			       stored in a cache coherent order*/
    void grow_row(uint row); /*Extends the storage of a row*/
    void grow_blocks();      /*Extends the storafe of for the blocks*/
    SpRowProxy<N, T> proxy;  /*Proxy object for [][] operator*/
    //ParallelSPMVTask<N, T>*task;/*Task descriptor*/
    //ThreadPool* pool;        /*Pool associated with task descriptor*/
    bool finalized;          /*True if the matrix is finalized. If the
			       matrix is altered after finalization,
			       this value becomes false*/
#if 0
    CSVExporter* exporter;   /*Exporter*/
    uint bandwidth;
#endif
  };

  void cuda_sum();


  template<int N, class T>
  void spmv(Vector<T>& r, const SpMatrix<N, T>& m, const Vector<T>& v);

  template<int N, class T>
  void spmv_partial(Vector<T>& r, const SpMatrix<N, T>& m, 
		    const Vector<T>& v, 
		    uint startRow, uint endRow);
  template<int N, class T>
  void spmv_parallel(Vector<T>& r, const SpMatrix<N, T>& m, const Vector<T>& v);

  template<int N, int M, class TPA, class TPB>
  void convertMatrix(SpMatrix<N, TPA> & a, const SpMatrix<M, TPB>& b){
    /*Create new empty matrix*/
    SpMatrix<N, TPA> na(b.getWidth(), b.getHeight());

    /*Assign empty matrix*/
    a = na;
    for(uint i=0;i<b.height/M;i++){
      for(uint j=0;j<b.row_lengths[i];j++){
	uint block_index = b.block_indices[i][j];
	uint col_index   = b.col_indices[i][j] * M;
	uint idx = 0;
	for(uint k=0;k<M;k++){
	  for(uint l=0;l<M;l++){
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

