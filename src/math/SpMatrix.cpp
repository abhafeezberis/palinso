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

#include "math/SpMatrix.hpp"
#include "math/Vector.hpp"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if 0
#include "core/Task.hpp"
#include "core/Thread.hpp"
#include "core/ThreadPool.hpp"
#include "core/Timer.hpp"
#endif

#include "core/Exception.hpp"

namespace CGF{
  template<int N, class T>
  T& SpRowProxy<N, T>::operator[](int c){
    int col = c;
    bool block_exists = false;
    int block_index;
    int block_row_col;
    int block_row_index = row/N;
    int block_col_index = col/N;

    if(!(block_col_index < matrix->width  / N)){
      message("Index = %d, %d", row, col);
      error("Index out of bounds");
    }

    if(!(block_row_index < matrix->height / N)){
      message("Index = %d, %d", row, col);
      error("Index out of bounds");
    }

    cgfassert(col < matrix->origWidth);

    cgfassert(block_col_index < matrix->width  / N);
    cgfassert(block_row_index < matrix->height / N);

    /*Find target block*/
#if 0
    for(int i=0;i<matrix->row_lengths[block_row_index];i++){
      if(matrix->col_indices[block_row_index][i] == block_col_index){
        block_exists = true;
        block_row_col = i;
        break;
      }
    }
#else
    //std::map<int, int>::iterator it =
    //matrix->block_map[block_row_index].find(block_col_index);
    int idx = matrix->block_map[block_row_index].findIndex(block_col_index);
    //if(it == matrix->block_map[block_row_index].end()){
    if(idx == Tree<int>::undefinedIndex){
      /*Block not found*/
      block_exists = false;
    }else{
      /*Block found*/
      block_exists = true;
      block_row_col = idx;//it->second;
      //block_row_col = it->second;
    }
#endif

    if(!block_exists){
      /*Create a new block*/
      if(matrix->n_blocks == matrix->n_allocated_blocks){
        matrix->grow_blocks();
      }
#if 0
      if(!matrix->row_lengths[block_row_index] <=
         matrix->allocated_length[block_row_index]){
        printf("Block row index = %d, row_lengths = %d, allocated = %d\n",
               block_row_index, matrix->row_lengths[block_row_index],
               matrix->allocated_length[block_row_index]);
      }
#endif
      cgfassert(matrix->row_lengths[block_row_index] <=
                matrix->allocated_length[block_row_index]);

      if(matrix->row_lengths[block_row_index] ==
         matrix->allocated_length[block_row_index]){
        matrix->grow_row(block_row_index);
      }

      matrix->block_indices[block_row_index][matrix->row_lengths[block_row_index]] = matrix->n_blocks;
      matrix->col_indices[block_row_index][matrix->row_lengths[block_row_index]] = block_col_index;
      block_index = matrix->n_blocks;

      //matrix->block_map[block_row_index][block_col_index] = matrix->row_lengths[block_row_index];
      matrix->block_map[block_row_index].insert(block_col_index, matrix->row_lengths[block_row_index]);

      matrix->row_lengths[block_row_index]++;

      /*Reset block*/
      memset(&(matrix->blocks[matrix->n_blocks]), 0, sizeof(SpMatrixBlock<N, T>));
      matrix->n_blocks++;
      //printf("%d, ", block_index);

      /*Structure of matrix has been altered, finalization is needed.*/
      matrix->finalized = false;
    }else{
      block_index = matrix->block_indices[block_row_index][block_row_col];

      /*Structure has not been altered, no finalization is needed yet.*/
    }

    int block_row = row%N;
    int block_col = col%N;

    return matrix->blocks[block_index].m[block_row*N + block_col];
  }

  template<int N, class T>
  T SpRowProxy<N, T>::operator[](int c)const{
    int col = c;
    bool block_exists = false;
    int block_index;
    int block_row_col;
    int block_row_index = row/N;
    int block_col_index = col/N;

    cgfassert(col < cmatrix->getWidth());

    cgfassert(block_col_index < cmatrix->width  / N);
    cgfassert(block_row_index < cmatrix->height / N);

    if(!(block_col_index < cmatrix->width  / N)){
      message("Index = %d, %d", row, col);
      error("Index out of bounds");
    }

    if(!(block_row_index < cmatrix->height / N)){
      message("Index = %d, %d", row, col);
      error("Index out of bounds");
    }

#if 0
    /*Find target block*/
    for(int i=0;i<cmatrix->row_lengths[block_row_index];i++){
      if(cmatrix->col_indices[block_row_index][i] == block_col_index){
        block_exists = true;
        block_row_col = i;
        break;
      }
    }
#else
    //std::map<int, int>::iterator it =
    //cmatrix->block_map[block_row_index].find(block_col_index);
    int idx = cmatrix->block_map[block_row_index].findIndex(block_col_index);
    //if(it == cmatrix->block_map[block_row_index].end()){
    if(idx == Tree<int>::undefinedIndex){
      /*Block not found*/
      block_exists = false;
    }else{
      /*Block found*/
      block_exists = true;
      block_row_col = idx;//it->second;
      //block_row_col = it->second;
    }
#endif

    if(!block_exists){
      /*Block does not exist and since this is a const object we can
        not create new blocks*/
      return 0.0;
    }else{
      block_index = cmatrix->block_indices[block_row_index][block_row_col];
    }

    int block_row = row%N;
    int block_col = col%N;

    return cmatrix->blocks[block_index].m[block_row*N + block_col];
  }

  template<int N, class T>
  SpRowProxy<N, T> SpMatrix<N, T>::operator[](int i){
    cgfassert(i<origHeight);
    return SpRowProxy<N, T>(i, this);
  }

  template<int N, class T>
  const SpRowProxy<N, T> SpMatrix<N, T>::operator[](int i)const{
    cgfassert(i<origHeight);
    const SpMatrix<N, T>* cmatrix = (const SpMatrix<N, T>*)this;
    return SpRowProxy<N, T>(i, cmatrix);
  }


  template<int N, class T>
  SpMatrix<N, T>::SpMatrix():width(0), height(0), origWidth(0),
                             origHeight(0), block_map(0), col_indices(0),
                             block_indices(0), comp_col_indices(0),
                             row_lengths(0), allocated_length(0), blocks(0),
                             n_allocated_blocks(0), n_blocks(0),
                             n_elements(0), /*task(0), pool(0),*/ finalized(false), tBlocks(0){
  }

  template<int N, class T>
  SpMatrix<N, T>::SpMatrix(int w, int h):block_map(0), col_indices(0),
                                         block_indices(0),
                                         comp_col_indices(0),
                                         row_lengths(0), allocated_length(0),
                                         blocks(0),
                                         n_allocated_blocks(0), n_blocks(0),
                                         n_elements(0), /*task(0), pool(0),*/
                                         finalized(false), tBlocks(0){
    //size = w;
    origWidth = w;
    origHeight = h;

    if(h%16 == 0){
      height = h;
    }else{
      height = ((h/16)+1)*16;
    }

    if(w%16 == 0){
      width = w;
    }else{
      width = ((w/16)+1)*16;
    }

    /*Allocate data for rows*/
    if((height/N)%32 == 0){
      row_lengths      = new int[height/N];
      allocated_length = new int[height/N];
      col_indices      = new int*[height/N];
      block_indices    = new int*[height/N];
      //block_map        = new std::map<int, int>[height/N];
      block_map        = new Tree<int>[height/N];
    }else{
      int extendedHeight = (height/N)/32;
      extendedHeight++;
      extendedHeight*=32;
      row_lengths      = new int[extendedHeight];
      allocated_length = new int[extendedHeight];
      col_indices      = new int*[extendedHeight];
      block_indices    = new int*[extendedHeight];
      //block_map        = new std::map<int, int>[extendedHeight];
      block_map        = new Tree<int>[extendedHeight];
    }

    for(int i=0;i<height/N;i++){
      allocated_length[i] = 32;
      row_lengths[i]      = 0;
      col_indices[i]      = new int[32];
      block_indices[i]    = new int[32];
    }

    /*Allocate blocks*/
    n_allocated_blocks = 1024;
    n_blocks = 0;
    n_elements = 0;

    alignedMalloc((void**)&blocks, 16,
                  (uint)n_allocated_blocks * sizeof(SpMatrixBlock<N, T>));

    proxy.matrix = this;

    comp_col_indices = 0;
    //task = 0;
    //pool = 0;
    //exporter = 0;
  }

  template<int N, class T>
  SpMatrix<N, T>::~SpMatrix(){
    alignedFree(blocks);

    for(int i=0;i<height/N;i++){
      delete [] col_indices[i];
      delete [] block_indices[i];
    }

    delete [] block_map;

    delete [] col_indices;
    delete [] block_indices;
    delete [] allocated_length;
    delete [] row_lengths;
    if(comp_col_indices)
      delete[] comp_col_indices;

#if 0
    if(task)
      delete task;
#endif

    if(tBlocks){
      delete[]tBlocks;
    }
  }

  template<int N, class T>
  void SpMatrix<N, T>::clear(){
    for(int i=0;i<n_blocks;i++){
      blocks[i].clear();
    }
  }

  template<int N, class T>
  SpMatrix<N, T>::SpMatrix(const SpMatrix<N, T>& m):width(0), height(0),
                                                    origWidth(0),
                                                    origHeight(0),
                                                    block_map(0),
                                                    col_indices(0),
                                                    block_indices(0),
                                                    comp_col_indices(0),
                                                    row_lengths(0),
                                                    allocated_length(0),
                                                    blocks(0),
                                                    n_allocated_blocks(0),
                                                    n_blocks(0),
                                                    n_elements(0)/*, task(0),
                                                                   pool(0)*/{
    *this = m;
  }

  template<int N, class T>
  SpMatrix<N, T>& SpMatrix<N, T>::operator=(const SpMatrix<N, T>& m){
    if(this == &m){
      /*Self assignment*/
      return *this;
    }

    if(blocks){
      alignedFree(blocks);

      for(int i=0;i<height/N;i++){
        delete [] col_indices[i];
        delete [] block_indices[i];
      }

      delete [] block_map;
      delete [] col_indices;
      delete [] block_indices;
      delete [] allocated_length;
      delete [] row_lengths;
    }

    height             = m.height;
    width              = m.width;
    origHeight         = m.origHeight;
    origWidth          = m.origWidth;
    n_allocated_blocks = m.n_allocated_blocks;
    n_blocks           = m.n_blocks;
    n_elements         = m.n_elements;

    proxy.matrix       = this;
    proxy.cmatrix      = this;

    /*Allocate new arrays*/
    if((height/N)%32 == 0){
      row_lengths      = new int[height/N];
      allocated_length = new int[height/N];
      col_indices      = new int*[height/N];
      block_indices    = new int*[height/N];
      //block_map        = new std::map<int, int>[height/N];
      block_map        = new Tree<int>[height/N];
    }else{
      int extendedHeight = (height/N)/32;
      extendedHeight++;
      extendedHeight*=32;
      row_lengths      = new int[extendedHeight];
      allocated_length = new int[extendedHeight];
      col_indices      = new int*[extendedHeight];
      block_indices    = new int*[extendedHeight];
      //block_map        = new std::map<int, int>[extendedHeight];
      block_map        = new Tree<int>[extendedHeight];
    }

    /*Copy data*/
    memcpy(row_lengths, m.row_lengths, sizeof(int)*(uint)height/N);
    memcpy(allocated_length, m.allocated_length,
           sizeof(int)*(uint)height/N);

    for(int i=0;i<height/N;i++){
      col_indices[i]   = new int[allocated_length[i]];
      block_indices[i] = new int[allocated_length[i]];

      memcpy(col_indices[i], m.col_indices[i],
             sizeof(int)*(uint)allocated_length[i]);
      memcpy(block_indices[i], m.block_indices[i],
             sizeof(int)*(uint)allocated_length[i]);

      block_map[i] = m.block_map[i];
    }

    /*Allocate blocks*/
    alignedMalloc((void**)&blocks, 16,
                  (uint)n_allocated_blocks * sizeof(SpMatrixBlock<N, T>));

    /*Copy blocks*/
    memcpy(blocks, m.blocks,
           sizeof(SpMatrixBlock<N, T>)*(uint)n_allocated_blocks);

    //task = 0;
    //pool = 0;

    if(comp_col_indices){
      delete [] comp_col_indices;
    }

    comp_col_indices = 0;

    return *this;
  }

  template<int N, class T>
  SpMatrix<N, T> SpMatrix<N, T>::operator+(const SpMatrix<N, T>& m) const{
    /*@Todo, add complete blocks and improve speed. Now O(N^2)*/
    SpMatrix<N, T> m2 = *this;

    m2 += m;
    return m2;
  }

  template<int N, class T>
  SpMatrix<N, T>& SpMatrix<N, T>::operator+=(const SpMatrix<N, T>& m){
    cgfassert(origWidth == m.origWidth);
    cgfassert(origHeight == m.origHeight);

    for(int i=0;i<m.height/N;i++){
      for(int j=0;j<m.row_lengths[i];j++){
        int block_index = m.block_indices[i][j];
        int block_col   = m.col_indices[i][j] * N;

        for(int k=0;k<N*N;k++){
          int l_row = k/N;
          int l_col = k%N;
          int row = i*N + l_row;
          int col = block_col + l_col;

          T val = m.blocks[block_index].m[k];

          if(val != 0){
            (*this)[row][col] += val;
          }
        }
      }
    }
    return *this;
  }

  template<int N, class T>
  SpMatrix<N, T> SpMatrix<N, T>::operator-(const SpMatrix<N, T>& m) const{
    cgfassert(origWidth == m.origWidth);
    cgfassert(origHeight == m.origHeight);
    SpMatrix<N, T> m2 = *this;

    m2 -= m;
    return m2;
  }

  template<int N, class T>
  SpMatrix<N, T>& SpMatrix<N, T>::operator-=(const SpMatrix<N, T>& m){
    cgfassert(origWidth == m.origWidth);
    cgfassert(origHeight == m.origHeight);

    for(int i=0;i<m.height/N;i++){
      for(int j=0;j<m.row_lengths[i];j++){
        int block_index = m.block_indices[i][j];
        int block_col   = m.col_indices[i][j] * N;

        for(int k=0;k<N*N;k++){
          int l_row = k/N;
          int l_col = k%N;
          int row = i*N + l_row;
          int col = block_col + l_col;

          T val = m.blocks[block_index].m[k];

          if(val != 0){
            (*this)[row][col] -= val;
          }
        }
      }
    }
    return *this;
  }

  template<int N, class T>
  SpMatrix<N, T> SpMatrix<N, T>::operator*(T f) const{
    SpMatrix<N, T> m2 = *this;

    for(int i=0;i<m2.n_blocks;i++){
      m2.blocks[i].mul(f);
    }
    return m2;
  }

  template<int N, class T>
  SpMatrix<N, T>& SpMatrix<N, T>::operator*=(T f){
    for(int i=0;i<n_blocks;i++){
      blocks[i].mul(f);
    }
    return *this;
  }

  template<int N, class T>
  SpMatrix<N, T> SpMatrix<N, T>::operator/(T f) const{
    SpMatrix<N, T> m2 = *this;

    for(int i=0;i<m2.n_blocks;i++){
      m2.blocks[i].div(f);
    }
    return m2;
  }

  template<int N, class T>
  SpMatrix<N, T>& SpMatrix<N, T>::operator/=(T f){
    for(int i=0;i<n_blocks;i++){
      blocks[i].div(f);
    }
    return *this;
  }

  template<int N, class T>
  SpMatrix<N, T> operator==(const SpMatrix<N, T>& m, T f){
    SpMatrix<N, T> r(m);

    for(int i=0;i<r.n_blocks;i++){
      r.blocks[i].cmpeq(f);
    }
    return r;
  }

  template<int N, class T>
  SpMatrix<N, T> operator!=(const SpMatrix<N, T>& m, T f){
    SpMatrix<N, T> r(m);
    for(int i=0;i<r.n_blocks;i++){
      r.blocks[i].cmpneq(f);
    }
    return r;
  }

  template<int N, class T>
  SpMatrix<N, T> operator==(T f, const SpMatrix<N, T>& m){
    return m==f;
  }

  template<int N, class T>
  SpMatrix<N, T> operator!=(T f, const SpMatrix<N, T>& m){
    return m!=f;
  }

  template<int N, class T>
  SpMatrix<N, T> operator<(const SpMatrix<N, T>& m, T f){
    SpMatrix<N, T> r(m);
    /*Just inspect all the blocks*/
    for(int i=0;i<r.n_blocks;i++){
      r.blocks[i].cmplt(f);
    }
    return r;
  }

  template<int N, class T>
  SpMatrix<N, T> operator<=(const SpMatrix<N, T>& m, T f){
    SpMatrix<N, T> r(m);
    /*Just inspect all the blocks*/
    for(int i=0;i<r.n_blocks;i++){
      r.blocks[i].cmple(f);
    }
    return r;
  }

  template<int N, class T>
  SpMatrix<N, T> operator>(const SpMatrix<N, T>& m, T f){
    SpMatrix<N, T> r(m);
    /*Just inspect all the blocks*/
    for(int i=0;i<r.n_blocks;i++){
      r.blocks[i].cmpgt(f);
    }
    return r;
  }

  template<int N, class T>
  SpMatrix<N, T> operator>=(const SpMatrix<N, T>& m, T f){
    SpMatrix<N, T> r(m);
    /*Just inspect all the blocks*/
    for(int i=0;i<r.n_blocks;i++){
      r.blocks[i].cmpge(f);
    }
    return r;
  }

  template<int N, class T>
  SpMatrix<N, T> operator<(T f, const SpMatrix<N, T>& m){
    return m>f;
  }

  template<int N, class T>
  SpMatrix<N, T> operator<=(T f, const SpMatrix<N, T>& m){
    return m>=f;
  }

  template<int N, class T>
  SpMatrix<N, T> operator>(T f, const SpMatrix<N, T>& m){
    return m<f;
  }

  template<int N, class T>
  SpMatrix<N, T> operator>=(T f, const SpMatrix<N, T>& m){
    return m<=f;
  }

  template<int N, class T>
  void SpMatrix<N, T>::printRow(int row)const{
    T mul = 200.0;
    T sum = 0;
    T sum2 = 0;
    int block_row = row/N;
    int local_row = row%N;

    for(int i=0;i<row_lengths[block_row];i++){
      int block_index = block_indices[block_row][i];
      int block_col   = col_indices[block_row][i] * N;

      for(int j=0;j<N;j++){
        printf("(%d, %d) = %20.20e, %20.20e\n", row, block_col + j, blocks[block_index].m[local_row*N + j], blocks[block_index].m[local_row*N + j]*mul);
        sum += blocks[block_index].m[local_row*N + j] * mul;
        sum2 += blocks[block_index].m[local_row*N + j];
      }
    }

    printf("sum = %10.10e\n", sum);
    printf("sum2 = %10.10e, %10.20e\n", sum2, sum2*200);
  }

  template<int N, class T>
  void SpMatrix<N, T>::analyse()const{
    int minrow = 100000;
    int maxrow = 0;
    int totalrow = 0;
    for(int i=0;i<height/N;i++){
      if(row_lengths[i] != 0){
        minrow = MIN(minrow, row_lengths[i]);
        maxrow = MAX(maxrow, row_lengths[i]);
        totalrow += row_lengths[i];
      }
    }

    printf("%d stored elements\n", n_elements);
    printf("%d minimal number of blocks\n", n_elements/(N*N));
    printf("%d allocated blocks\n", n_blocks);
    printf("Average block fill = %f\n", (float)n_elements/(float)n_blocks);

    printf("Shortest row = %d\n", minrow);
    printf("Longest  row = %d\n", maxrow);
    printf("Average  row = %f\n", (float)totalrow/(float)(height/N));
  }

  template<int N, class T>
  int SpMatrix<N, T>::getNElements()const{
    return n_elements;
  }

  template<int N, class T>
  int SpMatrix<N, T>::getNBlocks()const{
    return n_blocks;
  }

  template<int N, class T>
  int SpMatrix<N, T>::getShortestRow()const{
    int minrow = 100000;
    int maxrow = 0;
    int totalrow = 0;
    for(int i=0;i<height/N;i++){
      minrow = MIN(minrow, row_lengths[i]);
      maxrow = MAX(maxrow, row_lengths[i]);
      totalrow += row_lengths[i];
    }
    return minrow;
  }

  template<int N, class T>
  int SpMatrix<N, T>::getLongestRow()const{
    int minrow = 100000;
    int maxrow = 0;
    int totalrow = 0;
    for(int i=0;i<height/N;i++){
      minrow = MIN(minrow, row_lengths[i]);
      maxrow = MAX(maxrow, row_lengths[i]);
      totalrow += row_lengths[i];
    }
    return maxrow;
  }

  template<int N, class T>
  float SpMatrix<N, T>::getAverageRow()const{
    int minrow = 100000;
    int maxrow = 0;
    int totalrow = 0;
    for(int i=0;i<height/N;i++){
      minrow = MIN(minrow, row_lengths[i]);
      maxrow = MAX(maxrow, row_lengths[i]);
      totalrow += row_lengths[i];
    }
    return (float)totalrow/(float)(height/N);
  }

  template<int N, class T>
  float SpMatrix<N, T>::getAverageBlockFill()const{
    int minrow = 100000;
    int maxrow = 0;
    int totalrow = 0;
    for(int i=0;i<height/N;i++){
      minrow = MIN(minrow, row_lengths[i]);
      maxrow = MAX(maxrow, row_lengths[i]);
      totalrow += row_lengths[i];
    }
    return (float)n_elements/(float)n_blocks;
  }

  /*Protected functions*/

  template<int N, class T>
  void SpMatrix<N, T>::grow_row(int row){
    int* tmp_blk_index = block_indices[row];
    int* tmp_col_index = col_indices[row];

    col_indices[row]   = new int[allocated_length[row]*2];
    block_indices[row] = new int[allocated_length[row]*2];

    memcpy(col_indices[row], tmp_col_index,
           sizeof(int)*(uint)allocated_length[row]);

    memcpy(block_indices[row], tmp_blk_index,
           sizeof(int)*(uint)allocated_length[row]);

    allocated_length[row] *=2;

    delete [] tmp_blk_index;
    delete [] tmp_col_index;
  }

  template<int N, class T>
  void SpMatrix<N, T>::grow_blocks(){
    SpMatrixBlock<N, T>* tmp = blocks;

    alignedMalloc((void**)&blocks, 16,
                  (uint)n_allocated_blocks*2*sizeof(SpMatrixBlock<N, T>));

    memcpy(blocks, tmp, sizeof(SpMatrixBlock<N, T>)*(uint)n_allocated_blocks);
    n_allocated_blocks*=2;

    alignedFree(tmp);
  }

  template<int N, class T>
  std::ostream& operator<<(std::ostream& stream , const SpMatrix<N, T>& m){
    std::ios_base::fmtflags origflags = stream.flags();
    stream.setf(std::ios_base::scientific);
    //stream << std::setprecision(10);
    for(int row=0;row<m.origHeight;row++){
      for(int col=0;col<m.origWidth;col++){
        int block_index;
        int block_row;
        int block_col;
        bool found = false;

#if 1
        //std::map<int, int>::iterator it = m.block_map[row/N].find(col/N);

        //if(it != m.block_map[row/N].end()){
        int idx = m.block_map[row/N].findIndex(col/N);
        if(idx != Tree<int>::undefinedIndex){
          found = true;
          block_index = m.block_indices[row/N][idx];//[it->second];
          //block_index = m.block_indices[row/N][it->second];
          block_row = row%N;
          block_col = col%N;
        }
#else
        for(int k=0;k<m.row_lengths[row/N];k++){
          if(m.col_indices[row/N][k] == col/N){
            block_index = m.block_indices[row/N][k];
            block_row = row%N;
            block_col = col%N;
            found = true;
            break;
          }
        }
#endif

        if(found){
          stream << m.blocks[block_index].m[block_row*N+block_col] << "\t";
        }else{
          stream << 0.0f << "\t";
        }
      }
      stream<<std::endl;
    }
    stream.flags(origflags);
    return stream;
  }

  template<int N, class T>
  void SpMatrix<N, T>::reorderBlocks(){
    /*Order the blocks such that they are ordered in memory in the way
      the are accessed. This should improve the cache coherency*/

    /*First build a reverse index such that we know where each block
      is located. This information is needed to swap the blocks
      properly*/
    reverse_index_t* reverseIndices = new reverse_index_t[n_blocks];

    int blockIndex = 0;

    for(int i=0;i<height/N;i++){
      for(int j=0;j<row_lengths[i];j++){
        int block_index = block_indices[i][j];

        reverseIndices[block_index].blockRow = i;
        reverseIndices[block_index].blockCol = j;
      }
    }

    blockIndex = 0;

    /*Swap blocks*/

    for(int i=0;i<height/N;i++){
      for(int j=0;j<row_lengths[i];j++){
        int tmpBlockIndex = block_indices[i][j];

        if(blockIndex != tmpBlockIndex){
          /*Swap blocks and adapt indices*/

          reverse_index_t rev = reverseIndices[blockIndex];

          SpMatrixBlock<N, T> tmpBlock1 = blocks[tmpBlockIndex];
          SpMatrixBlock<N, T> tmpBlock2 = blocks[blockIndex];
          cgfassert(rev.blockRow != -1);
          cgfassert(rev.blockCol != -1);

          block_indices[i][j]    = blockIndex;
          blocks[blockIndex]     = tmpBlock1;
          blocks[tmpBlockIndex]  = tmpBlock2;
          block_indices[rev.blockRow][rev.blockCol] = tmpBlockIndex;

          /*Update reverse pointers*/
          reverseIndices[blockIndex].blockRow = -1;
          reverseIndices[blockIndex].blockCol = -1;
          reverseIndices[tmpBlockIndex].blockRow = rev.blockRow;
          reverseIndices[tmpBlockIndex].blockCol = rev.blockCol;
        }
        blockIndex++;
      }
    }
    delete [] reverseIndices;

    /*Align indices*/
#if 0
    int** tmp_col_indices = new int*[height/N];
    int** tmp_blk_indices = new int*[height/N];
    int n_indices = 0;
    for(int i=0;i<height/N;i++){
      tmp_col_indices[i] = new int[row_lengths[i]];
      memcpy(tmp_col_indices[i], col_indices[i], sizeof(int)*row_lengths[i]);
      n_indices+=row_lengths[i];
    }

    for(int i=0;i<height/N;i++){
      tmp_blk_indices[i] = new int[row_lengths[i]];
      memcpy(tmp_blk_indices[i], block_indices[i], sizeof(int)*row_lengths[i]);
    }

    for(int i=0;i<height/N;i++){
      delete[] col_indices[i];
      delete[] block_indices[i];
    }
    delete[] col_indices;
    delete[] block_indices;

    col_indices = tmp_col_indices;
    block_indices = tmp_blk_indices;
#endif

    /*Compact indices*/
    if(comp_col_indices)
      delete [] comp_col_indices;

    comp_col_indices = new int[n_blocks];

    int idx = 0;
    for(int i=0;i<height/N;i++){
      for(int j=0;j<row_lengths[i];j++, idx++){
        comp_col_indices[idx] = col_indices[i][j] * N;
      }
    }
  }

  template<int N, class T>
  void spmv(Vector<T>& r, const SpMatrix<N, T>& m, const Vector<T>& v){
    MatrixRange range;

    range.startRow = 0;
    range.endRow   = m.width/N;
    range.range    = m.width/N;

    spmv_partial(r, m, v, range);
  }

  /*Transposed multiplication*/
  template<int N, class T>
  void spmv_t(Vector<T>& r, const SpMatrix<N, T>& m, const Vector<T>& v){
    cgfassert(m.getHeight() == v.getSize());
    cgfassert(r.getSize()  == m.getWidth());
    cgfassert(m.finalized  == true);

    /*Clear tmp_data*/
    for(int i=0;i<m.width/N;i++){
      m.tBlocks[i].clear();
    }

    int idx = 0;
    /*Number of block rows*/
    int n_rows = m.height/N;

    const T* data = v.data;

    for(int i=0;i<n_rows;i++){
      int row_length = m.row_lengths[i];
      for(int j=0;j<row_length;j++, idx++){
        int blockIndex = m.comp_col_indices[idx]/N;
        m.tBlocks[blockIndex].vectorMulAddTranspose(&(m.blocks[idx]),
                                                    &(data[i*N]));
      }
    }

    /*Column reduce*/
    for(int i=0;i<m.width/N;i++){
      m.tBlocks[i].columnSumReduce();
    }

    for(int i=0;i<m.width/N;i++){
      for(int j=0;j<N;j++){
        r.data[i*N+j] = m.tBlocks[i].m[N*(N-1) + j];
      }
    }
  }

  int computeSteps(int size);
#if 0
  {
    /*Find a value, smaller than size, which is a power of 2*/
    int steps = size/2;
    int pwr = 1;
    while(steps >1){
      steps/=2;
      pwr++;
    }
    return 1<<pwr;
  }
#endif

  /*Performs a partial sparse matrix vector
    multiplication. I.e. performs the multiplication such that the
    result vector of this operation is a subvector of the actual
    solution. This function enables the parallel multiplication of a
    sparse vector with a vector. At the end the resulting subvectors
    are combined to the actual result*/

  /*The result vector is divided in n separate sub vectors and this
    function computes the result denoted with index*/
  template<int N, class T>
  void spmv_partial(Vector<T>& r, const SpMatrix<N, T>& m,
                    const Vector<T>& v, const MatrixRange mr){
    cgfassert(m.getWidth() == v.getSize());
    cgfassert(r.getSize() == m.getHeight());
    cgfassert(m.finalized == true);

    if(mr.range == 0)
      return;
    if(mr.startRow == mr.endRow){
      return;
    }

    int idx = m.block_indices[mr.startRow][0];

    const T* data = v.data;

    SpMatrixBlock<N, T> tmpBlock;

    for(int i=mr.startRow;i<mr.endRow;i++){
      int row_length = m.row_lengths[i];
      if(row_length > 0){
        tmpBlock.clear();

        for(int j=0;j<row_length;j++, idx++){
          tmpBlock.vectorMulAdd(&(m.blocks[idx]),
                                &(data[m.comp_col_indices[idx]]));
        }

        tmpBlock.rowSumReduce();
        for(int j=0;j<N;j++){
          r.data[i*N + j] = tmpBlock.m[j*N];
        }
      }
    }
  }

#if 0
  template<int N, class T>
  void spmv_parallel(Vector<T>& r, const SpMatrix<N, T>& m, const Vector<T>& v){
    cgfassert(m.finalized == true);
    m.task->setVectors(&r, &v);
    m.pool->assignTask(m.task);
  }

  template<int N, class T>
  void SpMatrix<N, T>::finalizeParallel(ThreadPool* _pool){
    pool = _pool;
    if(task)
      delete task;

    finalize();

    task = new ParallelSPMVTask<N, T>(pool, this);
  }
#endif

  template<int N, class T>
  void SpMatrix<N, T>::computeBlockDistribution(MatrixRange* mRange,
                                                VectorRange* vRange,
                                                int* n_blks,
                                                int n_segments)const{
    cgfassert(blocks != 0);
    /*Compute distribution of the blocks over the segments. The
      corresponding vector dimension per segment is a multiple of 16,
      because the partial vector and matrix functions operate with
      chunks of 16 floats*/

	int n_combined_rows = 16/N;
    int blocks_per_segment = (int)ceil((float)n_blocks/(float)n_segments);

    int currentBlockRow = 0;
    int currentRow = 0;

    int totalBlocks = 0;
    for(int i=0;i<n_segments;i++){
      mRange[i].startRow = currentRow;
      currentBlockRow = 0;
      while(currentBlockRow < blocks_per_segment){
        int stop = 0;
        int k = 0;
        for(;k<n_combined_rows;k++){
          if(currentRow + k < height/N){
            currentBlockRow += row_lengths[currentRow + k];
          }else{
            stop = 1;
            break;
          }
        }
        currentRow += k;
        if(stop){
          break;
        }
      }

      mRange[i].endRow = currentRow;
      n_blks[i] = currentBlockRow;
      totalBlocks += currentBlockRow;
    }
    cgfassert(n_blocks == totalBlocks);

    for(int i=0;i<n_segments;i++){
      vRange[i].startBlock = mRange[i].startRow * N;
      vRange[i].endBlock   = mRange[i].endRow * N;
      vRange[i].range = vRange[i].endBlock - vRange[i].startBlock;
      mRange[i].range = mRange[i].endRow - mRange[i].startRow;
    }
  }

  template<int N, class T>
  void multiplyJacobi(Vector<T>& r, const SpMatrix<N, T>& m,
                      const Vector<T>& v){
    cgfassert(m.getWidth() == v.getSize());
    cgfassert(r.getSize()  == m.getHeight());
    cgfassert(m.finalized);

    int idx = 0;
    int n_rows = m.height/N;

    const T* data = v.data;

    SpMatrixBlock<N, T> tmpBlock;

    for(int i=0;i<n_rows;i++){
      int row_length = m.row_lengths[i];

      if(row_length > 0){
        tmpBlock.clear();

        int diagBlockIndex = -1;

        for(int j=0;j<row_length;j++, idx++){
          int col_index = m.comp_col_indices[idx]/N;
          if(col_index == i){
            //Diagonal block
            tmpBlock.vectorMulAddJacobi(&(m.blocks[idx]),
                                        &(data[col_index*N]));
            diagBlockIndex = idx;
          }else{
            //Off-diagonal block
            tmpBlock.vectorMulAdd(&(m.blocks[idx]),
                                  &(data[col_index*N]));
          }
        }

        cgfassert(diagBlockIndex != -1);

        tmpBlock.rowSumReduce();

        /*Store and divide by diagonal*/
        for(int j=0;j<N;j++){
          r.data[i*N + j] = tmpBlock.m[j*N] / m.blocks[diagBlockIndex].m[j*N+j];
        }
      }
    }
  }

  template<int N, class T>
  void multiplyGaussSeidel(Vector<T>& r, const SpMatrix<N, T>& m,
                           const Vector<T>& v, const Vector<T>* b){
    cgfassert(m.getWidth() == v.getSize());
    cgfassert(r.getSize()  == m.getHeight());
    cgfassert(m.finalized);

    int idx = 0;
    int n_rows = m.height/N;

    const T* data = v.data;
    T* rdata = r.data;

    SpMatrixBlock<N, T> tmpBlock;

    for(int i=0;i<n_rows;i++){
      int row_length = m.row_lengths[i];
      if(row_length > 0){
        tmpBlock.clear();

        int diagBlockIndex = -1;

        for(int j=0;j<row_length;j++, idx++){
          int col_index = m.comp_col_indices[idx] / N;
          if(col_index == i){
            //Diagonal block
            /*Skip diagonal block for now, and is performed as last*/
            diagBlockIndex = idx;
          }else if(col_index < i){
            //Off-diagonal block, use vector r containing updated
            //values
            tmpBlock.vectorMulAdd(&(m.blocks[idx]),
                                  &(rdata[col_index*N]));
          }else{
            //Off-diagonal block, use vector v
            tmpBlock.vectorMulAdd(&(m.blocks[idx]),
                                  &(data[col_index*N]));

          }
        }

        cgfassert(diagBlockIndex != -1);

        tmpBlock.rowSumReduce();

        /*Store*/
        for(int j=0;j<N;j++){
          r.data[i*N + j] = -tmpBlock.m[j*N];

          if(b){
            r.data[i*N + j] += b->data[i*N + j];
          }
        }

        /*Perform multiplication of diagonal block*/
        int col_index = m.comp_col_indices[diagBlockIndex] / N;
        for(int j=0;j<N;j++){
          T diagElement = 0;

          for(int k=0;k<N;k++){
            if(j==k){
              /*Diagonal element*/
              diagElement = m.blocks[diagBlockIndex].m[j*N+k];
            }else if(j<k){
              /*Use vector v*/
              r.data[i*N+j] -= m.blocks[diagBlockIndex].m[j*N+k] *
                data[col_index*N + k];
            }else{
              /*Use updated k*/
              r.data[i*N+j] -= m.blocks[diagBlockIndex].m[j*N+k] *
                rdata[col_index*N + k];
            }
          }

          r.data[i*N + j] /= diagElement;
        }
      }
    }
  }

#if 0
  template SpMatrix<1, float>* SpMatrix<1, float>::reorderRCM()const;
  template SpMatrix<2, float>* SpMatrix<2, float>::reorderRCM()const;
  template SpMatrix<4, float>* SpMatrix<4, float>::reorderRCM()const;
  template SpMatrix<8, float>* SpMatrix<8, float>::reorderRCM()const;

  template SpMatrix<1, double>* SpMatrix<1, double>::reorderRCM()const;
  template SpMatrix<2, double>* SpMatrix<2, double>::reorderRCM()const;
  template SpMatrix<4, double>* SpMatrix<4, double>::reorderRCM()const;
  template SpMatrix<8, double>* SpMatrix<8, double>::reorderRCM()const;
#endif
  /*Instantiate templates explicitly*/
  template class SpMatrix<1, float>;
  template class SpMatrix<2, float>;
  template class SpMatrix<4, float>;
  template class SpMatrix<8, float>;
  //template class SpMatrix<16, float>;

  template class SpRowProxy<1, float>;
  template class SpRowProxy<2, float>;
  template class SpRowProxy<4, float>;
  template class SpRowProxy<8, float>;
  //template class SpRowProxy<16, float>;

  template class SpMatrix<1, double>;
  template class SpMatrix<2, double>;
  template class SpMatrix<4, double>;
  template class SpMatrix<8, double>;
  //template class SpMatrix<16, float>;

  template class SpRowProxy<1, double>;
  template class SpRowProxy<2, double>;
  template class SpRowProxy<4, double>;
  template class SpRowProxy<8, double>;
  //template class SpRowProxy<16, float>;

  template void spmv(Vector<float>& r, const SpMatrix<1, float>& m,
                     const Vector<float>& v);
  template void spmv(Vector<float>& r, const SpMatrix<2, float>& m,
                     const Vector<float>& v);
  template void spmv(Vector<float>& r, const SpMatrix<4, float>& m,
                     const Vector<float>& v);
  template void spmv(Vector<float>& r, const SpMatrix<8, float>& m,
                     const Vector<float>& v);
  //template void spmv(Vector& r, const SpMatrix<16, float>& m,
  //	     const Vector& v);

  template void spmv_t(Vector<float>& r, const SpMatrix<1, float>& m,
                       const Vector<float>& v);
  template void spmv_t(Vector<float>& r, const SpMatrix<2, float>& m,
                       const Vector<float>& v);
  template void spmv_t(Vector<float>& r, const SpMatrix<4, float>& m,
                       const Vector<float>& v);
  template void spmv_t(Vector<float>& r, const SpMatrix<8, float>& m,
                       const Vector<float>& v);
  //template void spmv_t(Vector& r, const SpMatrix<16, float>& m,
  //	     const Vector& v);

  template void spmv_partial(Vector<float>& r, const SpMatrix<1, float>& m,
                             const Vector<float>& v, const MatrixRange mr);
  template void spmv_partial(Vector<float>& r, const SpMatrix<2, float>& m,
                             const Vector<float>& v, const MatrixRange mr);
  template void spmv_partial(Vector<float>& r, const SpMatrix<4, float>& m,
                             const Vector<float>& v, const MatrixRange mr);
  template void spmv_partial(Vector<float>& r, const SpMatrix<8, float>& m,
                             const Vector<float>& v, const MatrixRange mr);
  //template void spmv_partial(Vector& r, const SpMatrix<16, float>& m,
  //			     const Vector& v, const MatrixRange mr);

  template void spmv(Vector<double>& r, const SpMatrix<1, double>& m,
                     const Vector<double>& v);
  template void spmv(Vector<double>& r, const SpMatrix<2, double>& m,
                     const Vector<double>& v);
  template void spmv(Vector<double>& r, const SpMatrix<4, double>& m,
                     const Vector<double>& v);
  template void spmv(Vector<double>& r, const SpMatrix<8, double>& m,
                     const Vector<double>& v);
  //template void spmv(Vector& r, const SpMatrix<16, float>& m,
  //	     const Vector& v);


  template void spmv_t(Vector<double>& r, const SpMatrix<1, double>& m,
                       const Vector<double>& v);
  template void spmv_t(Vector<double>& r, const SpMatrix<2, double>& m,
                       const Vector<double>& v);
  template void spmv_t(Vector<double>& r, const SpMatrix<4, double>& m,
                       const Vector<double>& v);
  template void spmv_t(Vector<double>& r, const SpMatrix<8, double>& m,
                       const Vector<double>& v);
  //template void spmv_t(Vector& r, const SpMatrix<16, float>& m,
  //	     const Vector& v);

  template void spmv_partial(Vector<double>& r, const SpMatrix<1, double>& m,
                             const Vector<double>& v, const MatrixRange mr);
  template void spmv_partial(Vector<double>& r, const SpMatrix<2, double>& m,
                             const Vector<double>& v, const MatrixRange mr);
  template void spmv_partial(Vector<double>& r, const SpMatrix<4, double>& m,
                             const Vector<double>& v, const MatrixRange mr);
  template void spmv_partial(Vector<double>& r, const SpMatrix<8, double>& m,
                             const Vector<double>& v, const MatrixRange mr);
  //template void spmv_partial(Vector& r, const SpMatrix<16, float>& m,
  //			     const Vector& v, const MatrixRange mr);

  template void multiplyJacobi(Vector<float>& r, const SpMatrix<1, float>& m,
                               const Vector<float>& v);
  template void multiplyJacobi(Vector<float>& r, const SpMatrix<2, float>& m,
                               const Vector<float>& v);
  template void multiplyJacobi(Vector<float>& r, const SpMatrix<4, float>& m,
                               const Vector<float>& v);
  template void multiplyJacobi(Vector<float>& r, const SpMatrix<8, float>& m,
                               const Vector<float>& v);

  template void multiplyJacobi(Vector<double>& r, const SpMatrix<1, double>& m,
                               const Vector<double>& v);
  template void multiplyJacobi(Vector<double>& r, const SpMatrix<2, double>& m,
                               const Vector<double>& v);
  template void multiplyJacobi(Vector<double>& r, const SpMatrix<4, double>& m,
                               const Vector<double>& v);
  template void multiplyJacobi(Vector<double>& r, const SpMatrix<8, double>& m,
                               const Vector<double>& v);

  template void multiplyGaussSeidel(Vector<float>& r,
                                    const SpMatrix<1, float>& m,
                                    const Vector<float>& v,
                                    const Vector<float>* b);
  template void multiplyGaussSeidel(Vector<float>& r,
                                    const SpMatrix<2, float>& m,
                                    const Vector<float>& v,
                                    const Vector<float>* b);
  template void multiplyGaussSeidel(Vector<float>& r,
                                    const SpMatrix<4, float>& m,
                                    const Vector<float>& v,
                                    const Vector<float>* b);
  template void multiplyGaussSeidel(Vector<float>& r,
                                    const SpMatrix<8, float>& m,
                                    const Vector<float>& v,
                                    const Vector<float>* b);

  template void multiplyGaussSeidel(Vector<double>& r,
                                    const SpMatrix<1, double>& m,
                                    const Vector<double>& v,
                                    const Vector<double>* b);
  template void multiplyGaussSeidel(Vector<double>& r,
                                    const SpMatrix<2, double>& m,
                                    const Vector<double>& v,
                                    const Vector<double>* b);
  template void multiplyGaussSeidel(Vector<double>& r,
                                    const SpMatrix<4, double>& m,
                                    const Vector<double>& v,
                                    const Vector<double>* b);
  template void multiplyGaussSeidel(Vector<double>& r,
                                    const SpMatrix<8, double>& m,
                                    const Vector<double>& v,
                                    const Vector<double>* b);

  template std::ostream& operator<<(std::ostream& stream ,
                                    const SpMatrix<1, float>& m);
  template std::ostream& operator<<(std::ostream& stream ,
                                    const SpMatrix<2, float>& m);
  template std::ostream& operator<<(std::ostream& stream ,
                                    const SpMatrix<4, float>& m);
  template std::ostream& operator<<(std::ostream& stream ,
                                    const SpMatrix<8, float>& m);
  //template std::ostream& operator<<(std::ostream& stream ,
  //				    const SpMatrix<16, float>& m);

  template std::ostream& operator<<(std::ostream& stream ,
                                    const SpMatrix<1, double>& m);
  template std::ostream& operator<<(std::ostream& stream ,
                                    const SpMatrix<2, double>& m);
  template std::ostream& operator<<(std::ostream& stream ,
                                    const SpMatrix<4, double>& m);
  template std::ostream& operator<<(std::ostream& stream ,
                                    const SpMatrix<8, double>& m);
  //template std::ostream& operator<<(std::ostream& stream ,
  //				    const SpMatrix<16, double>& m);

  template SpMatrix<1, float> operator==(const SpMatrix<1, float>& m, float f);
  template SpMatrix<2, float> operator==(const SpMatrix<2, float>& m, float f);
  template SpMatrix<4, float> operator==(const SpMatrix<4, float>& m, float f);
  template SpMatrix<8, float> operator==(const SpMatrix<8, float>& m, float f);
  //template SpMatrix<16, float> operator==(const SpMatrix<16, float>& m, float f);

  template SpMatrix<1, float> operator!=(const SpMatrix<1, float>& m, float f);
  template SpMatrix<2, float> operator!=(const SpMatrix<2, float>& m, float f);
  template SpMatrix<4, float> operator!=(const SpMatrix<4, float>& m, float f);
  template SpMatrix<8, float> operator!=(const SpMatrix<8, float>& m, float f);
  //template SpMatrix<16, float> operator!=(const SpMatrix<16, float>& m, float f);

  template SpMatrix<1, float> operator==(float f, const SpMatrix<1, float>& m);
  template SpMatrix<2, float> operator==(float f, const SpMatrix<2, float>& m);
  template SpMatrix<4, float> operator==(float f, const SpMatrix<4, float>& m);
  template SpMatrix<8, float> operator==(float f, const SpMatrix<8, float>& m);
  //template SpMatrix<16, float> operator==(float f, const SpMatrix<16, float>& m);

  template SpMatrix<1, float> operator!=(float f, const SpMatrix<1, float>& m);
  template SpMatrix<2, float> operator!=(float f, const SpMatrix<2, float>& m);
  template SpMatrix<4, float> operator!=(float f, const SpMatrix<4, float>& m);
  template SpMatrix<8, float> operator!=(float f, const SpMatrix<8, float>& m);
  //template SpMatrix<16, float> operator!=(float f, const SpMatrix<16, float>& m);

  template SpMatrix<1, float> operator<(const SpMatrix<1, float>& m, float f);
  template SpMatrix<2, float> operator<(const SpMatrix<2, float>& m, float f);
  template SpMatrix<4, float> operator<(const SpMatrix<4, float>& m, float f);
  template SpMatrix<8, float> operator<(const SpMatrix<8, float>& m, float f);
  //template SpMatrix<16, float> operator<(const SpMatrix<16, float>& m, float f);

  template SpMatrix<1, float> operator<=(const SpMatrix<1, float>& m, float f);
  template SpMatrix<2, float> operator<=(const SpMatrix<2, float>& m, float f);
  template SpMatrix<4, float> operator<=(const SpMatrix<4, float>& m, float f);
  template SpMatrix<8, float> operator<=(const SpMatrix<8, float>& m, float f);
  //template SpMatrix<16, float> operator<=(const SpMatrix<16, float>& m, float f);

  template SpMatrix<1, float> operator>(const SpMatrix<1, float>& m, float f);
  template SpMatrix<2, float> operator>(const SpMatrix<2, float>& m, float f);
  template SpMatrix<4, float> operator>(const SpMatrix<4, float>& m, float f);
  template SpMatrix<8, float> operator>(const SpMatrix<8, float>& m, float f);
  //template SpMatrix<16, float> operator>(const SpMatrix<16, float>& m, float f);

  template SpMatrix<1, float> operator>=(const SpMatrix<1, float>& m, float f);
  template SpMatrix<2, float> operator>=(const SpMatrix<2, float>& m, float f);
  template SpMatrix<4, float> operator>=(const SpMatrix<4, float>& m, float f);
  template SpMatrix<8, float> operator>=(const SpMatrix<8, float>& m, float f);
  //template SpMatrix<16, float> operator>=(const SpMatrix<16, float>& m, float f);

  template SpMatrix<1, float> operator<(float f, const SpMatrix<1, float>& m);
  template SpMatrix<2, float> operator<(float f, const SpMatrix<2, float>& m);
  template SpMatrix<4, float> operator<(float f, const SpMatrix<4, float>& m);
  template SpMatrix<8, float> operator<(float f, const SpMatrix<8, float>& m);
  //template SpMatrix<16, float> operator<(float f, const SpMatrix<16, float>& m);

  template SpMatrix<1, float> operator<=(float f, const SpMatrix<1, float>& m);
  template SpMatrix<2, float> operator<=(float f, const SpMatrix<2, float>& m);
  template SpMatrix<4, float> operator<=(float f, const SpMatrix<4, float>& m);
  template SpMatrix<8, float> operator<=(float f, const SpMatrix<8, float>& m);
  //template SpMatrix<16, float> operator<=(float f, const SpMatrix<16, float>& m);

  template SpMatrix<1, float> operator>(float f, const SpMatrix<1, float>& m);
  template SpMatrix<2, float> operator>(float f, const SpMatrix<2, float>& m);
  template SpMatrix<4, float> operator>(float f, const SpMatrix<4, float>& m);
  template SpMatrix<8, float> operator>(float f, const SpMatrix<8, float>& m);
  //template SpMatrix<16, float> operator>(float f, const SpMatrix<16, float>& m);

  template SpMatrix<1, float> operator>=(float f, const SpMatrix<1, float>& m);
  template SpMatrix<2, float> operator>=(float f, const SpMatrix<2, float>& m);
  template SpMatrix<4, float> operator>=(float f, const SpMatrix<4, float>& m);
  template SpMatrix<8, float> operator>=(float f, const SpMatrix<8, float>& m);
  //template SpMatrix<16, float> operator>=(float f, const SpMatrix<16, float>& m);

  /**/

  template SpMatrix<1, double> operator==(const SpMatrix<1, double>& m, double f);
  template SpMatrix<2, double> operator==(const SpMatrix<2, double>& m, double f);
  template SpMatrix<4, double> operator==(const SpMatrix<4, double>& m, double f);
  template SpMatrix<8, double> operator==(const SpMatrix<8, double>& m, double f);
  //template SpMatrix<16, double> operator==(const SpMatrix<16, double>& m, double f);

  template SpMatrix<1, double> operator!=(const SpMatrix<1, double>& m, double f);
  template SpMatrix<2, double> operator!=(const SpMatrix<2, double>& m, double f);
  template SpMatrix<4, double> operator!=(const SpMatrix<4, double>& m, double f);
  template SpMatrix<8, double> operator!=(const SpMatrix<8, double>& m, double f);
  //template SpMatrix<16, double> operator!=(const SpMatrix<16, double>& m, double f);

  template SpMatrix<1, double> operator==(double f, const SpMatrix<1, double>& m);
  template SpMatrix<2, double> operator==(double f, const SpMatrix<2, double>& m);
  template SpMatrix<4, double> operator==(double f, const SpMatrix<4, double>& m);
  template SpMatrix<8, double> operator==(double f, const SpMatrix<8, double>& m);
  //template SpMatrix<16, double> operator==(double f, const SpMatrix<16, double>& m);

  template SpMatrix<1, double> operator!=(double f, const SpMatrix<1, double>& m);
  template SpMatrix<2, double> operator!=(double f, const SpMatrix<2, double>& m);
  template SpMatrix<4, double> operator!=(double f, const SpMatrix<4, double>& m);
  template SpMatrix<8, double> operator!=(double f, const SpMatrix<8, double>& m);
  //template SpMatrix<16, double> operator!=(double f, const SpMatrix<16, double>& m);

  template SpMatrix<1, double> operator<(const SpMatrix<1, double>& m, double f);
  template SpMatrix<2, double> operator<(const SpMatrix<2, double>& m, double f);
  template SpMatrix<4, double> operator<(const SpMatrix<4, double>& m, double f);
  template SpMatrix<8, double> operator<(const SpMatrix<8, double>& m, double f);
  //template SpMatrix<16, double> operator<(const SpMatrix<16, double>& m, double f);

  template SpMatrix<1, double> operator<=(const SpMatrix<1, double>& m, double f);
  template SpMatrix<2, double> operator<=(const SpMatrix<2, double>& m, double f);
  template SpMatrix<4, double> operator<=(const SpMatrix<4, double>& m, double f);
  template SpMatrix<8, double> operator<=(const SpMatrix<8, double>& m, double f);
  //template SpMatrix<16, double> operator<=(const SpMatrix<16, double>& m, double f);

  template SpMatrix<1, double> operator>(const SpMatrix<1, double>& m, double f);
  template SpMatrix<2, double> operator>(const SpMatrix<2, double>& m, double f);
  template SpMatrix<4, double> operator>(const SpMatrix<4, double>& m, double f);
  template SpMatrix<8, double> operator>(const SpMatrix<8, double>& m, double f);
  //template SpMatrix<16, double> operator>(const SpMatrix<16, double>& m, double f);

  template SpMatrix<1, double> operator>=(const SpMatrix<1, double>& m, double f);
  template SpMatrix<2, double> operator>=(const SpMatrix<2, double>& m, double f);
  template SpMatrix<4, double> operator>=(const SpMatrix<4, double>& m, double f);
  template SpMatrix<8, double> operator>=(const SpMatrix<8, double>& m, double f);
  //template SpMatrix<16, double> operator>=(const SpMatrix<16, double>& m, double f);

  template SpMatrix<1, double> operator<(double f, const SpMatrix<1, double>& m);
  template SpMatrix<2, double> operator<(double f, const SpMatrix<2, double>& m);
  template SpMatrix<4, double> operator<(double f, const SpMatrix<4, double>& m);
  template SpMatrix<8, double> operator<(double f, const SpMatrix<8, double>& m);
  //template SpMatrix<16, double> operator<(double f, const SpMatrix<16, double>& m);

  template SpMatrix<1, double> operator<=(double f, const SpMatrix<1, double>& m);
  template SpMatrix<2, double> operator<=(double f, const SpMatrix<2, double>& m);
  template SpMatrix<4, double> operator<=(double f, const SpMatrix<4, double>& m);
  template SpMatrix<8, double> operator<=(double f, const SpMatrix<8, double>& m);
  //template SpMatrix<16, double> operator<=(double f, const SpMatrix<16, double>& m);

  template SpMatrix<1, double> operator>(double f, const SpMatrix<1, double>& m);
  template SpMatrix<2, double> operator>(double f, const SpMatrix<2, double>& m);
  template SpMatrix<4, double> operator>(double f, const SpMatrix<4, double>& m);
  template SpMatrix<8, double> operator>(double f, const SpMatrix<8, double>& m);
  //template SpMatrix<16, double> operator>(double f, const SpMatrix<16, double>& m);

  template SpMatrix<1, double> operator>=(double f, const SpMatrix<1, double>& m);
  template SpMatrix<2, double> operator>=(double f, const SpMatrix<2, double>& m);
  template SpMatrix<4, double> operator>=(double f, const SpMatrix<4, double>& m);
  template SpMatrix<8, double> operator>=(double f, const SpMatrix<8, double>& m);
  //template SpMatrix<16, double> operator>=(double f, const SpMatrix<16, double>& m);
}
