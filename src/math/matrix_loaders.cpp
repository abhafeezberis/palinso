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

#include "math/matrix_loaders.hpp"
#include "math/Vector.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <vector>

#define CHOP 32
#undef CHOP

#define MIRROR
#undef MIRROR

namespace CGF{
  template<int N, class T>
  SpMatrix<N, T>* load_matrix_market_exchange(const char* filename){
    int count = 0;
    int sym = 0;
    std::ifstream file(filename, std::ios::in);
    //std::ofstream file_out("matrix.m");
    cgfassert(N==1 || N==2 || N==4 || N==8 || N==16);
    SpMatrix<N, T>* m = 0;
    int bw_min = std::numeric_limits<int>::max();
    int bw_max = std::numeric_limits<int>::min();

    bool pattern = false;
    T patternValue = 1;

    bool array = false;
    int arrayIndex = 0;
    
    if(file.is_open()){
      std::string line;
      int lines = 0;
      int rows, cols;
      bool firstLine = false;

      if(file.is_open()){
        while(getline(file, line)){
          if(line[0] == '%'){
            /*Comment line*/
            //message("Comment string %s", line.c_str());
	    
            if(firstLine == false){
              /*First line contains matrix info.*/
	      
              std::string buf;
              std::stringstream ss(line);

              std::vector<std::string> tokens;

              while(ss >> buf){
                tokens.push_back(buf);

                //std::cout << buf << std::endl;
              }

              if((tokens.size() == 5) && 
                 (tokens[0] == "%%MatrixMarket") &&
                 (tokens[1] == "matrix")){
                if(tokens[2] == "coordinate"){
                  array = false;
                  //message("Coordinate matrix");
                }else if(tokens[2] == "array"){
                  array = true;
                  //message("Array matrix");
                }else{
                  warning("Invalid format specifier");
                  return 0;
                }

                /*Read symmetry*/
                if(tokens[4] == "general"){
                  sym = 0;
                  //message("general matrix");
                }else if(tokens[4] == "symmetric"){
                  sym = 1;
                  //message("symmetric matrix");
                }else{
                  warning("Invalid symmetry specifier");
                  return 0;
                }

                /*Read storage*/
                if(tokens[3] == "complex"){
                  warning("Complex matrices not supported");
                  return 0;
                }else if(tokens[3] == "real"){
                  //message("real matrix");
                }else if(tokens[3] == "integer"){
                  //message("integer matrix");
                }else if(tokens[3] == "pattern"){
                  //message("pattern matrix");
                  pattern = true;
                }else{
                  warning("Invalid storage specifier");
                  return 0;
                }
              }else{
                warning("Invalid MatrixMarket file");
                return 0;
              }
	      
              firstLine = true;
            }
            continue;
          }
          std::istringstream stream;
          if(lines == 0){
            /*Not read dimensions yet. Current line is no comment. ->
              Current line contains dimension info*/
            stream.str(line);
            stream >> rows;
            stream >> cols;
            stream >> lines;

            if(rows != cols){
              //warning("Non square matrix");
              //return 0;
            }

#ifdef CHOP
            if(rows > CHOP)
              rows = cols = CHOP;
#endif

            //message("rows = %d, cols = %d, lines = %d", rows, cols, lines);
            //message("%s", line.c_str());
#ifdef MIRROR
            m = new SpMatrix<N, T>(cols*2, rows*2);
#else
            m = new SpMatrix<N, T>(cols, rows);
#endif
            //file_out << "M = sparse("<<rows<<","<<cols<<");" << std::endl;
          }else{
            int col, row;
            T val = 0;
            stream.str(line);
            if(!array){
              /*Coordinate matrix*/
              stream >> row;
              stream >> col;
              if(!pattern){
                stream >> val;
              }else{
                val = patternValue;
              }
            }else{
              /*Array matrix, compute indices*/
              row = arrayIndex/cols;
              col = arrayIndex%cols;
              stream >> val;
              arrayIndex++;
            }
            count++;
            //val = 1;//CHOP * (row-1) + (col-1);
#ifdef CHOP
            if(row <= CHOP && col <= CHOP)
#endif
              m->set(row-1, col-1,val);
            if(col-1 > bw_max)
              bw_max = col-1;
            if(col-1 < bw_min)
              bw_min = col-1;
#ifdef MIRROR
            m->set(rows + row-1, cols + col-1,val);
            m->set(       row-1, cols + col-1,val);
            m->set(rows + row-1,        col-1,val);
#endif
            if(sym == 1){
              if(row != col){
                //val = 1;//CHOP * (col-1) + (row-1);
#ifdef CHOP
                if(row <= CHOP && col <= CHOP)
#endif
                  m->set(col-1, row-1,val);
                count++;
                if(row-1 > bw_max)
                  bw_max = row-1;
                if(row-1 < bw_min)
                  bw_min = row-1;
#ifdef MIRROR
                m->set(cols + col-1, rows + row-1,val);
                m->set(       col-1, rows + row-1,val);
                m->set(cols + col-1,        row-1,val);
#endif
              }
            }
          }
        }
      }
      //file_out.close();
      file.close();
      //m->setBandwidth(bw_max - bw_min);
    }
    //message("%d elements", count);
    return m;
  }

  template<class T>
  Vector<T>* load_matrix_market_exchange_vector(const char* filename){
    std::ifstream file(filename, std::ios::in);
    Vector<T>* v = 0;
    
    if(file.is_open()){
      std::string line;
      int lines = 0;
      int dim;
      if(file.is_open()){
        int index = 0;
        while(getline(file, line)){
          if(line[0] == '%'){
            /*Comment line*/
            //message("Comment string %s", line.c_str());
            continue;
          }
          std::istringstream stream;
          if(lines == 0){
            /*Not read dimensions yet. Current line is no comment. ->
              Current line contains dimension info*/
            stream.str(line);
            stream >> dim;
            lines = dim;
            //message("dim = %d", dim);
            v = new Vector<T>(dim);
          }else{
            T val = 0;
            stream.str(line);
            stream >> val;

            (*v)[index] = val;
            index++;
          }
        }
      }
      file.close();
    }
    return v;
  }

  template<int N, class T>
  void save_matrix_market_exchange(const char* filename, 
                                   const SpMatrix<N, T>* const mat){
    std::ofstream file(filename, std::ios::out);
    /*Write header.*/
    file << "%%MatrixMarket matrix coordinate real general" << std::endl;
    
    /*Count number of nonzeros*/
    int nnz = 0;

    for(int i=0;i<mat->height/N;i++){
      for(int j=0;j<mat->row_lengths[i];j++){
        int block_index = mat->block_indices[i][j];

        SpMatrixBlock<N, T>* block = &(mat->blocks[block_index]);
        for(int k=0;k<N*N;k++){
          T val = block->m[k];
	  
          if(val != 0.0){
            nnz++;
          }
        }
      }
    }

    file << mat->getWidth() << " " << mat->getHeight()
         << " " << nnz << std::endl; 
    
    for(int i=0;i<mat->height/N;i++){
      for(int j=0;j<mat->row_lengths[i];j++){
        int col_index = mat->col_indices[i][j];
        int block_index = mat->block_indices[i][j];

        SpMatrixBlock<N, T>* block = &(mat->blocks[block_index]);
        for(int k=0;k<N*N;k++){
          int row = i*N + k/N;
          int col = col_index * N + k%N;
          T val = block->m[k];
	  
          if(val != 0.0){
            /*Matlab index starts at 1*/
            file << row+1 << " " << col+1 << " ";
	    
            std::ios_base::fmtflags ff;
            ff = file.flags();
            file.precision(13);
            file << std::scientific;
            file << val << std::endl;
            file.flags(ff);
          }
        }
      }
    }
    file.close();
  }

  template<int N, class T>
  bool save_matrix_matlab(const char* filename, const SpMatrix<N, T>* const mat){
    std::ofstream file(filename, std::ios::out);
    
    for(int i=0;i<mat->height/N;i++){
      for(int j=0;j<mat->row_lengths[i];j++){
        int col_index = mat->col_indices[i][j];
        int block_index = mat->block_indices[i][j];

        SpMatrixBlock<N, T>* block = &(mat->blocks[block_index]);
        for(int k=0;k<N*N;k++){
          int row = i*N + k/N;
          int col = col_index * N + k%N;
          T val = block->m[k];
	  
          if(val != 0.0){
            /*Matlab index starts at 1*/
            file << row+1 << "\t" << col+1 << "\t" << val << std::endl;
          }
        }
      }
    }
    file.close();
    
    return true;
  }

  template<int N, class T>
  SpMatrix<N, T>* load_matrix_matlab(const char* filename){
    std::ifstream file(filename, std::ios::in);

    if(file.is_open()){
      std::string line;
      
      int dim;
      
      std::istringstream stream;

      getline(file, line);

      /*Read dim*/
      stream.str(line);
      stream >> dim;

      SpMatrix<N, T>* mat = new SpMatrix<N, T>(dim, dim);
      int row, col;
      double value;
      while(getline(file, line)){
        sscanf(line.c_str(), "%d, %d, %lf", &row, &col, &value);

        (*mat)[row][col] = (T)value;
      }
      return mat;
    }
    return 0;
  }

  /*Instantiate templates*/

  template SpMatrix<1, float>* load_matrix_market_exchange(const char* filename);
  template SpMatrix<2, float>* load_matrix_market_exchange(const char* filename);
  template SpMatrix<4, float>* load_matrix_market_exchange(const char* filename);
  template SpMatrix<8, float>* load_matrix_market_exchange(const char* filename);

  template void save_matrix_market_exchange(const char* filename, 
                                            const SpMatrix<1,float>*const mat);
  template void save_matrix_market_exchange(const char* filename, 
                                            const SpMatrix<2,float>*const mat);
  template void save_matrix_market_exchange(const char* filename, 
                                            const SpMatrix<4,float>*const mat);
  template void save_matrix_market_exchange(const char* filename, 
                                            const SpMatrix<8,float>*const mat);

  template bool save_matrix_matlab(const char* filename, 
                                   const SpMatrix<1, float>* const mat);

  template bool save_matrix_matlab(const char* filename, 
                                   const SpMatrix<2, float>* const mat);

  template bool save_matrix_matlab(const char* filename, 
                                   const SpMatrix<4, float>* const mat);

  template bool save_matrix_matlab(const char* filename, 
                                   const SpMatrix<8, float>* const mat);

  template SpMatrix<1, float>* load_matrix_matlab(const char* filename);
  template SpMatrix<2, float>* load_matrix_matlab(const char* filename);
  template SpMatrix<4, float>* load_matrix_matlab(const char* filename);
  template SpMatrix<8, float>* load_matrix_matlab(const char* filename);

  //template SpMatrix<16>* load_matrix_market_exchange(const char* filename,
  //					     uint sym);

  template SpMatrix<1, double>* load_matrix_market_exchange(const char* filename);
  template SpMatrix<2, double>* load_matrix_market_exchange(const char* filename);
  template SpMatrix<4, double>* load_matrix_market_exchange(const char* filename);
  template SpMatrix<8, double>* load_matrix_market_exchange(const char* filename);

  template void save_matrix_market_exchange(const char* filename, 
                                            const SpMatrix<1,double>*const mat);
  template void save_matrix_market_exchange(const char* filename, 
                                            const SpMatrix<2,double>*const mat);
  template void save_matrix_market_exchange(const char* filename, 
                                            const SpMatrix<4,double>*const mat);
  template void save_matrix_market_exchange(const char* filename, 
                                            const SpMatrix<8,double>*const mat);

  template bool save_matrix_matlab(const char* filename, 
                                   const SpMatrix<1, double>* const mat);

  template bool save_matrix_matlab(const char* filename, 
                                   const SpMatrix<2, double>* const mat);

  template bool save_matrix_matlab(const char* filename, 
                                   const SpMatrix<4, double>* const mat);

  template bool save_matrix_matlab(const char* filename, 
                                   const SpMatrix<8, double>* const mat);

  template SpMatrix<1, double>* load_matrix_matlab(const char* filename);
  template SpMatrix<2, double>* load_matrix_matlab(const char* filename);
  template SpMatrix<4, double>* load_matrix_matlab(const char* filename);
  template SpMatrix<8, double>* load_matrix_matlab(const char* filename);

  //template SpMatrix<16>* load_matrix_market_exchange(const char* filename,
  //					     uint sym);
}
