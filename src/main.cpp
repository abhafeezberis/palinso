/* Copyright (C) 2012 by Mickeal Verschoor

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

#include <iostream>
#include <fstream>
#include "core/version.hpp"
#include "math/LinSolveCG.hpp"
#include "math/matrix_loaders.hpp"

using namespace CGF;


// ls -R -d -1 **/*/*.mtx
//ls -R -d -1 ../shared/matrices/**/*/*.mtx > files.txt
std::vector<std::string> matrixFiles;

template<int N, class T>
void large_cg_test(uint i){
  message("Blocksize = %d", N);
  char filename[255];

  sprintf(filename, "%s", matrixFiles[i].c_str());

  SpMatrix<N, T>* mat = 0;

  /*Load a matrix from file*/
  mat = load_matrix_market_exchange<N, T>(filename);
  if(mat == 0){
    warning("Failed loading %s", filename);
    return;
  }

  if(mat){
    Vector<T> vec(mat->getWidth());
    for(uint j=0;j<mat->getWidth();j++){
      if(j%2==0)
	vec[j] = 1;
      else
	vec[j] = 1;
    }

    mat->finalize();

    LinSolve<N, T>* solver = 0; 

    /*Enable the tests you like to perform.*/
#if 1
    message("Single threaded test");
    solver = new LinSolveCG<N, T>(mat->getWidth());
    solver->setMatrix(mat);
    solver->setb(&vec);
    solver->preSolve();
    solver->solve();

    delete solver;
#endif
#if 1
    message("Dual threaded test");
    solver = new LinSolveCGParallel<N, T>(mat->getWidth(), 2);
    solver->setMatrix(mat);
    solver->setb(&vec);
    solver->preSolve();
    solver->solve();
    
    delete solver;
#endif
#if 1    
    message("Quad threaded test");

    solver = new LinSolveCGParallel<N, T>(mat->getWidth(), 4);
    solver->setMatrix(mat);
    solver->setb(&vec);
    solver->preSolve();
    solver->solve();
    
    delete solver;
#endif
#if 1
    message("Oct threaded test");

    solver = new LinSolveCGParallel<N, T>(mat->getWidth(), 8);
    solver->setMatrix(mat);
    solver->setb(&vec);
    solver->preSolve();
    solver->solve();
    
    delete solver;
#endif
#if 1
    message("Single cuda test");
    solver = new LinSolveCGGPU<N, T>(mat->getWidth(), 1);

    solver->setMatrix(mat);
    solver->setb(&vec);
    solver->preSolve();
    solver->solve();
    
    delete solver;
#endif
#if 1
    message("Dual cuda test");
    solver = new LinSolveCGGPU<N, T>(mat->getWidth(), 2);

    solver->setMatrix(mat);
    solver->setb(&vec);
    solver->preSolve();
    solver->solve();
    
    delete solver;
#endif
    
    delete mat;
  }else{
    warning("File %s not found. Test skipped.", filename);
  }
}

int main(int argc, char** argv){
  try{
    CGF::init(0);
    version_print();
    
    /*Adapt such that all your *.mtx files are listed. Currently it
      searches for patterns like
      ./shared/matrices/bib1/case1/case1.mtx */
    
    system("ls -R -d -1 ./shared/matrices/**/*/*.mtx > files.txt");
    
    std::ifstream fileList("files.txt", std::ios::in);
    if(fileList.is_open()){
      std::string line;
      if(fileList.is_open()){
	while(getline(fileList,line)){
	  matrixFiles.push_back(line);
	}
      }
    }
    
    //Or specify your own
    //std::string line("../../../Download/matrix.mtx");
    //matrixFiles.push_back(line);

    fileList.close();
#if 1

    /*Conjugate Gradient benchmark routine*/
    for(uint i=0;i<matrixFiles.size();i++){
      message("Matrix = %s", matrixFiles[i].c_str());
      large_cg_test<1, float>(i);
      large_cg_test<2, float>(i);
      large_cg_test<4, float>(i);
      large_cg_test<8, float>(i);
    }
    
#if 1 /*Enable for double-precision */
    for(uint i=0;i<matrixFiles.size();i++){
      message("Matrix = %s", matrixFiles[i].c_str());
      large_cg_test<1, double>(i);
      large_cg_test<2, double>(i);
      large_cg_test<4, double>(i);
      large_cg_test<8, double>(i);
    }
#endif
    
    return 0;
#endif
    
    CGF::destroy();
  }catch(Exception& e){
    std::cerr << e.getError();
  }
  return 0;
}

