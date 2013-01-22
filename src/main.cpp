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
#include <iostream>
#include <fstream>
#include <sstream>
#include "util/cuda_util.hpp"
#include <math.h>
#include <unistd.h>
#include "math/Vector.hpp"
#include "core/ThreadPool.hpp"
#include "core/Timer.hpp"
#include "core/Exception.hpp"
#include "math/matrix_loaders.hpp"
#include "math/LinSolve.hpp"
#include "math/LinSolveCG.hpp"

using namespace CGF;

// ls -R -d -1 **/*/*.mtx
//ls -R -d -1 ../shared/matrices/**/*/*.mtx > files.txt

std::vector<std::string> matrixFiles;

template<int N, class T>
void large_cg_test(uint i){
  message("Blocksize = %d\n\n", N);
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
    /*Construct vector b*/
    Vector<T> vec(mat->getWidth());
    for(uint j=0;j<mat->getWidth();j++){
      if(j%2==0)
	vec[j] = 1;
      else
	vec[j] = 1;
    }

    mat->finalize();

    Timer timer;
    LinSolve<N, T>* solver = 0;

    /*Enable the tests you like to perform.*/
#if 1
    message("CPU test single-threaded");

    solver = new LinSolveCG<N, T>(mat->getWidth());

    solver->setMatrix(mat);
    solver->setb(&vec);

    solver->preSolve();

    timer.start();
    solver->solve();
    timer.stop();
    message("time = %d usec\n", timer.getTimeUSec());
    
    /*Obtain result*/
    //result = solver->getx();

    delete solver;
#endif
#if 1
    int n_threads = 8;
    message("CPU test using %d threads", n_threads);

    solver = new LinSolveCGParallel<N, T>(mat->getWidth(), n_threads);

    solver->setMatrix(mat);
    solver->setb(&vec);

    solver->preSolve();

    timer.start();
    solver->solve();
    timer.stop();
    message("time = %d usec\n", timer.getTimeUSec());

    /*Obtain result*/
    //std::cout << solver->getx();

    delete solver;
#endif
#if 1
    uint n_devices = 1;
    message("Cuda test using %d devices", n_devices);

    solver = new LinSolveCGGPU<N, T>(mat->getWidth(), n_devices, 128, 
				     TexVector);
    solver->setMatrix(mat);
    solver->setb(&vec);

    solver->preSolve();

    timer.start();
    solver->solve();
    timer.stop();
    message("time = %d usec\n", timer.getTimeUSec());

    delete solver;
    
#endif

#if 0
    n_devices = 2;
    message("Cuda test using %d devices", n_devices);

    solver = new LinSolveCGGPU<N, T>(mat->getWidth(), n_devices, 128, 
				     TexVector);
    solver->setMatrix(mat);
    solver->setb(&vec);

    solver->preSolve();

    timer.start();
    solver->solve();
    timer.stop();
    message("time = %d usec\n", timer.getTimeUSec());

    delete solver;
    
#endif
    
    delete mat;
  }else{
    warning("File %s not found. Test skipped.", filename);
  }
}

int main(int argc, char** argv){
  CGF::init(0); /*0 normal init, 1 normal init and turns this process
		  into a daemon process. Usefull for remote
		  benchmarking. Output is redirected to a file.*/

  init_cuda_host_thread();
  
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
  
  fileList.close();
  
  /*Conjugate Gradient benchmark routine using single-precision*/
  for(uint i=0;i<matrixFiles.size();i++){
    message("Matrix = %s", matrixFiles[i].c_str());
    large_cg_test<1, float>(i);
    large_cg_test<2, float>(i);
    large_cg_test<4, float>(i);
    large_cg_test<8, float>(i);
  }
  
  /*Conjugate Gradient benchmark using double-precision */
  for(uint i=0;i<matrixFiles.size();i++){
    message("Matrix = %s", matrixFiles[i].c_str());
    large_cg_test<1, double>(i);
    large_cg_test<2, double>(i);
    large_cg_test<4, double>(i);
    large_cg_test<8, double>(i);
  }
  
  CGF::destroy();
  return 0;
}

