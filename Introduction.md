# Introduction #

_PaLinSo_ is the result of a part of my Ph.D. research conducted at the Eindhoven University of Technology. _PaLinSo_ stands for Parallel Linear Solver(s) which provides Linear Solvers that run on machines equiped with multi-core processors with or without GPU accelerators. The solvers are a part of a framework intended for Computer Graphics applications which heavily rely on fast iterative methods. In the future the rest of my current research will be added to this framework and made available.


# Details #

The current version contains three implementations of the _Preconditioned Conjugate Gradient method_, a single-threaded, multi-threaded and a multi-device GPU implementation. As a preconditioner a simple Jacobi (diagonal) preconditioner is used. In the near future PaLinSo will be extended such that various types of preconditioners and iterative linear algebra methods can be used.

PalinSo stores the sparse-matrix in _Block Compressed Sparse Row_ (BCSR) format. In order to obrain good performance results on modern GPUs, we store the individual blocks in a different order in GPU memory to obtain a good memory throughput. PaLinSo currently supports square blocks of size N\*N with N = {1, 2, 4, 8} for both single- and double-precision floating point arithmetic, for all implementations. Depending on the application, different block-sizes will give the best results.

# Compiling #
PaLinSo is compiled using the supplied Makefiles. Below the various options of the Makefile are listed:
  * `make clean`  -> removes all .o object files.
  * `make clean_dep` -> removes all .o object files and all .dep dependency files
  * `make`  -> normal compilation with Cuda and SSE2.
  * `make DEBUG=1` -> debug compilation, no optimizations
  * `make CUDA=0`  -> Cuda support is removed
  * `make NOSSE2=1` -> SSE2 support is removed

or a combination of DEBUG CUDA or NOSSE2

# Example #
The file main.cpp gives an example how to use one of the solvers.

`Solver* solver = new LinSolveCG<N, T>(S);`

creates a Conjugate Gradient solver containing an empty square matrix A with dimensions (S x S) and vectors x and b both with size S. Template argument N is the dimension of the used blocks. Valid values for N are 1, 2, 4, 8. Template argument T is float or double.

`Solver* solver = new LinSolveCGParallel<N, T>(S, n);`

creates a parallel solver using n threads.

`Solver* solver = new LinSolveCGGPU<N, T>(S, n, thr, tex);`

creates a parallel GPU solver using n devices. The kernels are executed using _thr_ threads. Valid values for _thr_ are 128, 256 (default) or 512 threads. Parameter _tex_ specifies the texture operation in the SpMV routine for fetching the vector that is multiplied with matrix A. Valid values are _TexNone_ (no texture caching), _TexVector_ (only cache the vector, default), _TexVectorAndIndices_ (caches both the vector and the indices to the vector entries). Depending on the architecture version the use of texture caching _may_ improve the performance.

The following functions are used to obtain pointers to the matrix and vectors allocated by the solver:

  * `SpMatrix<N, T>* mat = solver->getMatrix();`

  * `Vector<T>* x = solver->getx();`

  * `Vector<T>* b = solver->getb();`

Using the bracket operators, the vectors and matrix can be set, i.e.,

  * `(*mat)[0][0] = 10;`
  * `(*x)[0] = 1;`

Depending on the type of solver, the matrix has to be converted and/or uploaded to the GPU. Using `solver->preSolve()`, the matrix and vectors are prepared. Using `solver->solve(n_iter, tol)`, the linear system of equations Ax = b is solved using the _Preconditioned Conjugate Gradient method_ with a maximum of n\_iter iterations and a tolerance specified by tol.

After solving the linear system, the result is stored in vector x and is accessed using `solver->getx()`.

# Known issues #
Converting a sparse matrix into a GPU memory friendly format is quite complex and can take some time for large matrices. Since we were more focussed on optimizing the SpMV operation and the CG method, we didn't optimize the conversion. In a future release this part will be optimized as well.

Using other than Jacobi preconditioners require sparse-matrix sparse-matrix multiplications. Unfortunately this is not implemented yet. Since other projects require this operation, this will be added soon.

PaLinSo has been tested and developed on Linux x86\_64 using cuda 4.2 with gcc 4.7.2. We are currenlty porting PaLinSo to Windows and Android (With NEON support).

Currently three implementations exist of the same algorithm, which is in our opinion not optimal. Therefore we are investigating how we can specify one algorithm that compiles for various platforms.

# Future extensions #
In _Finite Element Method_ (FEM) simulations, a linear system has to be solved each and every iteration of the simulation. Since the structure of the sparse-matrix does not change during these simulations, it is convenient to re-use the structure of the matrix instead of re-creating one. This is especially the case when GPUs are involved. Therefore we are planning to add extesions which easilly updates the sparse-matrix in GPU memory, given a set of local element matrices (which are also stored in GPU memory). This eliminates the reordering and uploading of the sparse-matrix.

# Paper #
See http://www.win.tue.nl/~mverscho/wp/projects/cuda-sparse-matrix/ for more details.