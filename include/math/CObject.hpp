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

#ifndef CUDAOBJECT_HPP
#define CUDAOBJECT_HPP
#ifndef NO_CUDA
#include "math/Math.hpp"

namespace CGF{
  class Thread;
  class ThreadPool;

  class CObject{
  public:
    CObject(const ThreadPool* p);

    virtual ~CObject();
    
    const ThreadPool* getPool()const;

    uint getNDevices()const;

    void setMRanges(const MatrixRange* range);

    const MatrixRange* getMRange(uint i)const;

    const MatrixRange* getMRanges()const;

    void setVRanges(const VectorRange* range);

    const VectorRange* getVRange(uint i)const;

    const VectorRange* getVRanges()const;

    void setNBlocks(const uint* blocks);

    const uint* getNBlocks()const;

    void copyRanges(const CObject* obj);

    virtual void allocateDevice(const Thread* caller) = 0;
    virtual void deallocateDevice(const Thread* caller) = 0;

  protected:
    const ThreadPool* pool;
    uint n_devices;
    MatrixRange* mRange;
    VectorRange* vRange;
    uint* startBlock;
    uint* n_blocks;
  };
};
#endif/*NO_CUDA*/
#endif/*CUDAOBJECT_HPP*/
