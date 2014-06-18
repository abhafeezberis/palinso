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
#include "math/CObject.hpp"
#include "core/ThreadPool.hpp"

namespace CGF{
  CObject::CObject(const ThreadPool* p){
    pool = p;
    n_devices = 1;
    if(p){
      n_devices = p->getSize();
    }

    mRange     = new MatrixRange[n_devices];
    vRange     = new VectorRange[n_devices];
    n_blocks   = new int[n_devices];
    startBlock = new int[n_devices];
  }

  CObject::~CObject(){
    delete [] mRange;
    delete [] vRange;
    delete [] n_blocks;
    delete [] startBlock;
  }

  const ThreadPool* CObject::getPool()const{
    return pool;
  }

  int CObject::getNDevices()const{
    return n_devices;
  }
  
  void CObject::setMRanges(const MatrixRange* range){
    for(int i=0;i<n_devices;i++){
      mRange[i] = range[i];
    }
  }

  const MatrixRange* CObject::getMRange(int i)const{
    return &(mRange[i]);
  }

  const MatrixRange* CObject::getMRanges()const{
    return mRange;
  }

  void CObject::setVRanges(const VectorRange* range){
    for(int i=0;i<n_devices;i++){
      vRange[i] = range[i];
    }
  }

  const VectorRange* CObject::getVRange(int i)const{
    return &(vRange[i]);
  }

  const VectorRange* CObject::getVRanges()const{
    return vRange;
  }

  void CObject::setNBlocks(const int* blocks){
    for(int i=0;i<n_devices;i++){
      n_blocks[i] = blocks[i];
    }
  }

  const int* CObject::getNBlocks()const{
    return n_blocks;
  }

  void CObject::copyRanges(const CObject* obj){
    setMRanges(obj->getMRanges());
    setVRanges(obj->getVRanges());
    setNBlocks(obj->getNBlocks());
  }
  
}
#endif/*CUDA*/
