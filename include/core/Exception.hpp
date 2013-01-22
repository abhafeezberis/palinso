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

#ifndef EXCEPTION_HPP
#define EXCEPTION_HPP

#include "core/cgfdefs.hpp"
#include <string>

namespace CGF{
  class Exception{
  public:
    Exception(const uint l, const char* f, const char* m):
      line(l), file(f), msg(m){
    };
    
    virtual ~Exception(){
    };
    virtual const std::string getError() const = 0;
  protected:
    const uint line;
    const char* file;
    const char* msg;
  };

  class TimeOutException : public Exception{
  public:
    TimeOutException(const uint l, const char* f, const char* m=0):
      Exception(l, f, m){
    };
    virtual ~TimeOutException(){
    };
    const std::string getError() const;
  };

  class FileNotFoundException : public Exception{
  public:
    FileNotFoundException(const uint l, const char* f, const char* m=0):
      Exception(l, f, m){
    };
    virtual ~FileNotFoundException(){
    };
    const std::string getError() const;
  };

  class CUDAException : public Exception{
  public:
    CUDAException(const uint l, const char* f, const char* m=0):
      Exception(l, f, m){
    }

    virtual ~CUDAException(){
    }

    const std::string getError() const;
  };

  class ThreadPoolException : public Exception{
  public:
    ThreadPoolException(const uint l, const char* f, const char* m=0):
      Exception(l, f, m){
    }
    virtual ~ThreadPoolException(){
    }

    const std::string getError() const;
  };

  class MathException : public Exception{
  public:
    MathException(const uint l, const char* f, const char* m=0):
      Exception(l, f, m){
    }

    virtual ~MathException(){}

    const std::string getError() const;
  };

  class SolutionNotFoundException : public MathException{
  public:
    SolutionNotFoundException(const uint l, const char* f, const char* m=0):
      MathException(l, f, m){
    }
    virtual ~SolutionNotFoundException(){}

    const std::string getError() const;
  };

  class AlgException : public Exception{
  public:
    AlgException(const uint l, const char* f, const char* m = 0):
      Exception(l, f, m){
    }

    virtual ~AlgException(){}

    const std::string getError() const;

  };

  class DegenerateCaseException : public AlgException{
  public:
    DegenerateCaseException(const uint l, const char* f, const char* m = 0):
      AlgException(l, f, m){
    }

    virtual ~ DegenerateCaseException(){}

    const std::string getError() const;
  };
}

#endif/*EXCEPTION_HPP*/
