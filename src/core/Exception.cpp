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

#include "core/Exception.hpp"
#include <sstream>

namespace CGF{
  const std::string TimeOutException::getError() const{
    std::stringstream stream;
    
    stream << "TimeOutException occured in " << file << ":" << line;
    if(msg != 0){
      stream << "\t|\t" << msg;
    }
    stream << std::endl;
    return stream.str();
  }

  const std::string CUDAException::getError() const{
    std::stringstream stream;
    
    stream << "CUDAException occured in " << file << ":" << line;
    if(msg != 0){
      stream << "\t|\t" << msg;
    }
    stream << std::endl;
    return stream.str();
  }

  const std::string FileNotFoundException::getError() const{
    std::stringstream stream;
    
    stream << "FileNotFoundException occured in " << file << ":" << line;
    if(msg != 0){
      stream << "\t|\t file " << msg << " not found.";
    }
    stream << std::endl;
    return stream.str();
  }

  const std::string ThreadPoolException::getError() const{
    std::stringstream stream;
    
    stream << "ThreadPoolException occured in " << file << ":" << line;
    if(msg != 0){
      stream << "\t|\t" << msg << ".";
    }
    stream << std::endl;
    return stream.str();
  }

  const std::string MathException::getError() const{
    std::stringstream stream;
    
    stream << "MathException occured in " << file << ":" << line;
    if(msg != 0){
      stream << "\t|\t" << msg << ".";
    }
    stream << std::endl;
    return stream.str();
  }

  const std::string SolutionNotFoundException::getError() const{
    std::stringstream stream;
    
    stream << "SolutionNotFoundException occured in " << file << ":" << line;
    if(msg != 0){
      stream << "\t|\t" << msg << ".";
    }
    stream << std::endl;
    return stream.str();
  }

  const std::string AlgException::getError() const{
    std::stringstream stream;
    
    stream << "AlgException occured in " << file << ":" << line;
    if(msg != 0){
      stream << "\t|\t" << msg << ".";
    }
    stream << std::endl;
    return stream.str();
  }

  const std::string DegenerateCaseException::getError() const{
    std::stringstream stream;
    
    stream << "DegenerateCaseException occured in " << file << ":" << line;
    if(msg != 0){
      stream << "\t|\t" << msg << ".";
    }
    stream << std::endl;
    return stream.str();
  }

};
