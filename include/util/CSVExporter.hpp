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

#ifndef CSVEXPORTER_HPP
#define CSVEXPORTER_HPP

#include "core/cgfdefs.hpp"
#include "stdio.h"
#include <map>
#include <string>

namespace CGF{
  class CSVExporter{
  public:
    CSVExporter(FILE* f);
    ~CSVExporter();

    void addColumn(const char* columnName);
    void setValue(const char* column, float value);
    void setValue(const char* column, double value);
    void setValue(const char* column, uint value);
    void setValue(const char* column, int value);
    void setValue(const char* column, long value);
    void setValue(const char* column, ulong value);
    void setValue(const char* column, const char* value);
    
    void saveHeader();
    void saveRow();
    void clear(){
      n_columns = 0;
      valueMap.clear();
      indexMap.clear();
    }
  protected:
    FILE* file;
    bool headerWritten;
    std::map<std::string, std::string> valueMap;
    std::map<int, std::string> indexMap;
    int n_columns;
    
  };  
}

#endif/*CSVEXPORTER_HPP*/
