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

#include "util/CSVExporter.hpp"
#include <string.h>

namespace CGF{

  CSVExporter::CSVExporter(FILE* f){
    file = f;
    headerWritten = false;
    n_columns = 0;
  }

  CSVExporter::~CSVExporter(){
    
  }

  void CSVExporter::addColumn(const char* columnName){
    std::map<std::string, std::string>::iterator it;
    std::string key(columnName);

    it = valueMap.find(key);
    if(it == valueMap.end()){
      /*Column does not exist, create an entry*/
      std::string value;
      valueMap[key] = value;
      indexMap[n_columns] = key;
      n_columns++;
    }else{
      /*Column already exists*/
    }
  }

  void CSVExporter::setValue(const char* column, float value){
    std::map<std::string, std::string>::iterator it;
    std::string key(column);
    
    it = valueMap.find(key);

    if(it != valueMap.end()){
      char buffer[255];
      sprintf(buffer, "%10.10e%c", value,'\0');      
      std::string strval(buffer);
      valueMap[key] = strval;
    }else{
      warning("Column %s unknown", column);
    }
  }

  void CSVExporter::setValue(const char* column, double value){
    std::map<std::string, std::string>::iterator it;
    std::string key(column);
    
    it = valueMap.find(key);

    if(it != valueMap.end()){
      char buffer[255];
      sprintf(buffer, "%10.10e%c", value,'\0');      
      std::string strval(buffer);
      valueMap[key] = strval;
    }else{
      warning("Column %s unknown", column);
    }
  }

  void CSVExporter::setValue(const char* column, int value){
    std::map<std::string, std::string>::iterator it;
    std::string key(column);
    
    it = valueMap.find(key);

    if(it != valueMap.end()){
      char buffer[255];
      sprintf(buffer, "%d%c", value,'\0');
      std::string strval(buffer);
      valueMap[key] = strval;
    }else{
      warning("Column %s unknown", column);
    }
  }


  void CSVExporter::setValue(const char* column, uint value){
    std::map<std::string, std::string>::iterator it;
    std::string key(column);
    
    it = valueMap.find(key);

    if(it != valueMap.end()){
      char buffer[255];
      sprintf(buffer, "%d%c", value,'\0');
      std::string strval(buffer);
      valueMap[key] = strval;
    }else{
      warning("Column %s unknown", column);
    }
  }

  void CSVExporter::setValue(const char* column, long value){
    std::map<std::string, std::string>::iterator it;
    std::string key(column);
    
    it = valueMap.find(key);

    if(it != valueMap.end()){
      char buffer[255];
      sprintf(buffer, "%ld%c", value,'\0');
      std::string strval(buffer);
      valueMap[key] = strval;
    }else{
      warning("Column %s unknown", column);
    }
  }

  void CSVExporter::setValue(const char* column, ulong value){
    std::map<std::string, std::string>::iterator it;
    std::string key(column);
    
    it = valueMap.find(key);

    if(it != valueMap.end()){
      char buffer[255];
      sprintf(buffer, "%lu%c", value,'\0');
      std::string strval(buffer);
      valueMap[key] = strval;
    }else{
      warning("Column %s unknown", column);
    }
  }

  void CSVExporter::setValue(const char* column, const char* value){
    std::map<std::string, std::string>::iterator it;
    std::string key(column);
    
    it = valueMap.find(key);

    if(it != valueMap.end()){
      std::string strval(value);
      valueMap[key] = strval;
    }else{
      warning("Column %s unknown", column);
    }
  }

  void CSVExporter::saveHeader(){
    for(int i=0;i<n_columns;i++){
      fprintf(file, "%s", indexMap[i].c_str());

      if(i+1 == n_columns){
        fprintf(file, "\n");
      }else{
        fprintf(file, ";");
      }
    }
    
    headerWritten = true;
  }

  void CSVExporter::saveRow(){
    if(!headerWritten){
      saveHeader();
    }

    for(int i=0;i<n_columns;i++){
      fprintf(file, "%s", valueMap[indexMap[i]].c_str());

      if(i+1 == n_columns){
        fprintf(file, "\n");
      }else{
        fprintf(file, ";");
      }
    }
    fflush(file);
  }
}
