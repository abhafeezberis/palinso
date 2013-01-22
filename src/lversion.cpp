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

#include ".lversion.hpp"
#include "core/cgfdefs.hpp"
#include "core/version.hpp"

using namespace CGF;

void version_print()
{
  message("Compile host    = %s", COMPILE_HOST);
  message("Compiled by     = %s", COMPILE_BY);
  message("Compile time    = %s", COMPILE_TIME);
  message("Compile domain  = %s", COMPILE_DOMAIN);
  message("Compiler        = %s", COMPILER);
  message("Compiler flags  = %s", COMPILER_FLAGS);
  message("Compiler target = %s", COMPILER_TARGET);
  message("CUDA flags      = %s", NVCC_FLAGS);
  message("Version         = %s", VERSION);
  message("Project string  = %s", PROJECT_STRING);  
}
