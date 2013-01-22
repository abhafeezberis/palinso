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

#include "core/daemon.hpp"
#include "core/cgfdefs.hpp"
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>

namespace CGF{

  int daemon_process = 0;

  int continue_as_daemon_process(){
    pid_t pid;

    pid = fork();

    if(pid){
      daemon_process = 0;
      printf("Daemon: %d\n", pid);
      return 0;
    }

    setsid();
    umask(0);

    daemon_process = 1;
    return 1;
  };

  int is_daemon(){
    return daemon_process;
  };

  void redirect_std_file_descriptors(){
    /*Redirect stdout*/
    char buffer[1024];
    sprintf(buffer, "%d.stdout.txt", getpid());
    freopen(buffer, "w", stdout);

    /*Redirect stderr*/
    sprintf(buffer, "%d.stderr.txt", getpid());
    freopen(buffer, "w", stderr);
  }

};
