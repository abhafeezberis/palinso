#Master MAKEFILE 

#Defines some general rules for compiling C, C++ and CUDA programs
#This master MAKEFILE is included in a local, project specific
#Makefile, which defines some paths and local rules

#Making with CUDA=0 prevents from compiling CUDA code and including
#CUDA startup code for projects which do not depend on CUDA

CUDA	= 1
NOSSE2	= 0
NOSDL   = 0
DEBUG   = 0

#QT Library stuff
MOC4			= moc
ifeq ($(DEBUG),1)
STRIP_FLAGS		= 
COMPILATION_FLAGS       = -O0 -g -D_DEBUG 
#-pg
else
STRIP_FLAGS		= -ffunction-sections -fdata-sections 
COMPILATION_FLAGS       = -O4 -DN_DEBUG 
#-pg
endif
#COMPILATION_FLAGS 	= -O0 -pg -g -D_DEBUG
#

ifeq ($(CUDA),1)
CUDA_LIBS		= -lcudart 
else
#CUDA_LIBS		=
endif
#-lcublas

ifeq ($(NOSDL),1)
SDL :=
else
SDL := -DSDL
endif

#Compiler flags
INCLUDES	+= -I./include 

ifeq ($(NOSSE2),1)
SSE2	:=
else
SSE2	:= -DSSE2
endif

CCFLAGS := $(INCLUDES) -Wall -Werror $(COMPILATION_FLAGS) $(STRIP_FLAGS) -DUNIX $(GPU) -D_REENTRANT -ansi -pedantic $(SSE2) $(BENCHMARK)

#CUDA Compiler flags
ifeq ($(CUDA),1)
NVCCFLAGS	+= -Xcompiler "" $(INCLUDES) -DUNIX -DCUDA $(COMPILATION_FLAGS) -arch sm_13
#--compiler-bindir=/usr/bin/gcc-4.4 
#--device-emulation
CCFLAGS		+= -DCUDA	
else
NVCCFLAGS	=
endif

#Link against this libraries
LIBS 	:= 

#Compilers
CC 	= g++
CPP	= g++
ifeq ($(CUDA),1)
NVCC	= nvcc
else
NVCC	=
endif

#Default seachpath for MAKE
VPATH 	= include:$(CGF_INCLUDE_PATH):$(CGF_LIB_PATH)

#Automatically lookup our sources. When a source file is in a module
#directory, it wil be find, compiled and linked automatically
CFILES	:= $(wildcard $(patsubst %, %/*.c, $(MODULES)))
CCFILES	:= $(wildcard $(patsubst %, %/*.cpp, $(MODULES)))
CUFILES	:= $(wildcard $(patsubst %, %/*.cu, $(MODULES)))
HPPFILES:= $(wildcard $(patsubst %, %/*.hpp, include))
MPPFILES:= $(wildcard $(patsubst %, %/*.hpp, $(QT_MOC_DIR)))

#Create a list of object ande dependency files
ifeq ($(CUDA),1)
OBJ	:= $(CFILES:.c=.o) $(CCFILES:.cpp=.o) $(CUFILES:.cu=.o)
else
OBJ	:= $(CFILES:.c=.o) $(CCFILES:.cpp=.o)
endif

DEPS 	:= $(OBJ:.o=.dep)
COBJ	+= $(OBJ)
ifeq ($(CUDA),1)
CUBIN	:= $(CUFILES:.cu=.cubin)
CUPTX 	:= $(CUFILES:.cu=.ptx)
else
CUBIN	:= 
CUPTX	:=
endif

ASM     := $(CCFILES:.cpp=.s)
MOCS	:= $(MPPFILES:.hpp=.moc)
#DEPS 	+= $(MOCS)

#Temporary emacs files
TEMPFILES += $(CFILES:.c=.c~) $(CCFILES:.cpp=.cpp~) $(CUFILES:.cu=.cu~)
TEMPFILES += $(HPPFILES:.hpp=.hpp~)

#Rules for compiling different files
%.o:	%.c
	$(CC) $(CCFLAGS) -c $< -o $@ 

%.o:	%.cpp
	$(CPP) $(CCFLAGS) -c $< -o $@

%.s:	%.cpp
	$(CPP) $(CCFLAGS) -S $< -o $@

ifeq ($(CUDA),1)
%.o:	%.cu
	$(NVCC) -o $@ -c  $< $(NVCCFLAGS) 
else

endif

%.moc:	%.hpp
	$(MOC4) -o $@ $<
#	$(MOC4) -o $@ $<	

ifeq ($(CUDA),1)
%.cubin:	%.cu
	$(NVCC) -o $@ $< $(NVCCFLAGS) --cubin 

%.ptx:	%.cu
	$(NVCC) -o $@ $< $(NVCCFLAGS) --ptx 
else

endif

nocuda:
	@echo $(NVCCFLAGS)
	@echo $(NVCC)

touchdep: $(DEPS) 
	@touch $(DEPS)

#Rule for linking main executable
# $(CUBIN) #$(CUPTX) #$(ASM)
ifeq ($(DEBUG),1)
_main:	$(DEPS) $(OBJ) touchdep
	$(CC) -o $(EXE) $(COBJ) $(LIBS)
#-pg
else
_main:	$(DEPS) $(OBJ) touchdep
	$(CC) -o $(EXE) $(COBJ) $(LIBS) -Wl,-s -Wl,--gc-sections
endif
#

#Rule for creating static library
lib:	$(DEPS) $(OBJ) $(CUBIN)
	rm -f $(STATIC_LIB)
	ar rcs $(STATIC_LIB) $(OBJ)

debug_lib: $(DEPS) $(OBJ) $(CUBIN)
	rm -f $(STATIC_DEBUG_LIB)
	ar rcs $(STATIC_DEBUG_LIB) $(OBJ)

#	$(CC) -M $(CCFLAGS) > $@
#Rules for creating dependencies
%.dep: 	%.c
	@./$(SHARED_PATH)/dep.sh `dirname $*.c` $(CCFLAGS) $*.c > $@	

%.dep: 	%.cpp
	@./$(SHARED_PATH)/dep.sh `dirname $*.cpp` $(CCFLAGS) $*.cpp > $@	

ifeq ($(CUDA),1)
%.dep: 	%.cu
	@./$(SHARED_PATH)/nvdep.sh `dirname $*.cu` $(NVCCFLAGS) $*.cu > $@	
else

endif
#	$(NVCC) -M $(NVCCFLAGS) $< > $@

.PHONY: clean
#Clean up
clean:
	rm -f $(OBJ) *.linkinfo include/.version.hpp include/.lversion.hpp $(CUBIN) $(CUPTX) $(ASM) $(MOCS) $(TEMPFILES) Makefile~ 

ifneq ($(MAKECMDGOALS),clean_dep)
ifneq ($(MAKECMDGOALS),clean)
include $(DEPS)
endif
endif

clean_dep: clean
	rm -f $(DEPS)

#Forces make to create the moc files
qt:	$(MOCS) all




