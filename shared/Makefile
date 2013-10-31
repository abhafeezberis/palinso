#Master MAKEFILE 

#Defines some general rules for compiling C, C++ and CUDA programs
#This master MAKEFILE is included in a local, project specific
#Makefile, which defines some paths and local rules

#Making with NOCUDA=1 prevents from compiling CUDA code and including
#CUDA startup code for projects which do not depend on CUDA

NOCUDA	= 0
NOSSE2	= 0
NOSDL   = 1
DEBUG   = 0

#QT Library stuff
MOC4			= moc-qt4
ifeq ($(DEBUG),1)
STRIP_FLAGS		= 
COMPILATION_FLAGS       = -O0 -g -D_DEBUG
else
STRIP_FLAGS		= -ffunction-sections -fdata-sections 
COMPILATION_FLAGS       = -O4 -DN_DEBUG 
endif

ifeq ($(NOCUDA),1)
#CUDA_LIBS		=
else
CUDA_LIBS		= -lcudart 
endif

ifeq ($(NOSDL),1)
SDL :=
else
SDL := -DSDL
endif

#CGF Library stuff
CGF_INSTALL_PATH	= .
CGF_INCLUDE_PATH	= $(CGF_INSTALL_PATH)/include
#CGF_LIB_PATH		= $(CGF_INSTALL_PATH)/../lib
#CGF_LIB			= $(CGF_LIB_PATH)/cgf.a
#CGF_DEBUG_LIB		= $(CGF_LIB_PATH)/cgfd.a

BENCHMARK		= -DN_BENCHMARK	

#Compiler flags
INCLUDES	+= -I./include -I$(CGF_INCLUDE_PATH) 

ifeq ($(NOSSE2),1)
SSE2	:=
else
SSE2	:= -DSSE2
endif

CCFLAGS := $(INCLUDES) -Wall -Werror $(COMPILATION_FLAGS) $(STRIP_FLAGS) -DUNIX $(GPU) -D_REENTRANT -ansi $(SSE2) $(BENCHMARK) $(SDL)

#CUDA Compiler flags
ifeq ($(NOCUDA),1)
NVCCFLAGS	=
CCFLAGS		+= -DNO_CUDA	
else
NVCCFLAGS	+= -Xcompiler "" $(INCLUDES) -DCUDA -DUNIX $(DGPU) $(COMPILATION_FLAGS) -arch sm_13 $(BENCHMARK) 
CCFLAGS		+= -DCUDA
endif

#Link against these libraries
LIBS 	:= 

#Compilers
CC 	= g++
CPP	= g++
ifeq ($(NOCUDA),1)
NVCC	=
else
NVCC	= nvcc
endif

#Default seachpath for MAKE
VPATH 	= include:$(CGF_INCLUDE_PATH):$(CGF_LIB_PATH)

#Automatically lookup our sources. When a source file is in a module
#directory, it wil be find, compiled and linked automatically
CFILES	:= $(wildcard $(patsubst %, %/*.c, $(MODULES)))
CCFILES	:= $(wildcard $(patsubst %, %/*.cpp, $(MODULES)))
CUFILES	:= $(wildcard $(patsubst %, %/*.cu, $(MODULES)))
HPPFILES:= $(wildcard $(patsubst %, %/*.hpp, include))

#Create a list of object ande dependency files
ifeq ($(NOCUDA),1)
OBJ	:= $(CFILES:.c=.o) $(CCFILES:.cpp=.o)
else
OBJ	:= $(CFILES:.c=.o) $(CCFILES:.cpp=.o) $(CUFILES:.cu=.o)
endif

DEPS 	:= $(OBJ:.o=.dep)
COBJ	+= $(OBJ)
ifeq ($(NOCUDA),1)
CUBIN	:= 
CUPTX	:=
else
CUBIN	:= $(CUFILES:.cu=.cubin)
CUPTX 	:= $(CUFILES:.cu=.ptx)
endif

ASM     := $(CCFILES:.cpp=.s)
MOCS	:= $(HPPFILES:.hpp=.moc)

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

ifeq ($(NOCUDA),1)

else
%.o:	%.cu
	$(NVCC) -o $@ -c  $< $(NVCCFLAGS) 
endif

%.moc:	%.hpp
	$(MOC4) -o $(^D)/$@ $<

ifeq ($(NOCUDA),1)

else
%.cubin:	%.cu
	$(NVCC) -o $@ $< $(NVCCFLAGS) --cubin 

%.ptx:	%.cu
	$(NVCC) -o $@ $< $(NVCCFLAGS) --ptx 
endif

nocuda:
	@echo $(NVCCFLAGS)
	@echo $(NVCC)

touchdep: $(DEPS) 
	@touch $(DEPS)

#Rule for linking main executable
# $(CUBIN) #$(CUPTX) #$(ASM)
ifeq ($(DEBUG),1)
_main:	$(DEPS) $(OBJ) update_dep
	$(CC) -o $(EXE) $(COBJ) $(LIBS)
else
_main:	$(DEPS) $(OBJ) update_dep
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

ifeq ($(NOCUDA),1)

else
%.dep: 	%.cu
	@./$(SHARED_PATH)/nvdep.sh `dirname $*.cu` $(NVCCFLAGS) $*.cu > $@	
endif
#	$(NVCC) -M $(NVCCFLAGS) $< > $@

.PHONY: clean
#Clean up
clean:
	rm -f $(OBJ) *.linkinfo include/.version.hpp include/.lversion.hpp $(CUBIN) $(CUPTX) $(ASM) $(MOCS) $(TEMPFILES) Makefile~ 

ifneq ($(MAKECMDGOALS),clean_dep)
ifneq ($(MAKECMDGOALS),clean)
-include $(DEPS)
endif
endif

update_dep:
	@rm -f $(DEPS)

clean_dep: clean update_dep





