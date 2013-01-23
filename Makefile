PROJECT_NAME	= Conjugate_gradient_benchmark
MAJOR_VERSION	= 1
MINOR_VERSION	= 0
PATCH_VERSION	= 0
BUILD_VERSION 	= 0

#Path to shared directory, which contains some scripts
SHARED_PATH	= shared

#Add modules to project. Each module is a separate (sub)directory
MODULES := src src/core src/util src/math

#Define execuatable name and library name/location 
EXE		= main
STATIC_LIB 	= 

#Default rule
#all: lib
all: _main

#Include the master make file which can handle almost everything
include $(SHARED_PATH)/Makefile

#Include CGF static library
COBJ	+= $(CGF_LIB)

#Include some other shared libraries
LIBS 	+= $(CUDA_LIBS) -lpthread
###########################################################################
#Define some project specific rules

include/.lversion.hpp: 
	@echo \#define COMPILE_HOST \"`hostname`\" > $@
	@echo \#define COMPILE_BY \"`whoami`\" >>  $@
	@echo \#define COMPILE_TIME \"`date`\" >> $@
	@echo \#define COMPILE_DOMAIN \"`dnsdomainname`\" >> $@
	@echo \#define COMPILER \"`$(CC) $(CCFLAGS) -v 2>&1 | tail -1`\" >> $@
	@echo \#define COMPILER_FLAGS \"$(CCFLAGS) $(PRJFLAGS) $(INCPATH)\" >> $@
	@echo \#define COMPILER_TARGET \"`$(CC) -dumpmachine`\" >> $@
	@echo \#define NVCC_FLAGS \"$(NVCCFLAGS) \" >> $@	
	@echo \#define MAJOR_VERSION $(MAJOR_VERSION) >> $@
	@echo \#define MINOR_VERSION $(MINOR_VERSION) >> $@
	@echo \#define PATCH_VERSION $(PATCH_VERSION) >> $@
	@echo \#define BUILD_VERSION $(BUILD_VERSION) >> $@
	@echo \#define VERSION \"$(MAJOR_VERSION).$(MINOR_VERSION).$(PATCH_VERSION)\">> $@
	@echo \#define PROJECT_STRING \"$(PROJECT_NAME)\" >> $@

