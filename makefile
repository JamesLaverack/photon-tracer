APP      = photon-tracer-cpu

SRCEXT   = cpp
SRCDIR   = src
CUDIR	 = src/cu
CUEXT    = cu
OBJDIR   = obj
BINDIR   = bin
PTXDIR   = bin/ptx
IMGDIR   = img
OUTDIR   = output

QSUB_QUEUE = veryshort
QSUB_CC    = qsub
QSUB_FILE  = qsub_4core

SRCS     := $(shell find $(SRCDIR) -name '*.$(SRCEXT)')
SRCDIRS  := $(shell find . -name '*.$(SRCEXT)' -exec dirname {} \; | uniq)
OBJS     := $(patsubst %.$(SRCEXT),$(OBJDIR)/%.o,$(SRCS))

CUS    := $(shell find $(CUDIR) -name '*.$(CUEXT)')
CUDIRS := $(shell find . -name '*.$(CUEXT)' -exec dirname {} \; | uniq)
PTXS   := $(CUS:$(CUDIR)%=$(PTXDIR)%)
PTXS   := $(PTXS:.cu=.ptx)

CUDA_ONLY := obj/src/CUDAWrapper.o obj/src/OptiXRenderer.o
OBJS_S   := $(filter-out $(CUDA_ONLY), $(OBJS))

DEBUG       = -g
PERFORMANCE = -O3
WARNINGS    = -Wall -W -Wextra
INCLUDES    = -I./inc -I/home/laverack/NVIDIA-OptiX-SDK-3.0.0-linux64/include -I/usr/local/cuda-5.0/include
MPIFLAGS    =
CXX         = g++
CXXFLAGS    = -fmessage-length=0 -c $(DEBUG) $(INCLUDES) $(PERFORMANCE) $(WARNINGS) $(MPIFLAGS)
NVCC        = nvcc
NVCCFLAGS   = $(INCLUDES) $(LDFLAGS) -arch sm_20

.PHONY: all clean distclean

mpi: mpi-set-cxx app-serial clean

serial: app-serial clean

optix: optix-set-cxx app $(PTXS) clean

optix-mpi: optix-set-cxx mpi-set-cxx app $(PTXS) clean

mpi-set-cxx:
	$(eval CXX = mpicxx)
	$(eval MPIFLAGS = -D PHOTON_MPI)
	@echo "Set to MPI"

ptx: buildrepo $(PTXS)

optix-set-cxx:
	$(eval CXXFLAGS  = $(CXXFLAGS) -D PHOTON_OPTIX)
	$(eval NVCCFLAGS = $(NVCCFLAGS) -D PHOTON_OPTIX)
	$(eval LDFLAGS   = -L/home/laverack/NVIDIA-OptiX-SDK-3.0.0-linux64/lib64 -L/usr/local/cuda/lib64 -lcuda -loptix -lcudart)
	@echo "Set to Optix"

app: buildrepo $(OBJS)
	@mkdir -p `dirname $@`
	@echo "Linking $@..."
	$(CXX) $(OBJS) $(LDFLAGS) -o $(BINDIR)/$(APP)

app-serial: buildrepo $(OBJS_S)
	@mkdir -p `dirname $@`
	@echo "Linking $@..."
	$(CXX) $(OBJS_S) $(LDFLAGS) -o $(BINDIR)/$(APP)

$(OBJDIR)/%.o: %.$(SRCEXT)
	@echo "Generating dependencies for $<..."
	@$(call make-depend,$<,$@,$(subst .o,.d,$@))
	@echo "Compiling $<..."
	$(if $(findstring Wrapper,$<),   \
                $(NVCC) $(NVCCFLAGS) -x=cu -c $< -o $@,  \
                $(CXX)  $(CXXFLAGS) $< -o $@)


$(PTXDIR)/%.ptx: $(CUDIR)/%.$(CUEXT)
	@echo "Compiling $< to PTX..."
	$(NVCC) $(NVCCFLAGS) -ptx $< -o $@

submit:
	@$(RM) $(OUTDIR)/*
	@$(RM) photon-tracer.e*
	$(QSUB_CC) -q $(QSUB_QUEUE) $(QSUB_FILE)

clean:
	$(RM) -r $(OBJDIR)

imgclean:
	$(RM) -f photon-tracer.e*
	$(RM) output/*
	$(RM) photons-*.ppm

distclean: clean imgclean
	$(RM) -r $(BINDIR)

buildrepo:
	@$(call make-repo)
	@mkdir -p $(IMGDIR)
	@mkdir -p $(OUTDIR) 
	@mkdir -p $(PTXDIR)

define make-repo
   for dir in $(SRCDIRS); \
   do \
	mkdir -p $(OBJDIR)/$$dir; \
   done
endef


# usage: $(call make-depend,source-file,object-file,depend-file)
define make-depend
  $(CXX) -MM       \
        -MF $3    \
        -MP       \
        -MT $2    \
        $(CXXFLAGS) \
        $1
endef
