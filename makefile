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

DEBUG       = -g
PERFORMANCE = -O3
WARNINGS    = -Wall -W -Wextra
INCLUDES    = -I./inc -I/usr/local/optix/include -I/usr/local/cuda/include
MPIFLAGS    =
CXX         = g++
CXXFLAGS    = -fmessage-length=0 -c $(DEBUG) $(INCLUDES) $(PERFORMANCE) $(WARNINGS) $(MPIFLAGS)
LDFLAGS     = -L/usr/local/optix/lib64 -L/usr/local/cuda/lib64 -lcuda -loptix -lcudart
NVCC        = nvcc
NVCCFLAGS   = $(INCLUDES) $(LDFLAGS) -arch sm_20

.PHONY: all clean distclean

all: $(BINDIR)/$(APP) clean

mpi: mpi-set-cxx all

optix: optix-set-cxx all $(PTXS)

mpi-set-cxx:
	$(eval CXX = mpicxx)
	$(eval MPIFLAGS = -D PHOTON_MPI)
	@echo "Set to MPI"

ptx: buildrepo $(PTXS)

optix-set-cxx:
	$(eval CXXFLAGS = $(CXXFLAGS) -D PHOTON_OPTIX)
	$(eval NVCCFLAGS = $(NVCCFLAGS) -D PHOTON_OPTIX)
	@echo "Set to Optix"

$(BINDIR)/$(APP): buildrepo $(OBJS)
	@mkdir -p `dirname $@`
	@echo "Linking $@..."
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

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
