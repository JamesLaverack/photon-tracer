APP      = photon-tracer-cpu

SRCEXT   = cpp
SRCDIR   = src
OBJDIR   = obj
BINDIR   = bin
IMGDIR   = img
OUTDIR   = output

QSUB_QUEUE = veryshort
QSUB_CC    = qsub
QSUB_FILE  = qsub_4core

SRCS    := $(shell find $(SRCDIR) -name '*.$(SRCEXT)')
SRCDIRS := $(shell find . -name '*.$(SRCEXT)' -exec dirname {} \; | uniq)
OBJS    := $(patsubst %.$(SRCEXT),$(OBJDIR)/%.o,$(SRCS))

DEBUG       = -g
PERFORMANCE = -O3
WARNINGS    = -Wall -W -Wextra
INCLUDES    = -I./inc
MPIFLAGS    =
CXX         = g++
CXXFLAGS    = -fmessage-length=0 -c $(DEBUG) $(INCLUDES) $(PERFORMANCE) $(WARNINGS) $(MPIFLAGS)
LDFLAGS     =

.PHONY: all clean distclean

all: $(BINDIR)/$(APP) clean

mpi: mpi-set-cxx all

mpi-set-cxx:
	$(eval CXX = mpicxx)
	$(eval MPIFLAGS = -DMPI)
	@echo "Set to MPI"

$(BINDIR)/$(APP): buildrepo $(OBJS)
	@mkdir -p `dirname $@`
	@echo "Linking $@..."
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

$(OBJDIR)/%.o: %.$(SRCEXT)
	@echo "Generating dependencies for $<..."
	@$(call make-depend,$<,$@,$(subst .o,.d,$@))
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $< -o $@

submit:
	@$(RM) $(OUTDIR)/*
	@$(RM) photon-tracer.e*
	$(QSUB_CC) -q $(QSUB_QUEUE) $(QSUB_FILE)

clean:
	$(RM) -r $(OBJDIR)

distclean: clean
	$(RM) -r $(BINDIR)

buildrepo:
	@$(call make-repo)
	@mkdir -p $(IMGDIR)
	@mkdir -p $(OUTDIR) 

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
