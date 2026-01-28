# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = .
BUILDDIR     = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx-build using the -M option
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets for convenience
clean:
	rm -rf $(BUILDDIR)/*

livehtml:
	sphinx-autobuild -b html $(ALLSPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html

install:
	pip install -r requirements.txt

serve:
	cd $(BUILDDIR)/html && python -m http.server 8000 