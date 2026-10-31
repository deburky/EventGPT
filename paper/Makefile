# EventGPT paper: build the PDF and package the arXiv upload.
#
#   make        compile the PDF (pdflatex as on arXiv, run twice for cross-references)
#   make zip    compile, then stage the source into tarball/ and create tarball/eventgpt-arxiv.tar.gz
#   make clean  remove build artifacts and tarball/

TEX := eventgpt.tex
PDF := eventgpt.pdf
DIST := tarball
TARBALL := $(DIST)/eventgpt-arxiv.tar.gz
LATEX := pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all pdf zip clean

all: pdf

pdf: $(PDF)

FIGS := $(wildcard figures/*.pdf)

$(PDF): $(TEX) $(FIGS)
	$(LATEX) $(TEX)
	$(LATEX) $(TEX)

# Stage exactly what arXiv receives (source and figure PDFs, no paper PDF, no top-level folder) and tar it.
# Depends on pdf so a broken build never gets packaged.
zip: pdf
	rm -rf $(DIST)
	mkdir -p $(DIST)/figures
	cp $(TEX) $(DIST)/
	cp $(FIGS) $(DIST)/figures/
	cd $(DIST) && tar czf eventgpt-arxiv.tar.gz $(TEX) figures
	@echo "wrote $(TARBALL):"
	@tar tzf $(TARBALL)

clean:
	rm -rf $(DIST)
	rm -f $(PDF) *.aux *.log *.out *.toc *.synctex.gz
