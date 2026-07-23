DOCUMENT := demo_gateway_documentation

.PHONY: all clean

all: $(DOCUMENT).pdf

$(DOCUMENT).pdf: $(DOCUMENT).tex bibfile.bib architecture.pdf
	pdflatex -interaction=nonstopmode -halt-on-error $(DOCUMENT).tex
	bibtex $(DOCUMENT)
	pdflatex -interaction=nonstopmode -halt-on-error $(DOCUMENT).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(DOCUMENT).tex

clean:
	$(RM) $(DOCUMENT).aux $(DOCUMENT).bbl $(DOCUMENT).blg \
		$(DOCUMENT).log $(DOCUMENT).out $(DOCUMENT).toc
