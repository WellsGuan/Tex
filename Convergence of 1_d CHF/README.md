# Convergence of 1-D CHF

This directory is a clean multi-file LaTeX template for rewriting the project.

- `main.tex` controls the title page, style, contents, and chapter order.
- `macros.tex` contains shared notation.
- `tex/` contains the manuscript sections.
- `figures/` contains graphics.
- `References.bib` contains bibliography entries.

Compile from this directory with:

```sh
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```
