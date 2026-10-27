# STAT 701 HW3

- `HW3.tex`: complete assignment transcription with empty `Sol.` blocks.
- `HW3_assignment.pdf`: original assignment.
- `code/problem1.py`, `code/problem2.py`, `code/problem3.py`: function skeletons only; fill in each `NotImplementedError` before running.
- `data/nao.csv`: unchanged copy of the HW2 NAO data.
- `data/tide.csv`: reserved path for the tide gauge data from Canvas; not supplied yet. Adjust the path to match the actual file.
- `png/`: generated figures.

The Python files contain no comments. All three scripts are included in the LaTeX appendix. Paths are resolved relative to the homework directory regardless of the current working directory.

Suggested packages: NumPy, SciPy, pandas, Matplotlib, statsmodels. Use your existing Python environment.

Run from this directory after completing the implementations:

```bash
python3 code/problem1.py
python3 code/problem2.py
python3 code/problem3.py
xelatex HW3.tex
xelatex HW3.tex
```

Keep figures in `png/` and insert them into the corresponding solution with `\includegraphics`.
