# Thesis figures, conventions

Lessons from `experiments/plot_pruning.py` and `experiments/plot_hierarchy_steps.py`. Apply to every figure meant for the thesis.

## Resolution
- Always high resolution: PDF (vector) is the thesis version; every PNG is saved at 600 ppi (never 300 or less). Check with `PIL.Image.open(png).info["dpi"]`.

## Fonts (LaTeX look)
- Text and formulas are typeset by LaTeX itself: matplotlib pgf backend, `pgf.texsystem = pdflatex`, `pgf.rcfonts = False`, preamble `\usepackage{amsmath}\usepackage{amssymb}`. The PDF then embeds Computer Modern (CMR, CMMI, CMSY), the same fonts as the thesis (standard Overleaf `article`, 10pt).
- Fallback when `pdflatex` is missing: mathtext `"cm"` with serif `cmr10` (close, not identical).
- No matplotlib path effects (halos) on text: they make the pgf backend fall back to a sans font.

## Size
- Figure width = `\textwidth` of the article class at 10pt (345 pt = 4.77 in), included at `width=\textwidth`, so nothing is scaled: 9 pt text (`\small`), 8 pt legend (`\footnotesize`).

## Labels
- No label may sit on a line, mesh edge, fence, arrow or marker. Pick the offset per label so it lands in an empty gap (check the four diagonal directions first), then look at the rendered image and move it if anything touches it.
- No white boxes behind labels either: they cut visible gaps into lines.
- Legend labels short (e.g. "rep. of level k+1", not "becomes rep of level k+1"); no legend entry for elements that need none (e.g. plain grey points).

## Content (data-ink)
- Draw nothing without a meaning the reader can name. Examples: coloured edges x_A -> x_s were dropped; merges are shown as one spanning tree per new cell, not all inner edges; dot size stays constant across panels.
- A category with only a handful of barely visible instances gets no own style or legend entry; fold it into the nearest one and mention it in the caption (e.g. attached leftovers drawn like round 2).
- Encode with ink that is already there where possible (e.g. new reps dark, other points grey, instead of extra rings).
- Every marker type appears in the legend, and set membership is shown by the marker (e.g. t(x_A) is in Y_hat, so it uses the Y_hat diamond).
- Prefer a legend label that defines the object (e.g. "Y_hat(x_A) = Y cap U(x_A)").
- Use real data from the repo's own functions; choose the seed so that no element coincides visually with another (e.g. an edge passing through an unrelated point), and check this programmatically.
- Legend outside the panels (`loc="outside lower center"`), one line.

## Colour and marks
- Colour is what tells the series apart; one palette with fixed roles in every result figure (`thesis_style.py`: `BLUE`, `ORANGE`, `GREEN`, `KIND_COLOURS`; greys `INK`, `MUTED`, `FAINT`). Decided with Benedikt 2026-11-19.

| colour | hex | variants compared in an experiment | point kinds (when they are the series) |
|---|---|---|---|
| blue | #2b5d8c | the thesis default / our method (Delaunay 1-ring, with masks, no pruning, multiscale) | uniform |
| orange | #d17a22 | first alternative (8 sectors, without masks, exact facets / pruning on) | clustered |
| green | #3b8f6a | second alternative (6 sectors, closest 60 %) | grid |

- External dense references stay neutral: POT grey (`MUTED`), dense CPLEX black (`INK`). Other categories (e.g. the time shares) use the same three colours plus `FAINT`.
- Kinds and variants are never both series in one figure, so the reuse of the three colours is unambiguous; the legend always names them.
- Markers are the second channel for greyscale print and colour-blind readers: default circle (filled), first alternative square, second alternative triangle (both hollow); kinds: grid square, uniform circle, clustered triangle. The default comes first in strips and legends.
- With colour carrying the meaning, mean lines and series lines are solid; a line style is used only for a further variable (e.g. dashed d = 4, solid d = 5). Guide lines (slope 1, 2) are thin dashed `FAINT`, labelled along the line.
- Thin lines (about 0.4 for mesh, 1.0 to 1.3 for main lines), small markers at text width.

## Files and LaTeX
- Approved figures go into the folder `visuals` directly in the OneDrive "Masters thesis" folder, which is copied as a whole to Overleaf; only the newest version of each figure lives there, and experiment outputs go there only when Benedikt says so.
- **Keep `visuals` in sync:** whenever a figure listed below is re-rendered, copy the new PDF into `visuals` right away (same file name, overwrite), so `visuals` always holds the newest version. A new figure joins the list only when Benedikt approves it. If a source moves (e.g. a worktree is merged), update its row.

| figure in `visuals` | source (relative to the repo) | script |
|---|---|---|
| `candidate_sets_concept.pdf` | not in the repo yet (rendered in a Cowork session) | `plot_candidate_sets_concept.py` (not yet in `experiments/`) |
| `candidate_sets_level0_with_6_sectors.pdf` | `experiments/results/candidate_sets/` | `experiments/plot_candidate_sets.py` |
| `candidate_sets_runtime_with_6_sectors.pdf` | `experiments/results/candidate_sets/` | `experiments/plot_candidate_sets.py` |
| `hierarchy_rounds.pdf` | `experiments/results/` | `experiments/plot_hierarchy_steps.py` |
| `pruning_fences.pdf` | `experiments/results/` | `experiments/plot_pruning.py` |
| `pruning_summary_dimensions.pdf` | `experiments/results/pruning_summary/` | pruning summary plot script |
| `scaling_sparsity.pdf` | `.claude/worktrees/scaling/experiments/results/scaling/` (until `mtw-pruning` results move to the main tree) | `experiments/plot_scaling.py` |
| `scaling_time_guides.pdf` | `.claude/worktrees/scaling/experiments/results/scaling/` (same) | `experiments/plot_scaling.py` |
- In LaTeX every figure is included as `visuals/<name>.pdf` with a short label `fig:<name>`.

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{visuals/pruning_fences.pdf}
  \caption{...}
  \label{fig:pruning}
\end{figure}
```

## Process
- Render, look at the image, fix, render again before handing it over.
