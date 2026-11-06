# Thesis figures, conventions

Lessons from `experiments/plot_pruning.py` (pruning fences figure). Apply to every figure meant for the thesis.

## Labels
- No label may sit on a line, mesh edge, fence, arrow or marker. Pick the offset per label so it lands in an empty gap (check the four diagonal directions first), then look at the rendered image and move it if anything touches it.
- No white boxes or halos behind labels to hide overlaps: boxes cut visible gaps into lines, and matplotlib path-effect halos make the pgf backend fall back to a sans font instead of LaTeX.
- Every label must clearly belong to its point (place it close, in the same colour as the marker where that helps).

## Content
- Draw nothing without a meaning the reader can name. Example: coloured edges x_A -> x_s were dropped because they read as mesh edges; label one x_s and its y_s instead.
- Every marker type appears in the legend, and set membership is shown by the marker (e.g. t(x_A) is in Y_hat, so it uses the Y_hat diamond).
- Prefer a legend label that defines the object (e.g. "Y_hat(x_A) = Y cap P(x_A)").
- Use real data from the repo's own functions (e.g. the cost's `needed_candidates`), not hand-placed points.

## Typesetting
- Text through LaTeX (matplotlib pgf backend, pdflatex, amsmath/amssymb), so fonts are the thesis's Computer Modern. Fallback without pdflatex: mathtext "cm".
- Figure width = `\textwidth` of the article class at 10pt (345 pt = 4.77 in), included at `width=\textwidth`, so nothing is scaled: 9 pt text (`\small`), 8 pt legend (`\footnotesize`).
- Legend outside the panels (`loc="outside lower center"`), preferably one line.
- PDF is the thesis version (vector); PNG at 600 dpi for previews.

## Colour and marks
- Two-state encodings get a second channel besides colour (kept: solid blue circle; dropped: dashed orange square).
- Thin lines (about 0.4 for mesh, 0.9 to 1.1 for main lines), small markers at text width.

## Process
- Render, look at the image, fix, render again before handing it over.
