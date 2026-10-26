# Otsu threshold: document when it applies; the log transform described in the article is not implemented

**Labels:** threshold, documentation, article

## Measurement

`Flame._determine_auto_threshold()` runs `skimage.filters.threshold_otsu` on `dist_mat.data`, the
non-zero cosine scores. On the Ostrowski (2024b) benchmark, at the operating point (vocab 35,
ngram 5, n_out 2), 53,716 of the 53,730 Dk–KCh pairs have a non-zero score. The distribution has a
single mode near 0.05 and a thin tail of 13 pairs at 0.28 or above:

    0.00–0.04  15,446    0.16–0.20  212    0.32–0.36  1
    0.04–0.08  28,088    0.20–0.24   51    0.36–0.40  1
    0.08–0.12   8,641    0.24–0.28   12    0.44–0.56  6
    0.12–0.16   1,253    0.28–0.32    4    0.60–0.64  1

Otsu assumes two classes, and here there is one, so it selects 0.063. The benchmark uses the
article's fixed threshold, 0.28 (`benchmarks/ostrowski2024/run_benchmark.py`).

## Article vs. code

Article §8: "On this filtered subset (and, where the residual skew remains severe, on a
log-transform of it), Otsu's criterion …". The code has neither a log transform nor a skew test.

## Possible actions (maintainer's call)

- Bring the article text in line with the code, or implement the log-transform branch and
  re-measure it.
- Optionally, warn when the non-zero scores show no second mode, so that users set
  `-similarity_threshold` explicitly.
