# Otsu threshold accepts ~37% of all pairs on the Ostrowski benchmark; no log transform as the article describes

**Labels:** threshold, documentation, article

## What happens

`Flame._determine_auto_threshold()` runs `skimage.filters.threshold_otsu` on `dist_mat.data`, the
non-zero cosine scores. On the Ostrowski (2024b) benchmark (`benchmarks/ostrowski2024/run_benchmark.py`),
the inter-corpus matrix holds 22+113 Dk segments by 19+379 KCh segments, which is 53,730 pairs.

| configuration | Otsu threshold | pairs accepted | P | R | F1 |
|---|---|---|---|---|---|
| vocab 35, ngram 5, n_out 2 (article operating point) | 0.0631 | 19,878 | 0.00 | 1.00 | 0.00 |
| vocab 35, ngram 5, n_out 0 | 0.1357 | 9 | 1.00 | 0.39 | 0.56 |
| CLI defaults (vocab auto, ngram 6, n_out 1) | 0.0459 | 4 | 0.50 | 0.09 | 0.15 |

At the operating point, the very short BPE units (vocab 35) make most segment pairs share some
gapped feature. The "non-zero" filter therefore no longer removes the mass of unrelated pairs, and
Otsu splits inside that background.

## Article vs. code

Article §8: "On this filtered subset (and, where the residual skew remains severe, on a
log-transform of it), Otsu's criterion …". The code has no log transform and no skew test.

## Reproduce

    python benchmarks/ostrowski2024/run_benchmark.py
    # results/run.json -> runs -> "FLAME (vocab 35, ngram 5, n_out 2)" -> otsu

## Not done

Nothing in `flame.py` was changed. Either the article text should be brought in line with the
code, or the log-transform branch should be implemented and re-measured. That is the
maintainer's call.
