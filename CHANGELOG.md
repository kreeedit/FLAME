# Changelog

All notable changes to FLAME are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Releases are archived on Zenodo under the
concept DOI [10.5281/zenodo.15805448](https://doi.org/10.5281/zenodo.15805448).

## [Unreleased]

Documentation, benchmark packaging, metadata, and one report-layer fix. The similarity algorithm and its defaults are unchanged.

### Added

- `benchmarks/ostrowski2024/`: a replication package for the evaluation in the FLAME article
  (target: *Digital Scholarship in the Humanities*), Section 10. `run_benchmark.py` fetches the
  Ostrowski (2024b) evaluation set from Zenodo, checks its MD5, rebuilds the chain-level segments
  and distractors, and regenerates Tables 1 and 2 (`results/table1.csv`, `results/table2.csv`,
  `results/run.json`). `--grid` reruns the 459-configuration search that selected the operating
  point. `perturbation.py` runs Task A (recall under synthetic noise) and writes
  `results/perturbation.csv` and `results/perturbation.png`.
- README sections: Method, Evaluation, Reproducibility, References. There is also a Zenodo DOI
  badge.
- `docs/README_UPDATE_REPORT.md`: reproduced numbers next to the article's, plus the
  discrepancies listed below.
- `docs/issues/`: issue drafts for the discrepancies, ready to file.

### Changed

- **Report layer only:** the words after the last matching block of a pair are now classified
  (bridge / orthographic variant / insertion), like the gaps between matches. Before, they were
  printed unmarked and never reached `linguistic_variations.tsv`. As with interior gaps, a tail
  longer than `max_gap_words` on either side stays unmarked. Similarity scores, fingerprints,
  thresholds, `similarity_summary.tsv`, and clusters are unchanged, and the TSV only gains rows.
  (`SimilarityVisualizer.highlight_similarities()`, `generate_linguistic_summary_tsv()`.)
- `CITATION.cff` and the README *Cite* block now give the concept DOI (10.5281/zenodo.15805448),
  version 1.2.0, and its version DOI (10.5281/zenodo.23473940). The earlier version DOIs are kept.
- `html_comparsion.png` renamed to `html_comparison.png`.

### Documented discrepancies (article vs. code). Neither side has been changed, except the report fix noted in item 5.

1. **Table 1 caption wording (not a contradiction between the tables).** Tables 1 and 2 report the
   same run (13 detections, 12 correct, 1 false positive), scored on two gold sets. Table 1 uses the
   19 literal + weak-literal pairs, the subset Ostrowski's baseline figures cover: R = 12/19 = 0.63,
   F1 = 0.75. Table 2 uses all 23 pairs, which adds the four non-literal pairs from Regests 4, 21
   (two pairs), and 22: R = 12/23 = 0.52, F1 = 0.67. The caption's phrase "full 23-pair gold
   standard" should say 19 pairs. The README shows both, with 0.75 as the headline, and
   `results/table1.csv` carries both as `gold_set = literal_weak_19 / all_23`.
2. **"− subword (word-level, vocab ≈ 360)"** in Table 2 is not a word-level tokenizer. It is BPE with
   `-vocab_size auto`, which the morphology estimator resolves to 360 units on this corpus, and
   that is flame.py's default setting.
3. **Similarity weighting.** Article §8 describes cosine similarity over raw count vectors.
   `Flame.compute_similarity_matrix()` applies scikit-learn's `TfidfTransformer` (smoothed IDF, L2
   normalization) before the cosine. TF-IDF arrived in b6d0a22 (1.1.0); 1.0.0 used raw counts, as
   the article describes. The Table 1 row label ("gapped fingerprint + TF-IDF") and every number in
   the benchmark come from the TF-IDF version.
4. **Otsu threshold.** Article §8 says Otsu's method runs on the non-zero scores "and, where the
   residual skew remains severe, on a log-transform of it". `Flame._determine_auto_threshold()` runs
   `skimage.filters.threshold_otsu` on the non-zero scores only, with no log transform. On the
   benchmark, at the article's operating point, Otsu selects 0.063. That accepts 19,878 of the
   53,730 pairs (P < 0.01, F1 = 0.00).
5. **Alignment report, §10.5.** In Dk G5r / KCh r141, *ind herschapie* is rendered as an
   **insertion** (present on one side only), not as a bridge. The three-way
   bridge/variant/insertion classification dates from 784b655. The Regest 5 example
   (*friddens* / *waren vredens*) was not classified because it follows the last matching block.
   This is fixed in the report layer (see *Changed*): Dk G1r / KCh a5r5 now shows
   *friddens* / *vredens* as an orthographic variant, and Dk G1r / KCh a5r4 shows *friddens* /
   *waren vredens* as one bridge. *waren* is not flagged separately.
6. **False positives, §10.5.** The article says several non-background false positives are
   cross-matches inside documented regest clusters. At the operating point there is one false
   positive, and it involves a background segment (Dk H-G1v / KCh bg_0301). A segment-to-segment
   cross-match (Dk G1r / KCh a5r-1) appears only in the ablated configurations.
7. **Evaluation protocol, §10.2.** The protocol describes Task B as a diplomatist-annotated
   stratified sample, with a word-n-gram detector and TRACER as baselines. Table 1 instead reports
   the Ostrowski gold standard, plus Passim/BLAST/TextPAIR figures quoted from Ostrowski (2024a).
8. **Operating point vs. defaults.** The benchmark runs use `-min_text_length 15` (the default 150
   would discard most segments) and `-char_norm_min_freq 2` (default 1). At flame.py's CLI defaults
   (`vocab_size auto`, `ngram 6`, `n_out 1`, Otsu), FLAME reaches F1 = 0.15 on all 23 pairs
   (0.17 on the 19-pair subset).
9. **Provenance of the article's grid search.** The grid search ran on `flame_hashtfidf.py`, a
   development copy that is not tracked in the repository. Its tokenizer, fingerprint, similarity,
   threshold, and vocabulary-estimator functions match `flame.py` except for docstrings, and the
   benchmark reproduces Table 2 exactly on the tracked `flame.py`.
10. **Hash wording.** The article and README call the fingerprint hash a "rolling" Rabin–Karp hash.
    `Flame.leave_n_out_grams()` evaluates the polynomial for each sub-gram directly, as a vectorized
    matrix product, with no rolling update. The hash values are the same either way.

## [1.2.0] - 2026-10-19

Zenodo: [10.5281/zenodo.23473940](https://doi.org/10.5281/zenodo.23473940). Git tag `v1.2.0`.

### Added

- Clustering of related pairs into recurring legal formulas (`flame_clustering.py`, ported from
  KONI). It covers gapped core alignment, Louvain/clique/union linkage, and a separate section for
  near-duplicate charters (`-cluster_*`, `-core_*`, `-min_duplicate_tokens`, `-gen_clusters`).
- Glob patterns in `input_path`/`input_path2`. Reports name files by their path relative to the
  input root (540f5ca).
- Optional phonetic-reduction and bigram-normalization layers (834c990, c0221b0, 847290d).
- New visualization (20ee6ee). `FlameApp.spec` is now tracked for the PyInstaller build.

### Changed

- The engine and the HTML visualizer share one normalization, so orthographic variants are no
  longer shown as unmatched gaps (bf59fcd).
- `fuzz_threshold` now separates an orthographic variant from a genuine bridge, and its default
  moved from 0.75 to 0.70. One-sided gaps are reported as insertions (784b655).
- The README gives the Python and pip version requirements (16abd25).

### Removed

- `flame_cluster.py`, superseded by `flame_clustering.py` (7238275).

## [1.1.0] - 2026-06-26

Released on Zenodo under the version label `GIU`
([10.5281/zenodo.20932532](https://doi.org/10.5281/zenodo.20932532)). Git tag `GIU`. Reconstructed
from the git log between tags `BPE` and `GIU`.

### Added

- Tabbed GUI (data, normalization, auto-tune) and an Otsu/percentile drop-down (766ebca, 88da6e2).
- Self-supervised auto-tune of `ngram`/`n_out`, and a bridge-word fuzzy slider in the HTML report
  (b6d0a22).
- Filename-year heuristic that places the earlier document on the left of the report (b517ea6).
- Optional report generation (`gen_*`, `no_reports`) (d9e9144). Terminal guide for CLI runs
  (9774338).

- `CITATION.cff` (7d8ea00).

### Changed

- Feature counts are TF-IDF weighted before the cosine. In 1.0.0 the cosine ran on raw counts
  (b6d0a22).
- Levenshtein replaced by rapidfuzz (0197273). Threshold handling unified (5df4f34).

### Fixed

- `NoneType` and namespace errors (ab00a88, 16764d8). Cleaner exit on errors (a3a8d58).

## [1.0.0] - 2025-07-04

Zenodo: [10.5281/zenodo.15805449](https://doi.org/10.5281/zenodo.15805449). Git tag `BPE`.

### Added

- First archived release: leave-n-out gapped n-gram fingerprint, adaptive MUFI/Unicode
  normalization, BPE subword tokenizer (Hugging Face `tokenizers`) with a morphology-based
  vocabulary-size estimate, cosine similarity over raw feature counts, Otsu threshold,
  HTML/TSV/heatmap reports, and the Tkinter GUI.

[Unreleased]: https://github.com/kreeedit/FLAME/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/kreeedit/FLAME/compare/GIU...v1.2.0
[1.1.0]: https://github.com/kreeedit/FLAME/compare/BPE...GIU
[1.0.0]: https://github.com/kreeedit/FLAME/releases/tag/BPE
