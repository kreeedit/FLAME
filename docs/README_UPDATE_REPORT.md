# README update report

Branch `main`, on top of `v1.2.0` (7238275). Nothing has been pushed, tagged, or released. This
report covers three rounds: the original task, follow-up 1 (0.75 as the headline), and follow-up 2
(short README, fixed threshold, layer runs, Regest 5).

## Read this first

- **Tables 1 and 2 of the article reproduce exactly from the tracked `flame.py`**:
  `python benchmarks/ostrowski2024/run_benchmark.py`. The 459-configuration grid (`--grid`) lands
  on the article's operating point: vocab 35, ngram 5, n_out 2, threshold 0.28.
- **Two denominators, one run.** There are 13 detections: 12 gold pairs and 1 background false
  positive.
  - **19 literal + weak-literal pairs**, the subset Ostrowski's baseline figures cover: P 0.92,
    R 0.63, **F1 0.75**. This is the README headline.
  - **All 23 pairs**: P 0.92, R 0.52, F1 0.67. The ablation is scored on this set.
  - The 4 non-literal pairs are **Regest 4** (Dk F5r2-1 / KCh a4v1-1), **Regest 21** (Dk F6v1-1
    and Dk F7r-1 / KCh b1r6-1), and **Regest 22** (Dk F6v3-1 / KCh b1r7-1), numbered as in
    Ostrowski (2024c).
- **Threshold.** The benchmark passes the article's 0.28. The non-zero scores are unimodal (53,716
  of 53,730 pairs non-zero, one mode near 0.05), so Otsu has nothing to split. The Otsu row is
  gone from all tables.
- **Dialect layers (A1–A3) do not raise the README figure.** At 0.28 on the 19 pairs: A0 F1 0.75,
  A1 0.60, A2 0.69 (P 1.00), and A3 0.69 (P 1.00). The README therefore reports A0 and has no
  "A-best" row.
- **Task A.** Across A0–A3, the mean recall difference FLAME − contiguous lies between −0.08 and
  +0.06 at every rate. It is largest in A0, which is the plotted run.
- **Regest 5 is resolved in the report layer.** Words after the last matching block are now
  classified, so Dk *friddens* / KCh *vredens* shows as an orthographic variant. This is the only
  change to `flame.py`; similarity, fingerprints, and thresholds are untouched.
- **README: 217 lines** (was 901). The detail moved verbatim to `docs/` and to
  `benchmarks/ostrowski2024/README.md`, which is the full report.
- Issues were not filed, because `gh` is not installed and nothing is pushed. Drafts are in
  `docs/issues/`.

## Commits (oldest first)

1. `1ec8b98` Rename html_comparsion.png to html_comparison.png
2. `e048cba` Cite the Zenodo concept DOI; list version DOIs for 1.2.0, GIU and 1.0.0
3. `f21a621` Add Ostrowski (2024) replication package for the article's Tables 1-2
4. `304be02` run_benchmark: judge git_dirty on the code files only, not on results/
5. `7b2919a` Add Ostrowski benchmark results produced at 304be02
6. `a7f1efa` run_benchmark: name false positives in run.json only when there are at most 50
7. `a1353eb` README: add Method, Evaluation, Reproducibility and References sections
8. `7e45510` Add CHANGELOG (1.0.0, 1.1.0/GIU, 1.2.0, Unreleased) and issue drafts
9. `99451b6` Regenerate run.json with the capped false-positive list (13 KB)
10. `970fe26` Add docs/README_UPDATE_REPORT.md for the maintainer
11. `849a24c` Report: classify the words after the last matching block
12. `d6e9266` Benchmark: fixed threshold 0.28 and normalization-layer runs A0-A3
13. `66fac3a` Benchmark results for threshold 0.28 and layer runs A0-A3; full report
14. `60d35c8` README: restructure into a short user document; move detail to docs/
15. `f7dc9da` CHANGELOG and issue drafts: threshold 0.28, layer runs, README move
16. (this report) Update docs/README_UPDATE_REPORT.md

The current `results/` were produced at `d6e9266` (`run.json`: `git_dirty: false`, 558 s with
`--grid`). Later commits touch only documentation.

## Reproduced vs. article

From `benchmarks/ostrowski2024/results/table1.csv` and `table2.csv`.

**Table 1** (threshold 0.28)

| Row | Article | Reproduced |
|---|---|---|
| Passim / BLAST / TextPAIR | quoted from Ostrowski (2024a) | quoted unchanged (`source = quoted`) |
| FLAME, 19 pairs | 0.92 / 0.63 / 0.75 / 12 | 0.92 / 0.63 / 0.75 / 12 |
| FLAME, all 23 pairs | (= Table 2, row 1) | 0.92 / 0.52 / 0.67 / 12 |

**Table 2** (all 23 pairs, best-F1 threshold per row): every row reproduces exactly.

| Configuration | Article P / R / F1 | Reproduced | threshold |
|---|---|---|---|
| vocab 35, ngram 5, n_out 2 | 0.92 / 0.52 / 0.67 | 0.92 / 0.52 / 0.67 | 0.28 |
| − gapping (n_out 0) | 0.69 / 0.48 / 0.56 | 0.69 / 0.48 / 0.56 | 0.12 |
| − fine tokens (vocab 200) | 0.50 / 0.30 / 0.38 | 0.50 / 0.30 / 0.38 | 0.11 |
| − subword ("word-level, vocab ≈ 360") | 0.62 / 0.22 / 0.32 | 0.62 / 0.22 / 0.32 | 0.10 |

The last row is BPE with `-vocab_size auto` (360 units), not a word-level tokenizer. The label is
kept as in the article and explained in the benchmark README.

## Layer runs A0–A3 (follow-up 2, §1b)

**Table 1 at 0.28** (P / R / F1, pairs found):

| Run | Layers | 19 pairs | all 23 pairs |
|---|---|---|---|
| A0 | none | 0.92 / 0.63 / 0.75, 12 | 0.92 / 0.52 / 0.67, 12 |
| A1 | bigram | 0.82 / 0.47 / 0.60, 9 | 0.82 / 0.39 / 0.53, 9 |
| A2 | bigram + phonetic | 1.00 / 0.53 / 0.69, 10 | 1.00 / 0.43 / 0.61, 10 |
| A3 | + auto_tune (→ ngram 4, n_out 1) | 1.00 / 0.53 / 0.69, 10 | 1.00 / 0.43 / 0.61, 10 |

**Task A, mean recall at r = 0 → r = 0.30, and mean precision at r = 0 / r = 0.30:**

| Run | FLAME recall | contiguous recall | FLAME P | contiguous P | mean gap |
|---|---|---|---|---|---|
| A0 | 0.52 → 0.43 | 0.48 → 0.37 | 0.92 / 0.96 | 0.69 / 0.82 | +0.03 |
| A1 | 0.39 → 0.37 | 0.39 → 0.35 | 0.82 / 0.96 | 0.69 / 0.89 | −0.01 |
| A2 | 0.43 → 0.36 | 0.43 → 0.36 | 1.00 / 0.95 | 0.83 / 0.97 | 0.00 |
| A3 | 0.43 → 0.38 | 0.48 → 0.42 | 1.00 / 1.00 | 0.61 / 0.81 | −0.05 |

The full per-rate tables are in `benchmarks/ostrowski2024/README.md`.

**Which run the README reports, and why:** A0. It has the highest F1 at 0.28 on both gold sets,
and it is the article's configuration. Per the instruction, an "A-best" row would be added only if
the layers improved on A0, and they do not.

**Threshold interpretation (please check).** The instruction table says "0.28 fixed" for every
run. FLAME is scored at 0.28. The contiguous variant is scored at 0.12, its clean best-F1
threshold, as in the original A0 run. At a shared 0.28, the contiguous variant's mean recall in A0
is 0.30 → 0.16 at precision 1.00. Both are in `perturbation.csv` (`recall_at_0_28`,
`precision_at_0_28`), and the benchmark README states both.

## What moved where (follow-up 2, §2)

| From the README | To |
|---|---|
| Method, five long stage descriptions | `docs/METHOD.md` (verbatim; the Otsu sentence reworded per §1a, links fixed, references added) |
| Worked example, Method Comparison table | dropped (instructed) |
| Evaluation detail: ablation, Task A, threshold note, TRACER note, run metadata, Regest 5 | `benchmarks/ostrowski2024/README.md` |
| All CLI Arguments table; CLI and GUI detail; input-path detail; PyInstaller detail | `docs/CLI.md` |
| Clustering (full text, measurements, anchor experiments); column-layout note | `docs/CLUSTERING.md` |
| Recipes (full `DEFAULT_PARAMS` blocks) | `docs/RECIPES.md`; the README keeps a one-line command per recipe |
| References not cited in the trimmed README (Smith 2014, Gladstone 2018, Büchler 2014, Korkiakangas 2024) | the benchmark README and `docs/METHOD.md` |

**Deviation from the brief.** The brief says "Recipes: keep" and also "README ≤ 240 lines". The
recipes alone were 145 lines, so the README keeps a recipes table (one command each, plus a new
row for the benchmark's short-segment setting), and the full blocks live in `docs/RECIPES.md`.

**Verification** (follow-up 2, §4): the README is 217 lines. `grep -c "0.75"` gives 3 (TextPAIR's
P, the Table 1 FLAME row, and the 19-pair row). `grep -c "Otsu"` gives 2 (the Method sentence and
the Otsu 1979 reference). `grep -ci "upper bound\|does not show\|has not been run"` gives 0. Each of
`docs/METHOD.md`, `docs/CLUSTERING.md`, `docs/CLI.md`, and `benchmarks/ostrowski2024/README.md` is
linked once. A script compared every table row in both READMEs against `table1.csv`,
`table2.csv`, and `perturbation.csv` and found 0 mismatches. All relative links and anchors
resolve.

## The report-layer change (follow-up 2, §1c)

`SimilarityVisualizer.highlight_similarities()` and `generate_linguistic_summary_tsv()` now
classify the tail after the last matching block, under the same rule as the gaps: at most
`max_gap_words` on each side, three classes. Leading gaps were already classified.

On the benchmark at 0.28, `linguistic_variations.tsv` gains 14 rows and loses none. Among them:
Dk G1r / KCh a5r5 *friddens* / *vredens* is an orthographic variant, and Dk G1r / KCh a5r4
*friddens* / *waren vredens* is a bridge. *waren* is not flagged on its own, as the article
describes it; classification works per gap, not per word. `similarity_summary.tsv` and
`clusters.tsv` are unchanged, and `/tmp/test_clustering.py` gives 160 OK both before and after.
The Georg_problems0928 regression sample was not rerun, because its input corpus is not in the
repository.

## Discrepancies (article vs. code)

All are in `CHANGELOG.md` under *Documented discrepancies*, and items 2–7 have issue drafts.

1. Table 1 caption: says "full 23-pair"; the figures are on 19 pairs.
2. "− subword (word-level)" is BPE with `vocab_size auto`.
3. §8 count-cosine vs. TF-IDF in the code (TF-IDF since b6d0a22; 1.0.0 used counts).
4. §8 Otsu "on a log-transform": not implemented. The benchmark scores are unimodal.
5. §10.5 bridges: *ind herschapie* is an insertion. Regest 5 is now shown, with the
   per-gap caveat above.
6. §10.5 false positives: the only false positive at the operating point is a background segment.
7. §10.2 protocol (diplomatist sample, TRACER) differs from Table 1's basis. TRACER is not
   included.
8. §10.3 Task A hypothesis vs. the measured A0–A3 range.
9. The grid search originally ran on the untracked `flame_hashtfidf.py`. Its core functions match
   `flame.py`, and the reproduction is exact.
10. "Rolling" hash: the polynomial is evaluated directly. Only the wording differs.

## Not reproduced or not done

- Passim, BLAST, TextPAIR: quoted, not rerun. TRACER: no figure exists.
- "− normalization" ablation (§10.4 text, not in Table 2): not run; it would need a code switch.
- The issue tracker: drafts only.

## Other notes

- Local `main` was fast-forwarded to `origin/main` (7238275) before the first commit.
- Zenodo labels the 2026-06-26 release `GIU`. It is called 1.1.0 here, reconstructed from
  `BPE..GIU`.
- `matplotlib` and `cffconvert` were installed into the project venv. `CITATION.cff` validates.
- In the HTML report, the legend shows the bridge swatch in yellow, while bridges render in red
  and matches in yellow (visible in `alignment_regest05.png`). This is pre-existing and was not
  changed.
