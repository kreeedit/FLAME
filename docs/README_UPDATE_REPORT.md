# README update report

Branch `main`, on top of `v1.2.0` (7238275). Nothing has been pushed, tagged, or released. The
algorithm and its defaults are unchanged: no commit touches `flame.py`, `flame_gui.py`, or
`flame_clustering.py`.

## Read this first

- **All article numbers in Tables 1 and 2 reproduce exactly from the tracked `flame.py`.** One
  command does it: `python benchmarks/ostrowski2024/run_benchmark.py`. The 459-configuration grid
  search (`--grid`) also lands on the article's operating point.
- **Tables 1 and 2 are the same run on two gold denominators.** 13 detections, 12 correct, 1 false
  positive.
  - **19 literal + weak-literal pairs** (the subset Ostrowski's Passim/BLAST/TextPAIR figures
    cover): P 0.92, R 12/19 = 0.63, **F1 0.75**. This is the README headline.
  - **All 23 pairs**, which adds the 4 non-literal (allusive) pairs: P 0.92, R 12/23 = 0.52, F1 0.67.
    Table 2 (ablation) uses this set.
  - The 4 non-literal pairs: Dk F5r2-1 / KCh a4v1-1 (**Regest 4**), Dk F6v1-1 and Dk F7r-1 /
    KCh b1r6-1 (**Regest 21**, two pairs), Dk F6v3-1 / KCh b1r7-1 (**Regest 22**). Regest numbers
    follow Ostrowski (2024c). None is detected at the operating point (cosines 0.08–0.19).
- **The Otsu default does not work at the operating point.** It selects 0.063 and accepts 19,878
  of 53,730 pairs, giving F1 0.00. The README shows this as the third row of the gold-set table.
- **Task A (synthetic noise) does not show the steep divergence the article predicts.** Gapped
  FLAME stays at or above the contiguous variant, but by at most 0.06 recall, within the seed
  spread.
- **Two §10.5 claims about the alignment report do not hold for the current code** (details below).
- Issues could not be filed: `gh` is not installed here, and nothing is pushed. Three drafts are
  ready in `docs/issues/`.

## Commits

1. `1ec8b98` Rename html_comparsion.png to html_comparison.png
2. `e048cba` Cite the Zenodo concept DOI; list version DOIs for 1.2.0, GIU and 1.0.0
3. `f21a621` Add Ostrowski (2024) replication package for the article's Tables 1-2
4. `304be02` run_benchmark: judge git_dirty on the code files only, not on results/
5. `7b2919a` Add Ostrowski benchmark results produced at 304be02
6. `a7f1efa` run_benchmark: name false positives in run.json only when there are at most 50
7. `a1353eb` README: add Method, Evaluation, Reproducibility and References sections
8. `7e45510` Add CHANGELOG (1.0.0, 1.1.0/GIU, 1.2.0, Unreleased) and issue drafts
9. `99451b6` Regenerate run.json with the capped false-positive list (13 KB)
10. (this report) Add docs/README_UPDATE_REPORT.md

`results/run.json` records commit 7e45510, `git_dirty: false`. The benchmark code is unchanged since a7f1efa, and the commits after it touch only documentation.

## Reproduced vs. article

From `benchmarks/ostrowski2024/results/table1.csv` and `table2.csv`, produced at the commit
recorded in `results/run.json`.

**Table 1** (literal_weak_19 unless noted)

| Row | Article | Reproduced |
|---|---|---|
| Passim / BLAST / TextPAIR | quoted from Ostrowski (2024a) | not rerun (tools not in repo); quoted unchanged |
| FLAME P / R / F1 / pairs | 0.92 / 0.63 / 0.75 / 12 | 0.92 / 0.63 / 0.75 / 12 (threshold 0.28) |
| FLAME, all_23 | (Table 2 row 1) | 0.92 / 0.52 / 0.67 / 12 |
| FLAME, Otsu threshold | not in article | 0.0631 → 0.00 / 1.00 / 0.00 (19,878 detections) |
| FLAME, CLI defaults + Otsu | not in article | all_23: 0.50 / 0.09 / 0.15; literal_weak_19: 0.50 / 0.11 / 0.17 |

**Table 2** (all_23, best-F1 threshold per row)

| Configuration | Article P / R / F1 | Reproduced P / R / F1 | threshold |
|---|---|---|---|
| vocab 35, ngram 5, n_out 2 | 0.92 / 0.52 / 0.67 | 0.92 / 0.52 / 0.67 | 0.28 |
| − gapping (n_out 0) | 0.69 / 0.48 / 0.56 | 0.69 / 0.48 / 0.56 | 0.12 |
| − fine tokens (vocab 200) | 0.50 / 0.30 / 0.38 | 0.50 / 0.30 / 0.38 | 0.11 |
| − subword ("word-level, vocab ≈ 360") | 0.62 / 0.22 / 0.32 | 0.62 / 0.22 / 0.32 | 0.10 |

**Grid search** (`--grid`, 459 configurations × 36 thresholds, 9–13 min on this machine): the optimum is
vocab 35, ngram 5, n_out 2, threshold 0.28, F1 0.667 on all 23 pairs, identical to
`gridsearch_log.txt` from June.

**Qualitative example cosines** (`run.json`): Dk G5r / KCh r141 0.5396 (article: 0.54). Dk G4v-1 /
KCh r141 0.5538. Dk G1r / KCh a5r4 0.2827.

**Task A** (`perturbation.py`, seeds 42–46, mean recall over all 23 pairs):

| r | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|---|---|
| FLAME (n_out 2, thr 0.28) | 0.52 | 0.50 | 0.47 | 0.44 | 0.41 | 0.39 | 0.43 |
| contiguous (n_out 0, thr 0.12) | 0.48 | 0.44 | 0.43 | 0.43 | 0.41 | 0.37 | 0.37 |

The article has no numbers for Task A, only the hypothesis in §10.3.

**Determinism:** two consecutive runs of `run_benchmark.py` and of `perturbation.py` gave
byte-identical CSVs. The rebuilt segments are byte-identical to the local `mappa_dk/` and
`mappa_kch/` the June grid search used. The local copy of the TEI file has the same MD5 as the
Zenodo file.

## Discrepancies and how each was handled

Code was never changed to match the article, or the article to match the code. Each item is in
`CHANGELOG.md` under *Documented discrepancies*, and items 2–7 also have an issue draft.

1. **Table 1 vs. Table 2.** Not a contradiction. They are the same run on 19 and 23 pairs, as above.
   The only fault is the Table 1 caption wording ("full 23-pair gold standard"). Handled by
   showing both denominators in the README, labeling Table 2 "all 23 pairs", and writing both to
   `table1.csv` with a `gold_set` column. No "Table 1 vs Table 2 recall discrepancy" issue was
   opened, because 0.75 reproduces.
2. **"− subword (word-level, vocab ≈ 360)"** is BPE with `-vocab_size auto` (360 units), which is
   flame.py's default setting, not a word-level tokenizer. The label is kept in the README table
   as instructed, with an explanation under it. Issue draft 03.
3. **Count vs. TF-IDF cosine** (article §8 vs. `compute_similarity_matrix`). TF-IDF entered in
   b6d0a22 (1.1.0); 1.0.0 used raw counts. Issue draft 03.
4. **Otsu "on a log-transform"** is not implemented, and Otsu fails at the operating point. Issue
   draft 01.
5. **§10.5 report rendering.** *ind herschapie* is an insertion, not a bridge. In *friddens* /
   *waren vredens*, the trailing divergence after the last matching block is printed unmarked.
   The README describes what the code actually shows. Issue draft 02.
6. **§10.5 false positives.** The only false positive at the operating point is a background
   segment, not a regest cross-match. Issue draft 03.
7. **§10.2 Task B protocol** (diplomatist sample, TRACER baseline) is not what Table 1 reports.
   TRACER has not been run. Issue draft 03.
8. **Grid search ran on the untracked `flame_hashtfidf.py`.** Its core functions match `flame.py`
   except for docstrings (AST comparison), and the reproduction on `flame.py` is exact.
9. **"Rolling" hash.** The polynomial is evaluated directly per sub-gram, so only the wording
   differs.

## What could not be reproduced or done

- **Passim, BLAST, TextPAIR**: not rerun. They are quoted from Ostrowski (2024a) and marked
  `source = quoted` in `table1.csv`.
- **TRACER** (named in article §10.2): no figure exists, and none is printed.
- **Ablation "− normalization"** (mentioned in article §10.4 but absent from Table 2): not run,
  because the article has no number to check against. It would need a flag that disables the
  normalization stack, which is a code change.
- **The friddens/vredens rendering** described in §10.5: the current code does not produce it
  (issue draft 02).
- **Issue tracker**: drafts only (`docs/issues/01–03`). They reference files that exist only after
  these commits are pushed.

## Other changes worth checking

- The local `main` was one commit behind `origin/main` (7238275, which deletes `flame_cluster.py`).
  It was fast-forwarded before any commit, so the work sits on `v1.2.0`.
- The Zenodo record lists the 2026-06-26 release under the version label `GIU`. CHANGELOG and
  `CITATION.cff` call it 1.1.0, reconstructed from the git range `BPE..GIU`. The prompt's "1.1.x"
  has no other counterpart on Zenodo.
- Concept DOI 10.5281/zenodo.15805448 was read from the Zenodo API (`conceptrecid` of both
  version records).
- `matplotlib` (for the plot) and `cffconvert` (to validate `CITATION.cff`, which passes) were
  installed into the project `venv`. `benchmarks/ostrowski2024/requirements.txt` lists matplotlib.
- The Key Features bullet on the automatic threshold no longer says it removes manual guesswork,
  because the evaluation shows otherwise. "Novel words" in the comparison table became "unseen
  words".
- `README.md` previously never referenced `html_comparsion.png`. The renamed `html_comparison.png`
  is now shown under *Outputs*. The local `.claudesignore` reference was updated too; that file is
  untracked.
