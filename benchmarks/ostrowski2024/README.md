# Ostrowski (2024) benchmark: replication package

This package regenerates every number in the README's [Evaluation](../../README.md#evaluation)
section and in Tables 1–2 of the FLAME article (§10) from code in this repository.

## One command

From the repository root, with the dependencies from `requirements.txt` installed:

```bash
python benchmarks/ostrowski2024/run_benchmark.py          # Tables 1 and 2, a few seconds
python benchmarks/ostrowski2024/run_benchmark.py --grid   # plus the 459-config grid search, about 15 min
python benchmarks/ostrowski2024/perturbation.py           # Task A noise curve, under a minute; needs matplotlib (benchmarks/ostrowski2024/requirements.txt)
```

| Script | Writes |
| --- | --- |
| `run_benchmark.py` | `results/table1.csv` (one row per system and gold set, `gold_set` = `literal_weak_19` or `all_23`), `results/table2.csv` (ablation, `all_23`), `results/run.json` (git commit, Python and package versions, seed, wall-clock time, dataset MD5, every metric at the best-F1 and the Otsu threshold, false positives by name) |
| `run_benchmark.py --grid` | `results/grid.csv` (every configuration × threshold) and the grid optimum in `run.json` |
| `perturbation.py` | `results/perturbation.csv`, `results/perturbation.png` |

The pipeline itself uses no random numbers: BPE training, hashing, and the TF-IDF cosine are
deterministic, and two consecutive runs give byte-identical tables. The seed (`SEED = 42` in
`run_benchmark.py`, seeds 42–46 in `perturbation.py`) only drives the synthetic noise of Task A.

`alignment_regest13.png` is a screenshot of the HTML report produced by

```bash
python flame.py -input_path <segments>/dk -input_path2 <segments>/kch -min_text_length 15 \
    -vocab_size 35 -ngram 5 -n_out 2 -char_norm_min_freq 2 -similarity_threshold 0.53
```

cropped to the Dk `G5r` / KCh `r141` pair.

## The dataset

Ostrowski, A. (2024b). *Evaluation Set for Historical Text Reuse Detection.* Zenodo.
<https://doi.org/10.5281/zenodo.13378221>. The set is a single TEI file,
`Evaluation_Set_for_Historical_Text_Reuse_Detection.xml` (150,581 bytes, MD5
`2381353fb446b3e85e6e1deed4e59c61`).

**License: CC BY-SA 4.0.** That license would permit redistribution with attribution under the
same license. The file is still **not** vendored here, because FLAME itself is Apache-2.0 and the
derived segments would have to carry the share-alike terms. `run_benchmark.py` downloads the file
from the Zenodo record on first use, checks the MD5, and stops if it differs. To work offline,
pass a local copy with `--xml path/to/file.xml`.

Expected layout after the first run. `data/` is git-ignored:

```
benchmarks/ostrowski2024/
├── data/
│   ├── Evaluation_Set_for_Historical_Text_Reuse_Detection.xml
│   └── segments/
│       ├── dk/    Dk_seg_<head-id>.txt (22), Dk_bg_0001.txt … (113)
│       ├── kch/   KCh_seg_<head-id>.txt (19), KCh_bg_0001.txt … (379)
│       └── gold_standard.json   19 links = 23 Dk–KCh pairs
└── results/
```

## How the corpus is built (`build_segments.py`, article §10.1)

The TEI file marks reuse as `<seg type="tr">` fragments chained by `prev`/`next`. A fragment is 7–49
characters long. The unit of reuse is the whole chain, so each chain is written as one file, named
after its head segment. The KCh side's `corresp` attributes give the gold standard: 19 links that
expand to **23 Dk–KCh pairs**, graded by Ostrowski as literal (13), weak literal (6), and
non-literal (4). Runs of `<l>` lines with no reuse annotation are merged into distractor chunks of
about 120 characters, and chunks under 20 characters are dropped. This gives 113 Dk and 379 KCh
distractors. FLAME runs in two-directory mode with Dk as `input_path` and KCh as `input_path2`, so
every one of the 135 Dk files is compared with every one of the 398 KCh files (53,730 pairs).

**Two gold sets.** Ostrowski's baseline figures cover the 19 literal and weak-literal pairs, so
that subset (`literal_weak_19`) is the basis for comparing FLAME with them, and it is the headline
in the README. `all_23` adds the four non-literal pairs, which reuse sense without sharing surface
form: Dk F5r2-1 / KCh a4v1-1 (Regest 4), Dk F6v1-1 and Dk F7r-1 / KCh b1r6-1 (Regest 21), and
Dk F6v3-1 / KCh b1r7-1 (Regest 22). The regest numbers come from Ostrowski (2024c,
<https://doi.org/10.5281/zenodo.14410766>). The TEI file does not carry them, so they were
assigned by matching each segment's text against the passages each regest quotes (see
`REGEST_OF_NON_LITERAL` in `run_benchmark.py`). On the 19-pair set, a detection of a non-literal
gold pair is left out of scoring rather than counted as a false positive. At the operating point
none is detected, so precision is the same on both sets.

A detected pair counts as a true positive if it is one of the gold pairs. Every other detected
Dk–KCh pair is a false positive, including a pair of two annotated segments that belong to
different regests.

## Settings

All runs pass `-min_text_length 15 -char_norm_min_freq 2 -vocab_min_word_freq 5 -vocab_coverage
0.85`. flame.py's default `min_text_length` of 150 would drop most segments, because the median
chain is 77 characters. Everything else is flame.py's `DEFAULT_PARAMS`. The engine is called
through `Flame.load_corpus()` and `Flame.compute_similarity_matrix()`, exactly as `flame.py` and
the GUI call it. Nothing is patched.

**Operating point.** vocab 35, ngram 5, n_out 2, threshold 0.28 is the best-F1 cell on all 23
pairs (ties broken by precision) of the grid `vocab_size ∈ {30…40, 45, 60, 80, 120, 200, auto}` × `ngram ∈ {2, 3, 4, 5, 6,
8, 10}` × `n_out ∈ {0, 1, 2, 4, 5}` × threshold 0.05–0.40 in steps of 0.01. The grid was scored
on the same 23 gold pairs that Table 1 reports, and **there is no held-out split**. Treat the
best-F1 figures as an in-sample upper bound. The ablation rows (Table 2) are scored the same way,
each at its own best threshold.

## Baselines

The Passim, BLAST, and TextPAIR rows of Table 1 are quoted from Ostrowski (2024a), not rerun. None
of the three tools is part of this repository. According to the article's Table 1 caption,
Ostrowski ran them as open retrieval over the full texts and scored the literal and weak-literal
subset. Their recall values match that reading: 5/19 = 0.26 and 3/19 = 0.16. `table1.csv` marks these rows with
`source = quoted`.

## Task A: synthetic noise (`perturbation.py`)

Every KCh file is perturbed at rate r ∈ {0, 0.05, …, 0.30} with seeds 42–46. Each word is hit with
probability r by one applicable operation: a character substitution, insertion, or deletion; a
u/v or i/j interchange; consonant doubling or undoubling; or an orthographic variant attested in the
gold pairs (16 spellings, found with flame.py's own `classify_gap`). The full pipeline is rerun
for each rate and seed. Two systems are compared at their clean best-F1 thresholds: FLAME at the
operating point (0.28), and the same pipeline with `n_out 0` (0.12). The README reports the mean
recall over the 23 pairs. The per-run recall, precision, TP, and FP are in
`results/perturbation.csv`.

