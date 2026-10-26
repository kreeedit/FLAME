# Ostrowski (2024) benchmark: replication package and full report

This folder regenerates every number in the README's [Evaluation](../../README.md#evaluation)
section and in Tables 1–2 of the FLAME article (Kovács, in preparation, §10) from code in this
repository. It also holds the material the README does not show: the ablation, the
normalization-layer runs, the noise curves, the threshold selection, and the run metadata.

## One command

From the repository root, with the dependencies from `requirements.txt` installed:

```bash
python benchmarks/ostrowski2024/run_benchmark.py          # Tables 1 and 2, a few seconds
python benchmarks/ostrowski2024/run_benchmark.py --grid   # plus the 459-config grid search, 10-15 min
python benchmarks/ostrowski2024/perturbation.py           # Task A, A0-A3, about 1.5 min; needs matplotlib (requirements.txt here)
```

| Script | Writes |
| --- | --- |
| `run_benchmark.py` | `results/table1.csv` (columns `layers` and `gold_set` = `literal_weak_19` / `all_23`; threshold 0.28), `results/table2.csv` (ablation, `all_23`, best-F1 threshold per row), `results/run.json` (metadata, below) |
| `run_benchmark.py --grid` | `results/grid.csv` (every configuration × threshold) and the grid optimum in `run.json` |
| `perturbation.py` | `results/perturbation.csv` (column `run` = A0–A3), `results/perturbation.png` |

The pipeline uses no random numbers: BPE training, hashing, and the TF-IDF cosine are deterministic,
and two consecutive runs give byte-identical tables. The seeds (`SEED = 42` in `run_benchmark.py`,
42–46 in `perturbation.py`) drive only the synthetic noise of Task A.

## Results

### Table 1: detection at threshold 0.28

| System | Layers | Gold set | P | R | F1 | pairs found / gold |
|---|---|---|---|---|---|---|
| Passim (Ostrowski 2024a) | | 19 literal + weak-literal | 1.00 | 0.26 | 0.42 | 5 / 19 |
| BLAST (Ostrowski 2024a) | | 19 literal + weak-literal | 0.38 | 0.26 | 0.31 | 5 / 19 |
| TextPAIR (Ostrowski 2024a) | | 19 literal + weak-literal | 0.75 | 0.16 | 0.26 | 3 / 19 |
| FLAME, A0 (vocab 35, ngram 5, n_out 2) | none | 19 literal + weak-literal | 0.92 | 0.63 | 0.75 | 12 / 19 |
| FLAME, A0 | none | all 23 | 0.92 | 0.52 | 0.67 | 12 / 23 |
| FLAME, A1 | bigram | 19 literal + weak-literal | 0.82 | 0.47 | 0.60 | 9 / 19 |
| FLAME, A1 | bigram | all 23 | 0.82 | 0.39 | 0.53 | 9 / 23 |
| FLAME, A2 | bigram + phonetic | 19 literal + weak-literal | 1.00 | 0.53 | 0.69 | 10 / 19 |
| FLAME, A2 | bigram + phonetic | all 23 | 1.00 | 0.43 | 0.61 | 10 / 23 |
| FLAME, A3 (auto_tune → ngram 4, n_out 1) | bigram + phonetic | 19 literal + weak-literal | 1.00 | 0.53 | 0.69 | 10 / 19 |
| FLAME, A3 | bigram + phonetic | all 23 | 1.00 | 0.43 | 0.61 | 10 / 23 |

At A0, the 13 detections are 12 gold pairs (11 of 13 literal, 1 of 6 weak literal, 0 of 4
non-literal) and one pair involving a background segment (Dk H-G1v / KCh bg_0301). The README
reports A0, which has the highest F1 of the four layer settings at this threshold.

### Table 2: ablation, all 23 pairs

Each configuration is scored at its own best-F1 threshold on the grid 0.05–0.40.

| Configuration | P | R | F1 | threshold |
|---|---|---|---|---|
| FLAME (vocab 35, ngram 5, n_out 2) | 0.92 | 0.52 | 0.67 | 0.28 |
| − gapping (n_out = 0, contiguous) | 0.69 | 0.48 | 0.56 | 0.12 |
| − fine tokens (vocab 200) | 0.50 | 0.30 | 0.38 | 0.11 |
| − subword (word-level, vocab ≈ 360) | 0.62 | 0.22 | 0.32 | 0.10 |

The last row is labeled as in the article. The configuration is BPE with `-vocab_size auto`, which
flame.py's morphology estimator resolves to 360 units on this corpus.

### Grid search

The 459 structural configurations (`vocab_size` ∈ {30–40, 45, 60, 80, 120, 200, auto} × `ngram` ∈
{2, 3, 4, 5, 6, 8, 10} × `n_out` ∈ {0, 1, 2, 4, 5}, n_out < ngram) are each scored at 36
thresholds (0.05–0.40). The best F1 on all 23 pairs, with ties broken by precision, is at vocab 35,
ngram 5, n_out 2, threshold 0.28 (F1 0.667). The operating point is therefore selected on the same
gold pairs that Table 1 reports; there is no held-out split.

## Threshold selection

Otsu is intended for corpora whose score distribution is bimodal; on this benchmark it is not, and
the article fixes the threshold at 0.28. `run_benchmark.py` therefore passes
`-similarity_threshold 0.28`.

## Task A: synthetic noise (`perturbation.py`)

Every KCh file (reuse chains and distractors) is perturbed at rate r ∈ {0, 0.05, …, 0.30}, seeds
42–46. At rate r, each word is edited with probability r by one operation drawn from those that
apply to it: a character substitution, insertion, or deletion; a u/v or i/j interchange; consonant
doubling or undoubling; or a swap to an orthographic variant attested in the gold pairs (16
spellings, found with flame.py's `classify_gap`). The Dk side is not perturbed. The full pipeline is
rerun for every run, rate, and seed, BPE training included.

FLAME (n_out 2) is scored at the fixed threshold 0.28. The contiguous variant (n_out 0) is scored at
0.12, where it reaches its best F1 on clean text (Table 2). In A3, auto_tune selects ngram 4,
n_out 1 for FLAME at every rate and seed. The contiguous variant takes the swept ngram (4) and keeps
n_out 0.

**Mean recall over the 23 gold pairs** (mean of 5 seeds):

| Run | System | r = 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|---|---|---|
| A0 none | FLAME | 0.52 | 0.50 | 0.47 | 0.44 | 0.41 | 0.39 | 0.43 |
| | contiguous | 0.48 | 0.44 | 0.43 | 0.43 | 0.41 | 0.37 | 0.37 |
| A1 bigram | FLAME | 0.39 | 0.38 | 0.37 | 0.35 | 0.36 | 0.36 | 0.37 |
| | contiguous | 0.39 | 0.39 | 0.38 | 0.39 | 0.37 | 0.35 | 0.35 |
| A2 bigram + phonetic | FLAME | 0.43 | 0.43 | 0.41 | 0.38 | 0.38 | 0.33 | 0.36 |
| | contiguous | 0.43 | 0.43 | 0.42 | 0.37 | 0.37 | 0.34 | 0.36 |
| A3 + auto_tune | FLAME | 0.43 | 0.43 | 0.42 | 0.37 | 0.38 | 0.34 | 0.38 |
| | contiguous | 0.48 | 0.47 | 0.47 | 0.45 | 0.44 | 0.39 | 0.42 |

**Mean precision at r = 0 and r = 0.30:**

| Run | FLAME, r = 0 | FLAME, r = 0.30 | contiguous, r = 0 | contiguous, r = 0.30 |
|---|---|---|---|---|
| A0 | 0.92 | 0.96 | 0.69 | 0.82 |
| A1 | 0.82 | 0.96 | 0.69 | 0.89 |
| A2 | 1.00 | 0.95 | 0.83 | 0.97 |
| A3 | 1.00 | 1.00 | 0.61 | 0.81 |

Across A0–A3, the mean recall difference FLAME − contiguous lies between −0.08 and +0.06 at every
rate. It is largest in A0 (mean over the rates +0.03), which is the run plotted below; the other
runs are the faint lines. With both systems at the same threshold, 0.28, the contiguous variant's
mean recall is 0.30 at r = 0 and 0.16 at r = 0.30 in A0, at precision 1.00 (columns
`recall_at_0_28` and `precision_at_0_28`).

![Recall vs. synthetic noise](results/perturbation.png)

## Alignment examples

![Alignment report, Regest 13](results/alignment_regest13.png)

![Alignment report, Regest 5](results/alignment_regest05.png)

The screenshots are cropped from the HTML report of

```bash
python flame.py -input_path <segments>/dk -input_path2 <segments>/kch -min_text_length 15 \
    -vocab_size 35 -ngram 5 -n_out 2 -char_norm_min_freq 2 -similarity_threshold 0.28
```

(Regest 13 from a run at `-similarity_threshold 0.53`, which keeps fewer pairs on the page). Cosines
at the operating point (`run.json`, `qualitative_example_cosines`): Dk G5r / KCh r141 0.5396, Dk
G4v-1 / KCh r141 0.5538, Dk G1r / KCh a5r5 0.3695, Dk G1r / KCh a5r4 0.2827. The Regest 5 report
classifies the words after the last matching block, which the HTML report does since the Unreleased
report change in `CHANGELOG.md`.

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

Expected layout after the first run (`data/` is git-ignored):

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
that subset (`literal_weak_19`) is the basis for comparing FLAME with them. `all_23` adds the four
non-literal pairs, which reuse sense without sharing surface form: Dk F5r2-1 / KCh a4v1-1
(Regest 4), Dk F6v1-1 and Dk F7r-1 / KCh b1r6-1 (Regest 21), and Dk F6v3-1 / KCh b1r7-1
(Regest 22). The regest numbers come from Ostrowski (2024c). The TEI file does not carry them, so
they were assigned by matching each segment's text against the passages each regest quotes (see
`REGEST_OF_NON_LITERAL` in `run_benchmark.py`). On the 19-pair set, a detection of a non-literal gold
pair is left out of scoring rather than counted as a false positive. At A0 none is detected
(cosines 0.08–0.19), so precision is the same on both sets.

A detected pair counts as a true positive if it is one of the gold pairs. Every other detected Dk–KCh
pair is a false positive, including a pair of two annotated segments from different regests.

## Settings

All runs pass `-min_text_length 15 -char_norm_min_freq 2 -vocab_min_word_freq 5 -vocab_coverage
0.85`. flame.py's default `min_text_length` of 150 would drop most segments, because the median
chain is 77 characters. Everything else is flame.py's `DEFAULT_PARAMS`, apart from the layer
switches of A1–A3 (`bigram_normalization_enabled`, `phonetic_reduction_enabled`, `auto_tune`) with
their default rules. The engine is called the way `flame.py`'s `main()` calls it:
`Flame.load_corpus()`, then `Flame.auto_tune_parameters()` when `auto_tune` is on, then
`Flame.compute_similarity_matrix()`. Nothing is patched.

## Baselines

The Passim (Smith et al. 2014), BLAST, and TextPAIR (Gladstone 2018) rows are quoted from
Ostrowski (2024a), not rerun; none of the three tools is part of this repository. According to the
article, Ostrowski ran them as open retrieval over the full texts and scored the literal and
weak-literal subset, and their recall values match that reading (5/19 = 0.26, 3/19 = 0.16).
`table1.csv` marks these rows `source = quoted`. TRACER (Büchler et al. 2014), which the article's
protocol also names, is not included: no TRACER figure is available for this benchmark.

## Run metadata (`results/run.json`)

Each run records the git commit and whether the benchmark code differed from it (`git_dirty`); the
Python version and platform; the versions of numpy, scipy, scikit-learn, scikit-image, tokenizers,
rapidfuzz, fargv, nltk, and tqdm; the seed; the wall-clock time; the dataset DOI, file name, and MD5;
the corpus counts; the fixed parameters; for every configuration, the metrics at its best-F1
threshold and at 0.28, with false positives named when there are at most 50; the cosines of the
alignment examples; the four non-literal pairs with their regests; and, with `--grid`, the grid
optimum. The published results were produced with Python 3.12.3, numpy 2.5.3, scipy 1.18.1,
scikit-learn 1.9.1, scikit-image 0.26.0, and tokenizers 0.23.2.

## References

Büchler, M., Burns, P. R., Müller, M., Franzini, E., & Franzini, G. (2014). Towards a historical text re-use detection. In C. Biemann & A. Mehler (Eds.), *Text Mining: From Ontology Learning to Automated Text Processing Applications* (pp. 221–238). Springer. https://doi.org/10.1007/978-3-319-12655-5_11

Gladstone, C. (2018). TextPAIR: A new high-performance sequence aligner. *ARTFL Project Research Blog*. https://artfl.blogspot.com/2018/12/textpair-new-high-performance-sequence.html

Kovács, T. (in preparation). Manuscript on FLAME, prepared for *Digital Scholarship in the Humanities*.

Ostrowski, A. (2024a). Ansätze und Tools für Historische Text Reuse Detection: Fragmentierter Text Reuse am Beispiel ripuarischer Inkunabeln des 15. Jahrhunderts. In *DHd 2024, Conference Abstracts*. Passau.

Ostrowski, A. (2024b). *Evaluation Set for Historical Text Reuse Detection*. Zenodo. https://doi.org/10.5281/zenodo.13378221

Ostrowski, A. (2024c). *Dokumentation von Parallelstellen zwischen dem 'Dornenkranz von Köln' und der 'Koelhoffschen Chronik'*. Zenodo. https://doi.org/10.5281/zenodo.14410766

Smith, D. A., Cordell, R., Dillon, E. M., Stramp, N., & Wilkerson, J. (2014). Detecting and modeling local text reuse. In *Proceedings of the IEEE/ACM Joint Conference on Digital Libraries (JCDL)* (pp. 183–192). London: IEEE.
