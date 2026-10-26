# All CLI arguments

Arguments use a single leading dash (`-input_path`, not `--input_path`), in either the
`-param value` or `-param=value` form. Booleans are presence switches (see the
[README](../README.md#command-line-interface-cli)). `python flame.py -h` prints the same list with
the defaults, which live in `DEFAULT_PARAMS` in `flame.py`.

| Parameter | Default | Description |
| --- | --- | --- |
| `input_path` | `''` | **Required.** Path to the primary corpus: a directory, or a glob pattern such as `./fsdb/DE-LANRWR**/*.htr.txt`. |
| `input_path2` | `''` | Optional second corpus (directory or glob pattern) for cross-corpus comparison. |
| `file_suffix` | `.txt` | File extension of text documents to process. Ignored when `input_path` is a glob pattern, which then decides alone which files are read. |
| `keep_texts` | `10000` | Maximum number of texts to load from each directory. |
| `deduplicate` | `False` | Skip files whose text is identical to one already loaded, keeping the shortest name. Off by default: whether two identical files are two witnesses or one is a scholarly judgement. The two sides of an inter-corpus run are deduplicated independently, so a document shared by both is still compared across them. |
| `ngram` | `6` | The size of the n-gram window for feature generation. |
| `n_out` | `1` | Number of tokens to "leave out" (drop) from each n-gram window. |
| `min_text_length` | `150` | Minimum character length for a file to be included in the corpus. |
| `similarity_threshold` | `'auto'` | Similarity cutoff score. Can be a float (e.g., `0.75`) or `'auto'`. |
| `auto_threshold_method` | `'otsu'` | Method for auto-thresholding: `'otsu'` or `'percentile'`. |
| `char_norm_alphabet` | `abcdef...` | String of allowed lowercase base characters for normalization. |
| `char_norm_strategy` | `'normalize'` | Strategy for handling unknown out-of-alphabet characters. |
| `char_norm_min_freq` | `1` | Minimum frequency for the adaptive normalizer to register an automated Unicode rule. |
| `phonetic_reduction_enabled` | `False` | Enables the phonetic reduction layer (character mapping to a reduced alphabet). |
| `phonetic_reduction_alphabet` | `aefiklmno...` | Target reduced alphabet for phonetic mapping. Characters outside this set become spaces. |
| `phonetic_reduction_rules` | `b>p,c>k,...` | Comma-separated phonetic mapping rules in `src>dst` format (e.g., `b>p,c>k`). |
| `bigram_normalization_enabled` | `False` | Enables bigram normalization (multi-character to single-character replacement). |
| `bigram_normalization_rules` | `ss>s,ff>f,...` | Comma-separated bigram rules in `src>dst` format (e.g., `ss>s,ie>i,au>u`). Source must be 2+ chars, destination exactly 1. |
| `vocab_size` | `'auto'` | Target subword vocabulary size. An integer, or `'auto'` to calculate via morphology. |
| `vocab_min_word_freq` | `5` | Minimum frequency for a word to be evaluated for affix candidates. |
| `vocab_coverage` | `0.85` | Desired morphological coverage percentage of the corpus when `vocab_size` is `'auto'`. |
| `fuzz_threshold` | `0.70` | Orthographic-variant vs. bridge cutoff (0-1): words between two matches whose spelling-folded forms score at least this high are an *Orthographic Variant*, below it a genuine *Bridge*. Measured on real two-copies comparisons, genuine divergences score ≤ 0.40 and spelling variants ≥ 0.72. |
| `max_gap_words` | `5` | Maximum structural token length allowed inside an individual non-matching gap segment. Larger gaps are left unmarked rather than classified. |
| `cluster_threshold` | `0.70` | Edit-similarity cutoff (0-1) at which two pairs' shared formulas count as *the same* formula, judged on the formula rather than the whole charter. `1.0` gives verbatim-only clusters. `0.70` is the default rather than KONI's `0.85`, because on the MOM corpus `0.85` split one papal privilege template into clusters of 24, 12, 6 and 2 charters over differing recipients, and `0.70` merges them with no false merge among clusters of four or more. `0.65` starts over-merging. |
| `cluster_min` | `2` | Minimum number of pairs a cluster must contain to be reported. |
| `cluster_linkage` | `'louvain'` | How the edit-similarity graph becomes clusters. `louvain` (default) finds modularity communities — cores denser among themselves than with the rest — keeping one template whole while refusing to chain two formulas that merely share a phrase; it needs `networkx`. `clique` splits each group into **maximal cliques**, so every formula matches every other one, but a chained group is broken apart and a formula between two groups appears in **both** clusters. `union` is KONI's original union-find, which chains. Not a boolean on purpose: `-flag` arguments are presence switches, so a `True`-by-default boolean could not be turned off from the command line. |
| `core_gap_tolerance` | `8` | How wide a gap the core aligner may bridge, in tokens, on either side. Strict word-for-word matching breaks a formula in two at the first inserted word (`uestro auctoritate` against `auctoritate` was enough); eight tokens spans the widest slot measured in the corpus without bridging two genuinely different formulas. |
| `core_min_tokens` | `12` | Shortest aligned window (its shorter side) that still counts as a formula. Below it the pair falls back to the longest contiguous run; the run reports how many took that path. |
| `core_anchors` | `''` | Where a formula is looked for: the performative verbs that carry a charter's legal act (`donamus`, `contulimus`, `confirmamus`…), as comma-separated **stems** prefix-matched against the folded tokens. Without them the alignment maximises matched tokens, and the longest shared run in a mediaeval charter is the protocol — on the MOM corpus no core came out below 56 tokens (median 208) and 28 of 40 clusters opened on a protocol phrase. A pair whose shared text carries no anchor keeps the unanchored window. **Empty by default, and there is no built-in list** — see [Clustering](CLUSTERING.md) for the measurements behind that. `-core_anchors=donau,contul,tradid` supplies them; `-core_anchors=` (the `=` form — fargv crashes on a bare empty argument) and `-core_anchors none` both select the unanchored extraction. |
| `core_anchor_window` | `15` | How far the window may extend past the anchor, in tokens, on either side. |
| `core_max_tokens` | `50` | Hard ceiling on the reported window, in tokens; the window is centred on the anchor and clamped inside the aligned span. A dispositio runs 15–40 words while a protocol panel runs to 200, so without a ceiling a long generic overlap outvotes a short specific one. `0` leaves the window uncapped. The near-duplicate test reads the pair's whole shared span, not this window. |
| `core_identity_threshold` | `0.0` | Minimum identity (`2 × matched / (len_a + len_b)`, the same normalisation as `cluster_threshold`) for an aligned window to be accepted. `0.0` accepts every window the aligner returns; raise it to demand a tighter formula and send the loose pairs back to the contiguous rule. |
| `min_duplicate_tokens` | `400` | A cluster is reported as a **near-duplicate charter** rather than a formula when its shared window is at least this many words **and** covers at least 60% of the shorter charter — both conditions, see [Clustering](CLUSTERING.md). |
| `min_core_specificity` | `0.0` | Drop clusters whose `Specificity` (`CoreTokens` × `MeanIDF`) is below this; `0.0` keeps everything. The knob for universal chancery phrases: a formula every charter carries has a low mean IDF whatever its length, so `10`–`15` keeps the specific formulas and discards the generic scaffolding. Filtering happens before cluster IDs are handed out, so the `ClusterID` columns of the other TSVs stay consistent. |
| `stopwords_file` | `''` | Path to a plain-text list of tokens, one per line. Listed tokens score IDF `0`, so a formula built only from them scores `Specificity 0` and `min_core_specificity` can remove it — the hand-written counterpart to the data-driven IDF. |
| `auto_tune` | `False` | Enables self-supervised parameter discovery via temporary synthetic noise sweeps. |
| `auto_tune_sample_size` | `30` | Number of document vectors to isolate and sample when executing an `auto_tune` sweep. |
| `no_reports` | `False` | If True, skips generating user-facing visual summaries and reports completely. |
| `gen_comparison_html` | `True` | Generate interactive side-by-side HTML comparison reports. |
| `gen_summary_tsv` | `True` | Generate a TSV summary of related documents and matching segments. |
| `gen_linguistic_tsv` | `True` | Generate a granular TSV of linguistic variations (alternative spellings, substitutions). |
| `gen_heatmap` | `True` | Generate an interactive Plotly similarity heatmap HTML file. Skipped, with a printed note, when the similarity matrix reaches 2000 documents in either dimension — a corpus that large is read better from the cluster and pair reports. |
| `gen_clusters` | `True` | Generate the clustering (recurring legal formulas) report. **Also adds a `ClusterID` column to `similarity_summary.tsv` and `linguistic_variations.tsv`** — see the note under [Outputs](../README.md#outputs). |

## Command-Line Interface (CLI)

Arguments use a **single leading dash** (`-input_path`, not `--input_path`), in either
the `-param value` or `-param=value` form.

Boolean switches are **presence switches**: `-auto_tune` on its own turns the option on, and so do
`-auto_tune False`, `-auto_tune false` and `-auto_tune 0` — the value is ignored. A boolean that
defaults to `True` (the `gen_*` report switches) therefore cannot be switched off from the command
line; use the GUI checkbox or change the default in `DEFAULT_PARAMS` in `flame.py`.

To see all available options and their defaults, run:

```bash
python flame.py -h

```

**Example (Using Auto-Tune):**

```bash
python flame.py -input_path ./path/to/texts -auto_tune True -similarity_threshold auto

```

**Example (A subset of a nested corpus, given as a pattern):**

```bash
python flame.py -input_path './fsdb/DE-LANRWR**/*.htr.txt' -similarity_threshold 0.60

```

## Input Paths, Patterns, and How Files Are Named

`input_path` (and `input_path2`) accepts either a **directory** or a **glob pattern**:

- **A directory** is searched recursively for every file ending in `file_suffix` (`*.txt` by default), at any depth.
- **A glob pattern** is expanded as-is and decides alone which files are read — `file_suffix` is deliberately **not** applied on top of it, so a pattern like `*.htr` is not silently emptied by a `.txt` filter.

This matters for corpora that nest one folder per document and reuse the same filename everywhere (an
*fsdb* tree of `DE-LANRWR001/text.htr.txt`, `DE-LANRWR002/text.htr.txt`, …). A bare filename would be
ambiguous there, so **every report names a text by its path relative to the input root**, giving
`DE-LANRWR001/text.htr.txt`. When the input is flat, that relative path *is* the filename, so existing
outputs are unchanged.

Two further properties:

- **The file list is sorted**, so corpus order — and with it the row/column order of the distance matrix and the heatmap — is reproducible rather than following the filesystem.
- **`**` is only recursive as a standalone path component.** `./fsdb/DE-LANRWR**/*.htr.txt` reads the `**` exactly like `*`, descending a single level, which is what the fsdb layout needs. For unbounded depth write `./fsdb/**/*.htr.txt`. FLAME prints a note when it sees a doubled star that is not a standalone component.

## Graphical User Interface (GUI)

The window is organised into three tabs:

1. **Data & Core Setup** — the primary corpus path and an optional secondary one for inter-corpus runs (each accepts a folder or a glob pattern), `file_suffix`, `keep_texts`, `min_text_length`, `ngram` / `n_out`, the similarity threshold (a number or `auto`) with its method dropdown, and the duplicate-file switch.
2. **Normalization & Philology** — the character-normalization strategy and its minimum frequency, the bigram and phonetic reduction layers, the BPE vocabulary settings, and `fuzz_threshold`, `max_gap_words`, `cluster_threshold`, `cluster_min` and `min_core_specificity`.
3. **Auto-Tune & Outputs** — the auto-tune switch and its sample size, one checkbox per generated report, the `cluster_linkage` dropdown, and the run button.

The pipeline runs on a worker thread, so the window stays responsive. Its stdout and stderr are
streamed live into the *Pipeline Execution Standard Output Log* pane, where the run's summary lines
appear (`Anchoring: …`, `Duplicate links: …`, the clustering statistics), and when the run finishes
one button per generated report opens it directly.

**Not every parameter has a GUI field.** The GUI carries a widget for 36 of the 43 parameters; the
seven it does not expose are exactly the clustering internals — `core_anchors`, `core_anchor_window`,
`core_max_tokens`, `core_gap_tolerance`, `core_min_tokens`, `core_identity_threshold` and
`min_duplicate_tokens`. Those keep their `DEFAULT_PARAMS` value, so the GUI and the CLI agree on the
defaults; changing one means the command line or editing `flame.py`. The boolean `gen_*` switches are
the mirror image: the command line can turn those *on*, never off — see
[Command-Line Interface](#command-line-interface-cli).

## Standalone Application (PyInstaller)

`FlameApp.spec` packages the GUI and the pipeline behind it into a self-contained folder, so the tool
can be handed to a philologist without a Python environment:

```bash
pip install pyinstaller
pyinstaller FlameApp.spec
```

The result is `dist/FlameApp/`, a **onedir** bundle (`COLLECT`) rather than a single file, so the
executable does not unpack itself on every start. Two spec details are deliberate: Plotly's
`package_data` is bundled as data, which the offline heatmap needs at runtime, and `console=True` is
kept on so a failure inside a packaged run is readable in the launching terminal — set it to `False`
for a windowless build once the packaging is known good.
