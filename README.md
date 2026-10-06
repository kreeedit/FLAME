# FLAME: Formulaic Language Analysis in Medieval Expressions
 

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**FLAME** is a Python-based tool with both **Command-Line (CLI)** and **Graphical (GUI)** interfaces, designed for identifying and analyzing formulaic language and text reuse, particularly in historical corpora like medieval charters. It uses a **Leave-N-Out (LNO) n-gram** approach, which is highly effective for detecting variant forms of expressions that differ due to scribal variations, regional dialects, or other textual modifications. It automatically learns normalization rules from the corpus itself (handling medieval ligatures and special characters) and uses subword tokenization to absorb rare words and morphological variants. It automatically suggests an optimal vocabulary size for the tokenizer based on the corpus's statistical properties, offers an autonomous **Self-Supervised Auto-Tune** engine to discover ideal window properties, and automatically determines an optimal similarity cutoff score using Otsu's method.

A downloadable demo of the HTML output can be found in the repository (`text_comparisons.html`).

<p align="center">
  <img src="flame-little-flame.gif" width="200" alt="FLAME animation" />
</p>

## How It Works

The LNO-gram approach systematically creates robust features from text. For a given sequence of words (an n-gram), it generates multiple variants by omitting a specified number of tokens. This allows the system to identify underlying similarities even if the surface forms are not identical.

Consider the medieval charter opening: *"In nomine sancte et individue trinitatis amen"*

1.  **Generate n-grams**: The tool slides a window of a specified length (e.g., 5 words) across the text.
2.  **Create LNO variants**: For each 5-gram, it creates subsequences by removing a specified number of tokens (e.g., 1). For the 5-gram `[In, nomine, sancte, et, individue]`, it would generate features like `[_, nomine, sancte, et, individue]`, `[In, _, sancte, et, individue]`, etc.
3.  **Hashing**: Each variant is converted into a unique, memory-efficient integer hash using a vectorised polynomial rolling hash.
4.  **Similarity Calculation**: The tool calculates the cosine similarity between documents based on the frequency of these shared sparse feature hashes, scaled via TF-IDF.
5.  **Visualization**: Results are presented in interactive reports that highlight matching patterns in their original context with browser-side adjustments.

### Method Comparison

The LNO-gram method offers a balance of context-preservation and flexibility that is often superior to traditional n-grams or skip-grams for historical text analysis.

| Method | Input Text | Subword Tokens (Example) | Generated Patterns (Examples) | Match Score | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **N-gram** | `In nomine sancte et individue` | `[' In', ' nomine', ' sancte', ' et', ' individue']` | `[In nomine sancte et individue]` | 1.0 | Rigid. Fails if a single word changes (e.g., `dei` for `nomine`). |
| (n=5) | `In dei nomine sancte et` | `[' In', ' dei', ' nomine', ' sancte', ' et']` | `[In dei nomine sancte et]` | 0.0 | No tolerance for variation. |
| **Skip-gram**| `In nomine sancte et individue` | `[' In', ' nomine', ' sancte', ' et', ' individue']` | `[In sancte]`, `[nomine et]` | ~0.4 | Loses word order and creates noisy, out-of-context pairs. |
| (n=2, k=1) | `In dei nomine sancte et` | `[' In', ' dei', ' nomine', ' sancte', ' et']` | `[In nomine]`, `[dei sancte]` | ~0.3 | High noise, low contextual accuracy. |
| **FLAME** | `In nomine sancte et individue` | `[' In', ' nom', 'ine', ' sanct', 'e', ' et', ' in', 'di', 'vid', 'ue']` | `[nomine sancte et _]`, `[In _ sancte et individue]`... | ~0.95 | **High flexibility.** Captures whole-word and sub-word variations. |
| **(LNO-gram + Subword)** | `In dei nomine sancte et` | `[' In', ' dei', ' nom', 'ine', ' sanct', 'e', ' et']` | `[dei nomine sancte et _]`... | ~0.90 | **Robust.** Effectively matches even with novel words or spellings by comparing their constituent parts. |

*Where `n` is the window size, `k` is the number of skips, and `r` is the number of removed tokens. Match scores are illustrative.*

---

## Key Features

-   **Advanced LNO-gram Analysis**: Systematically generates partial matches by removing combinations of tokens from traditional n-grams.
-   **Autonomous Parameter Auto-Tuning**: Features a self-supervised "trial digging" engine that injects synthetic transcription/dialect noise into a sample of your text to automatically find the optimal `ngram` and `n_out` setup for your specific data.
-   **Adaptive Character Normalization**: Autonomously learns and applies normalization rules (e.g., `é` -> `e`, MUFI ligatures) using rapid, vectorized NumPy lookup views over the Unicode Basic Multilingual Plane.
-   **Bigram Normalization**: Optional preprocessing layer that collapses common doubled consonants and diphthongs (e.g., `ss`→`s`, `ie`→`i`, `au`→`u`) via configurable multi-character to single-character replacement rules, reducing orthographic noise before further processing.
-   **Phonetic Reduction Layer**: Optional rule-based phonetic character mapping (e.g., `b`→`p`, `c`→`k`, `v`→`f`) that reduces the character set to a configurable target alphabet, inspired by Metaphone/Soundex principles for handling scribal variation in medieval texts.
-   **BPE Subword Tokenization**: Employs a Byte-Pair Encoding tokenizer with automatically suggested vocabulary size based on corpus morphology, absorbing rare words and orthographic variants into shared subword units.
-   **Inter-Corpus Comparison**: Supports two-directory mode (`input_path2`) for cross-collection similarity analysis between distinct corpora.
-   **Flexible Corpus Selection**: Takes either a directory (searched recursively) or a glob pattern, so you can analyse a subset of a large corpus — e.g. `./fsdb/DE-LANRWR**/*.htr.txt` — without copying files around.
-   **Unambiguous File Identity**: Reports name every text by its path relative to the input root, so deeply nested corpora that reuse the same filename per folder (as *fsdb* does) stay distinguishable. Flat corpora are unaffected, since there the relative path is the filename.
-   **Duplicate Detection (optional)**: Corpora assembled from editions, archival copies and re-downloads routinely hold the same charter under several names (`CSGIII_1004_VI_17.txt`, `CSGIII_1004_VI_17 (1).txt`, `CH-StiASG_UNDATED_Urkunden_A.1.A.13.txt`). By default FLAME compares every file it finds, exactly as a scholar would expect from what is on disk; with `deduplicate` on it skips files whose text it has already loaded, keeps the shortest-named copy, and prints every file it dropped and why.
-   **Automatic Threshold Detection**: Intelligently determines the optimal similarity threshold using Otsu's method on non-zero sparse data, removing manual guesswork.
-   **Clustering (recurring legal formulas)**: Groups the related pairs by the actual formula they share — the shared wording found by **gapped local alignment**, seeded by default on the **performative verb** that carries the legal act (`donamus`, `contulimus`, `confirmamus`…), so a cluster is built on the transaction rather than on the protocol panel every charter inherits — and a cluster answers *"these charters use the same legal formula / record the same legal transaction"*, not merely *"these charters are similar"*. Grouping happens on the spelling-folded wording, so `vnd`/`und` and other medieval variants of one formula stay one cluster, and a pair that shares enough text to be a copy of a charter is kept in the report whatever its core looks like. See [Clustering](#clustering-recurring-legal-formulas).
-   **Multi-Format Reporting**: Generates interactive side-by-side HTML comparisons with dynamic fuzzy sliders, a similarity heatmap, a TSV summary of related documents, and a granular linguistic variations TSV capturing alternative spellings and lexical substitutions.
-   **Modern Tabbed Interface (GUI)**: Built with a clean, beginner-friendly tabbed layout (`ttk.Notebook`) to separate data configurations, philological fine-tuning, and execution reporting.
-   **Dynamic Client-Side Highlights**: Side-by-side alignment outputs include interactive HTML/JS sliders, letting you change the fuzzy match sensitivity for structural bridge words on the fly inside your web browser.
-   **High Performance & Scalability**: Handles heavy historical corpora by utilizing memory-efficient sparse matrices and fast matrix-vector products.

---

## Installation

requirements: python 3.9 min, pip 26.1 min

It is highly recommended to use a Python virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kreeedit/FLAME
    cd FLAME
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:** Run the following command in a Python interpreter to download the necessary tokenizer models.
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('punkt_tab')
    ```

### Standalone Application (PyInstaller)

`FlameApp.spec` packages the GUI (and the whole pipeline behind it) into a self-contained
folder, so the tool can be handed to a philologist without a Python environment:

```bash
pip install pyinstaller
pyinstaller FlameApp.spec
```

The result is `dist/FlameApp/` — a **onedir** bundle (`COLLECT`), not a single file, so the
executable starts without unpacking itself each time. Two details in the spec are deliberate:
Plotly's `package_data` is bundled as data, which the offline heatmap needs at runtime, and
`console=True` is kept on so a failure inside a packaged run is readable in the terminal it was
launched from — set it to `False` for a windowless build once the packaging is known good.

## Usage

You can run the analysis using either the GUI or the CLI.

### Graphical User Interface (GUI)

The GUI provides an intuitive way to set the pipeline parameters, execute autonomous tuning sweeps, and monitor progress across tab containers.

<p align="center">
  <img src="flame_gui.png" width="400" />
</p>

To launch the graphical interface, run:
```bash
python flame_gui.py

```

The window is organised into three tabs:

1. **Data & Core Setup** — the primary corpus path and an optional secondary one for inter-corpus runs (each accepts a folder or a glob pattern), `file_suffix`, `keep_texts`, `min_text_length`, `ngram` / `n_out`, the similarity threshold (a number or `auto`) with its method dropdown, and the duplicate-file switch.
2. **Normalization & Philology** — the character-normalization strategy and its minimum frequency, the bigram and phonetic reduction layers, the BPE vocabulary settings, and the philological knobs: `fuzz_threshold` (variant vs. bridge), `max_gap_words`, `cluster_threshold`, `cluster_min` and `min_core_specificity`.
3. **Auto-Tune & Outputs** — the auto-tune switch and its sample size, one checkbox per generated report, the `cluster_linkage` dropdown, and the run button.

The pipeline runs on a worker thread, so the window stays responsive. Its stdout and stderr are streamed live into the *Pipeline Execution Standard Output Log* pane — which is where the run's summary lines appear, including `Anchoring: …`, `Duplicate links: …` and the clustering statistics — and when the run finishes, one button per generated report opens it directly.

**Not every parameter has a GUI field.** The GUI carries a widget for 36 of the 43 parameters; the seven it does not expose are exactly the clustering internals — `core_anchors`, `core_anchor_window`, `core_max_tokens`, `core_gap_tolerance`, `core_min_tokens`, `core_identity_threshold` and `min_duplicate_tokens`. Those keep their `DEFAULT_PARAMS` value, which the GUI passes through unchanged, so the GUI and the CLI agree on the defaults; changing one means the command line, or editing `DEFAULT_PARAMS` in `flame.py`. An analysis that needs a different anchor list or a different window is therefore run from the CLI. (The boolean `gen_*` switches are the mirror image: the command line can turn those *on*, never off — see [Command-Line Interface](#command-line-interface-cli).)

### Command-Line Interface (CLI)

Arguments use a **single leading dash** (`-input_path`, not `--input_path`), in either
the `-param value` or `-param=value` form.

Boolean switches are **presence switches**: `-auto_tune` on its own turns the option on, and writing
`-auto_tune False`, `-auto_tune false` or `-auto_tune 0` after it also turns it *on* — the value is
ignored. A boolean that defaults to `True` (the `gen_*` report switches) therefore cannot be switched
off from the command line; use the GUI checkbox, or change the default in `DEFAULT_PARAMS` in
`flame.py`.

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

### Input Paths, Patterns, and How Files Are Named

`input_path` (and `input_path2`) accepts either a **directory** or a **glob pattern**:

- **A directory** is searched recursively for every file ending in `file_suffix`
  (`*.txt` by default), at any depth.
- **A glob pattern** is expanded as-is, and then decides alone which files are read —
  `file_suffix` is deliberately **not** applied on top of it, so a pattern like
  `*.htr` is not silently emptied by a `.txt` filter.

This matters for corpora that nest one folder per document and reuse the same filename
everywhere (for example an *fsdb* tree of `DE-LANRWR001/text.htr.txt`,
`DE-LANRWR002/text.htr.txt`, …). Because a bare filename would be ambiguous there,
**every report names a text by its path relative to the input root**, so the above comes
out as `DE-LANRWR001/text.htr.txt`. When the input is flat, that relative path *is* the
filename, so existing outputs are unchanged.

Two further properties worth knowing:

- **The file list is sorted**, so corpus order — and with it the row/column order of the
  distance matrix and the heatmap — is reproducible instead of following whatever order
  the filesystem happened to return.
- **`**` is only recursive as a standalone path component.** `./fsdb/DE-LANRWR**/*.htr.txt`
  reads the `**` exactly like `*`, i.e. it descends a single level — which is what the
  fsdb layout needs. For unbounded depth, write `./fsdb/**/*.htr.txt`. FLAME prints a note
  when it sees a doubled star that is not a standalone component.

### All CLI Arguments

| Parameter | Default | Description |
| --- | --- | --- |
| `input_path` | `''` | **Required.** Path to the primary corpus: a directory, or a glob pattern such as `./fsdb/DE-LANRWR**/*.htr.txt`. |
| `input_path2` | `''` | Optional second corpus (directory or glob pattern) for cross-corpus comparison. |
| `file_suffix` | `.txt` | File extension of text documents to process. Ignored when `input_path` is a glob pattern, which then decides alone which files are read. |
| `keep_texts` | `10000` | Maximum number of texts to load from each directory. |
| `deduplicate` | `False` | Skip files whose text is identical to one already loaded. Off by default: whether two identical files are two witnesses or one is a scholarly judgement. Among identical files the shortest name is kept. Two corpora in an inter-corpus run are deduplicated independently, so a document shared by both sides is still compared across them. |
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
| `vocab_size` | `'auto'` | Target subword vocabulary size. Can be an integer or `'auto'` to calculate via morphology. |
| `vocab_min_word_freq` | `5` | Minimum frequency for a word to be evaluated for affix candidates. |
| `vocab_coverage` | `0.85` | Desired morphological coverage percentage of the corpus when `vocab_size` is `'auto'`. |
| `fuzz_threshold` | `0.70` | Orthographic-variant vs. bridge cutoff (0-1). Words between two matches whose spelling-folded forms score at least this high are reported as an *Orthographic Variant*; below it they are a genuine *Bridge*. Calibrated on real two-copies comparisons, where genuine divergences score ≤ 0.40 and spelling variants ≥ 0.72. |
| `max_gap_words` | `5` | Maximum structural token length allowed inside an individual non-matching gap segment. Larger gaps are left unmarked rather than classified. |
| `cluster_threshold` | `0.70` | Edit-similarity cutoff (0-1) at which two pairs' shared formulas count as *the same* formula. Judged on the formula, not on the whole charter: two charters may share a formula nearly verbatim while differing everywhere else. Set it to `1.0` for verbatim-only clusters. The default is `0.70`, not KONI's `0.85`: measured on the MOM corpus, `0.85` split one papal privilege template into four clusters of 24, 12, 6 and 2 charters purely because the recipients differed (an abbey against a hospital), while `0.70` merges them into one 49-charter cluster and adds no false merge among the clusters of four or more members. `0.65` starts over-merging. |
| `cluster_min` | `2` | Minimum number of pairs a cluster must contain to be reported. `2` keeps every formula found in at least two places. |
| `cluster_linkage` | `'louvain'` | How the edit-similarity graph is turned into clusters. `louvain` (the default) finds modularity communities — groups of cores that are denser among themselves than with the rest — which keeps one template in one piece while still refusing to chain two formulas that merely share a phrase. `clique` is strict: each group is re-scored pairwise and split into **maximal cliques**, so every formula in a cluster matches every other one at `cluster_threshold`, but a chained group (A~B, B~C, but A≁C) is broken apart and a formula sitting between two groups appears in **both** clusters (each card names the other under `shares pair(s) with cluster …`). `union` is KONI's original union-find answer, which chains. `louvain` needs `networkx`. Not a boolean on purpose: `-flag` arguments are presence switches, so a `True`-by-default boolean could not be turned off from the command line. |
| `core_gap_tolerance` | `8` | How wide a gap the core aligner may bridge, in tokens, on either side. A mediaeval formula carries variable slots (a name, a place, a case ending), and strict word-for-word matching breaks the formula in two at the first inserted word — `uestro auctoritate` against `auctoritate` was enough. Eight tokens spans the widest slot measured in the corpus without bridging two genuinely different formulas. |
| `core_min_tokens` | `12` | Shortest aligned window (its shorter side) that still counts as a formula. Below it the pair falls back to the longest contiguous run, which is what the pipeline did before gapped alignment, and the run reports how many pairs took that path. |
| `core_anchors` | `''` | Where a formula is looked for: the performative verbs that carry a charter's legal act (`donamus`, `contulimus`, `confirmamus`…), as comma-separated **stems** prefix-matched against the folded tokens. Without them the alignment maximises matched tokens, and the longest shared run in a mediaeval charter is the protocol — measured on the MOM corpus, no core came out below 56 tokens (median 208) and 28 of 40 clusters opened on a protocol phrase. A pair whose shared text carries no anchor keeps the unanchored window, and the report counts how many did. **Empty by default, and there is no built-in list**: see [Where the window sits](#clustering-recurring-legal-formulas) for the measurements that removed it. `-core_anchors=donau,contul,tradid` supplies them for a run; `-core_anchors=` (the `=` form — fargv crashes on a bare empty argument) and `-core_anchors none` both select the unanchored extraction. |
| `core_anchor_window` | `15` | How far the window may extend past the anchor, in tokens, on either side. Fifteen is the length of the act's habitual surroundings (the operative clause and the sanctio after it) without reaching the arenga. |
| `core_max_tokens` | `50` | Hard ceiling on the reported window, in tokens; the window is centred on the anchor and clamped inside the aligned span. The legal act is short (a dispositio runs 15–40 words) while a protocol panel runs to 200, so without a ceiling a long generic overlap still outvotes a short specific one. `0` leaves the window uncapped. The near-duplicate test does **not** read this window — it reads the pair's whole shared span (`overlap`), which the ceiling leaves untouched — and since the pair-level duplicate rule is wired into the graph as a forced edge (see [Formulas versus copies](#clustering-recurring-legal-formulas)), a narrowed core cannot cost a copy of a charter its place in the report any more. That was not true before that link existed: anchored runs used to leave six charter documents sharing 400+ words in no cluster at all. The classification is not otherwise immune — the cores are the graph's nodes, so a narrower core still re-wires the graph and can move a document between families. |
| `core_identity_threshold` | `0.0` | Minimum identity (`2 × matched / (len_a + len_b)`, the same normalisation as `cluster_threshold`) for an aligned window to be accepted. `0.0` accepts every window the aligner returns; raise it to demand a tighter formula and send the loose pairs back to the contiguous rule. |
| `min_duplicate_tokens` | `400` | A cluster is reported as a **near-duplicate charter** rather than a formula when its shared window is at least this many words **and** covers at least `MAX_CORE_FRACTION` (60%) of the shorter charter. Both conditions are needed — see [Clustering](#clustering-recurring-legal-formulas). |
| `min_core_specificity` | `0.0` | Drop clusters whose `Specificity` (`CoreTokens` × `MeanIDF`) is below this. `0.0` keeps everything. This is the knob for the universal chancery phrases — a formula every charter carries scores a low mean IDF whatever its length, so a threshold around `10`–`15` leaves the specific formulas (papal *protectione suscipimus* clauses, named penalties) and discards the generic scaffolding. Filtering happens before cluster IDs are handed out, so the `ClusterID` column of the other TSVs stays consistent. |
| `stopwords_file` | `''` | Path to a plain-text list of tokens, one per line. Listed tokens score IDF `0`, so a formula built only from them scores `Specificity 0` and can be removed with `min_core_specificity`. This is the hand-written counterpart to the data-driven IDF: a diplomatic list (`et`, `in`, `de`, `sancte`, …) encodes what the editor already knows is not distinctive. Empty by default. |
| `auto_tune` | `False` | Enables self-supervised parameter discovery via temporary synthetic noise sweeps. |
| `auto_tune_sample_size` | `30` | Number of document vectors to isolate and sample when executing an `auto_tune` sweep. |
| `no_reports` | `False` | If True, skips generating user-facing visual summaries and reports completely. |
| `gen_comparison_html` | `True` | Generate interactive side-by-side HTML comparison reports. |
| `gen_summary_tsv` | `True` | Generate a TSV summary of related documents and matching segments. |
| `gen_linguistic_tsv` | `True` | Generate a granular TSV of linguistic variations (alternative spellings, substitutions). |
| `gen_heatmap` | `True` | Generate an interactive Plotly similarity heatmap HTML file. The heatmap is skipped, with a printed note, when the similarity matrix reaches 2000 documents in either dimension — a corpus that large is read better from the cluster and pair reports. |
| `gen_clusters` | `True` | Generate the clustering (recurring legal formulas) report. **Also adds a `ClusterID` column to `similarity_summary.tsv` and `linguistic_variations.tsv`** — see the note under [Outputs](#outputs). |

---

## Outputs

A run produces up to six kinds of report in the directory where it was started; which of them appear
depends on the `gen_*` switches and, for the heatmap, on the size of the corpus (see below). In all
of them a document is identified by its **path relative to the input root** (see
[Input Paths, Patterns, and How Files Are Named](#input-paths-patterns-and-how-files-are-named)).

1. **`dist_mat.npz`**: A SciPy sparse matrix file containing all pairwise similarity scores. Essential for downstream validation without re-computing features. **Where it lands depends on how FLAME was started:** the GUI writes it — along with the other intermediates (`temp_corpus.txt`, `bpe_tokenizer.json`) — into the directory it was launched from, while a CLI run keeps them in a temporary directory that is **removed when the run ends**. A command-line run therefore leaves the six reports below and nothing else; take the matrix from a GUI run, or from the engine's own API with an explicit working directory, if a downstream script needs it.
2. **`text_comparisons_XX.html`**: Interactive side-by-side alignment report files. This is the primary visualization engine for philological exploration. Features include:
* Synchronized scroll-locking and text matching cross-highlights.
* A **Live Fuzzy Slider** to dynamically adjust, in the browser, which structural bridges count as similar.
* A **three-way classification** of the words sitting between two matches, shown by colour and explained by an on-page legend:
  * **Bridge word** (yellow) — the two texts genuinely diverge here.
  * **Orthographic variant** (green) — the same wording spelled differently, e.g. `deßhalb` / `deshalb`.
  * **Insertion** (blue) — the wording is present in one of the two texts only.
* Directional layout control which places earlier documents on the left based on filename year markers.


3. **`similarity_heatmap.html`**: An interactive Plotly heatmap visualizing the full pairwise similarity matrix, useful for spotting clusters of related documents at a glance. It is generated only while both matrix dimensions stay **below 2000 documents**; above that the run prints `Skipping heatmap generation for large matrix` and produces no file, so a large corpus still needs `gen_heatmap` off to avoid a misleading expectation.

4. **`similarity_summary.tsv`**: A spreadsheet summary detailing related matches, document frequencies, and prominent, long-standing matching blocks (>4 words).
5. **`linguistic_variations.tsv`**: A structured corpus-wide register of what sits inside identical formulaic expressions, with columns `File_1`, `File_2`, `ClusterID`, `Variation_Type`, `Token_1`, `Token_2`. `Variation_Type` is one of:
* `Orthographic Variant` — the same wording spelled differently on each side.
* `Insertion` — present in one of the two texts only (the other column carries `-`).
* `Different Bridge Word` — genuinely divergent wording between two matches.

6. **`clusters.html`** / **`clusters.tsv`**: The clustering report — one card (HTML) and one row (TSV) per recurring legal formula, naming the formula, the documents using it, the pairs and their **two** scores, and the run's clustering statistics. The `Anchor` column names the performative verb the cluster's window was seeded on (empty when the pair's shared text carries none — a protocol-level match), which is the one-word answer to *which legal act is this cluster?*. The `DistinctTexts` column (and the same figure on each HTML card) gives the number of *different texts* behind the document list: where it is smaller than the number of documents, the pair count is inflated by repeated copies of the same text. The HTML opens with a summary of the input (how many charters, from where, with which thresholds, how many pairs the anchor served, and the exact command line that produced it), and — unless the corpus is too large to embed — the **full text of every charter in the cluster**, with the shared formula highlighted in place and each document name linking to it. See [Clustering](#clustering-recurring-legal-formulas).

The classification uses the same code path as the HTML report, so the register and the visualisation never disagree about what counts as a variant and what counts as a bridge.

> **Note — column layout changed.** `similarity_summary.tsv` and `linguistic_variations.tsv` carry a
> `ClusterID` column while `gen_clusters` is on (its default). Any script reading those files by
> column *position* has to be updated: the summary's `ClusterID` is the **second** column (right after
> `DocumentFilename`), the linguistic register's is the **third** (after `File_2`, before
> `Variation_Type`). Reading by **header name** is unaffected. `clusters.tsv` is a new file, so its
> columns affect nobody; for reference its order is
> `ClusterID, Size, Anchor, CoreFormula, CoreFolded, Documents, DistinctTexts, CoreTokens, MeanIDF,
> Specificity, Cohesion, Members, PairCosine, PairCoreRatio, SharedTokens, Coverage, Kind`.
> `Anchor` was added after `Size`, so a positional reader of `clusters.tsv` — a file new in the same
> release — should be checked against it.
> Turning the option off restores the older layout, so long as it is turned off in the GUI or by
> setting the default in `flame.py` — the command line cannot switch a boolean *off*, since `-flag`
> arguments are presence switches (see [Command-Line Interface](#command-line-interface-cli)).

---

## Clustering (recurring legal formulas)

Similarity tells you *which charters resemble each other*. It does not tell you **why** — and for
medieval charters the interesting "why" is usually legal: the same arenga, the same disposition
formula, the same transaction recorded by two scribes. FLAME's clustering answers that second
question. It borrows the three-stage algorithm from the **KONI** project (see
[Acknowledgements](#acknowledgements)), where it groups literary formulas, and re-targets it at
legal formulas.

**How it works.** For every pair that already cleared the similarity threshold, FLAME extracts the
pair's **core** by **gapped local alignment**: a chain of matching word-runs whose gaps stay within
`core_gap_tolerance` tokens on both sides. *Where that chain opens* is the anchoring question
(`core_anchors`, see [Where the window sits](#clustering-recurring-legal-formulas)). Left to itself
the aligner takes the chain that carries the most matched words, which on a mediaeval charter is the
protocol rather than the legal act — that is the shipped default. Given `core_anchors` it instead
seeds the chain on a **performative verb** — `donamus`, `contulimus`, `confirmamus` — and grows it
from there in both directions, capped at `core_max_tokens`. The anchor set is per-corpus input, not a
built-in list: see the measurements under
[Where the window sits](#clustering-recurring-legal-formulas). Strictly contiguous matching is what
the gapped chain replaced, and the
reason is diplomatic: a mediaeval
template *necessarily* varies at its slots, so a single inserted or substituted word — `uestro
auctoritate` against `auctoritate`, `eidem monasterio` against `hospitali uestro` — cut one formula
into two runs and the strict rule reported whichever half was longer. On the FLAME corpus that
fragmented one papal *confirmatio* into six micro-clusters and gave a median core of 27 words; the
aligned core spans the same formula at a median of 106 words (both figures measured with anchoring
off, so they describe the extraction rule, not the window). Measured against a token-level
Smith-Waterman implementation on the same pairs, that chain picks the same windows at a fraction of
the cost. The core is
taken on *spelling-folded* tokens (`vnd`/`und`, `czu`/`zu` and other medieval variant spellings fold
together), so one formula spelled two ways stays one formula — comparing raw spellings would split it
into as many clusters as it has spellings. Cores that are identical collapse in one pass; the
remaining *distinct* cores are then compared to each other (not pairs to pairs, which would repeat the
same comparison thousands of times) and merged by edit similarity at `cluster_threshold`.

Because those distinct cores are what the grouping runs on, membership follows the *core*, not the
pair: two copies of one charter whose cores were narrowed to differently-worded acts would not meet.
The pair-level near-duplicate rule is therefore evaluated first and the pairs that clear it are wired
into the graph as forced edges — see
[Formulas versus copies](#clustering-recurring-legal-formulas). The run prints how many pairs took
that route (`Duplicate links: …`), and how many pairs the anchor served (`Anchoring: …`), so a report
never leaves the reader guessing which of the two paths built a given cluster.

All of this is *already* formula-level work — the whole-charter cosine never decides who shares a
cluster with whom, it only decides which pairs are worth looking at. The report nevertheless showed
only that admission score, which made the grouping look like a whole-document similarity masquerading
as a formula match. Both numbers are now printed side by side, and `Cohesion` — the weakest
core-to-core score inside a cluster — is the figure that says how tight a cluster really is.

**Reading a cluster.** A cluster is not a similarity clique. For charters it can mean three things,
and the report shows you which applies:

- **the same legal formula** — an arenga, a *corroboratio*, a penalty clause, word for word across
  unrelated documents;
- **the same legal transaction** — one act, two or more surviving copies or confirmations;
- **a shared chancery template** — a formulary or *Formularbuch* in use by one scribe, notary, or
  issuing office.

The cluster's `CoreFormula` column is the formula as the charters actually spell it; `CoreFolded` is
what the clustering compared. When two charters land in one cluster despite visibly different
spelling, `CoreFolded` is the evidence for why. Keep in mind that the per-pair number in
`PairCosine` is the pair's *whole-charter* TF-IDF cosine — the score that let the pair into the
comparison — while `PairCoreRatio` is that pair's formula similarity to the cluster's headline
formula, i.e. the number that actually did the grouping; the two can differ a lot, because a pair can
share one formula nearly verbatim while being dissimilar everywhere else.

**Strict, transitive or density-based grouping (`cluster_linkage`).** Edit similarity at a fixed
threshold is generous enough to chain: A resembles B, B resembles C, A and C do not resemble each
other, and transitive merging files all three under one formula that no two of them share. The strict
answer is the opposite extreme — `clique` re-scores each group pairwise and splits it into its
**maximal cliques**, so every member matches every other member and `Cohesion` is at or above
`cluster_threshold` by construction. Neither extreme fits a *template*: "agrees with everyone" is
all-or-nothing, and a real formula varies at its slots, so the agreement edges do not close into
triangles everywhere. On the FLAME corpus the papal *confirmatio* survived as one connected component
that strict cliques then cut into five overlapping pieces, which is precisely what the formula report
must not do. The default `louvain` linkage asks the question the diplomatist is asking — which cores
are **denser among themselves** than with the rest — so the template stays in one cluster while two
formulas that merely share a phrase stay apart. `union` restores KONI's original transitive
behaviour, and is the right choice when you are looking for *families* of related formulas rather
than exact recurrence.

Two consequences are worth knowing. First, `louvain` and `union` return disjoint clusters, so
`ClusterID` is a single number and no pair appears twice. Second, `clique` *overlaps* by design and
FLAME keeps the overlap rather than resolving it — a formula that sits between two groups appears in
**both** clusters, marked on the card with `shares pair(s) with cluster N`, and `ClusterID` in the
summary TSVs becomes a comma-separated list. The default avoids that, but the strict mode remains
available for a reader who wants the guarantee that every formula in a cluster is within
`cluster_threshold` of every other.

**Formulas versus copies (the near-duplicate section).** Not every long shared window is a formula.
When a cluster's shared window covers most of the shorter charter, the two documents are not
*witnesses to a formula* but the same charter written out again, and the report lists those clusters
in their own section at the end so they neither pad nor distort the formula clusters above. Both
conditions have to hold, because either alone misreads a real case: the share alone calls a *short*
charter a duplicate when its text is mostly one formula (measured here: eighteen distinct charters
whose 160-word papal *confirmatio* is 99% of each), and the length alone calls a long shared *section*
a duplicate. So a near-duplicate needs a window of at least `min_duplicate_tokens` words (default
400 — longer than the longest formula in this corpus, 285 words) **and** at least 60% of the shorter
charter. Every card prints both numbers, `clusters.tsv` carries them in `SharedTokens` and `Coverage`
and marks each row `formula` or `near-duplicate` in `Kind`, and both thresholds are parameters, so a
reader who disagrees with the line can move it and see what changes.

Both numbers are read from **one** pair — the widest overlap that clears the rule, or the widest
overlap overall when none does. They used to be two independent maxima, the length taken from one
pair and the share from another, which meant the two figures printed on a card described no single
pair at all: measured on the MOM corpus that happened in 11 of the 40 clusters, in one case handing
the cluster a `Coverage` of 100% (a very short charter fully covered by some pair) alongside a
`SharedTokens` that came from a different, longer pair. The verdict is assembled from those two
numbers, so a borderline cluster's `Kind` rested on two separate pieces of evidence; no `Kind`
actually flipped on this corpus, but both figures now come from the pair that carries the label.
(`Coverage 1.000` with `Kind: formula` is not by itself a contradiction — the rule needs the word
count as well, which is why a short charter that is mostly one formula is not a copy.)

The rule also decides cluster *membership*, not just the label. Clustering runs on the cores, and a
core narrowed to the legal act (`core_max_tokens`) can match nothing even between two copies of one
charter: measured on the MOM corpus, anchored runs left six charter documents sharing 400+ words in
**no cluster at all**, their pair alone below `cluster_min`. So the pair-level rule is evaluated
first, and a pair that clears it is wired into the graph by that alone, whatever its core looks like.
Such a family of one pair is still below `cluster_min` and stays out of the cluster report — the run
prints how many, because a lone pair is a real find that only the pairs report can carry.

**How specific is a formula? (`CoreTokens`, `MeanIDF`, `Specificity`).** Length alone does not make a
formula interesting: `salutem et apostolicam benedictionem` and the `scripti patrocinio communimus`
clause are both recurring formulas, but only the second tells you anything about the charter it is
in. FLAME therefore scores each cluster's headline formula by **length × rarity**:

- `CoreTokens` — how many words the formula has;
- `MeanIDF` — the average inverse document frequency of its **distinct** words, in nats, smoothed as
  `log((1 + N) / (1 + df))` so a word present in every charter scores `0` rather than going negative;
- `Specificity` — `CoreTokens × MeanIDF`, the product of the two.

`min_core_specificity` cuts below a value you choose, and `stopwords_file` lets you zero out tokens
you already know are uninformative. Both act *before* cluster IDs are assigned, so cross-references
from the other TSVs stay valid.

**Watch the pair counts on a corpus with repeated files.** A corpus assembled from editions and
archival copies holds the same charter under several names — a shelfmark, an edition, and a `(1)`
re-download. Since a pair is a pair of *files*, one comparison then appears as nine, and a cluster
can read "9 pair(s), 6 document(s)" while holding only two texts. The `DistinctTexts` column and the
`(N distinct text(s))` note on each card expose exactly that, and `deduplicate` removes the repeats
upstream if you would rather not analyse them at all. Note the two are independent: the note reports
what is in the corpus, `deduplicate` decides what gets loaded.

**The summary block at the top of `clusters.html`.** It answers the reviewer's second question
directly: *on what input were these clusters computed*. It is a definition list (`<dl>`) of
name/value pairs:
one- or two-corpus mode, the path or glob
pattern, the file suffix, how many charters each side contributed, how many files were found, loaded,
skipped as too short, skipped as duplicates, or cut off by `keep_texts`, whether the similarity
threshold was set automatically (and by which method), the cluster threshold and linkage, a
timestamp, and the exact command line that produced the report.

**Reading the charters themselves.** Unless the corpus is too large to embed (in which case the
report says so and falls back to plain name lists), every charter in every cluster appears in the
report as a collapsible block containing its **complete text**, with the shared formula wrapped in
`<mark>`. The highlight comes from the same `(start, size)` span the clustering used, mapped back
through the tokeniser, so it marks the words the algorithm actually matched rather than a re-search
for the same string. Where a document carries two related formulas in one cluster, both are marked.

**Tuning.** `cluster_threshold` controls how strictly "the same formula" is judged: `1.0` clusters
only verbatim-identical cores, `0.70` (the default) also tolerates the substitution of the
template's variable slots — a different recipient, a different place — so that one template stays
one cluster, and `0.65` or below starts merging formulas that merely share a phrase. The default was
lowered from `0.85` after measuring both on the MOM corpus: at `0.85` a single papal privilege
template whose recipients differed fragmented into four clusters, at `0.70` it is one cluster and no
other cluster of four or more members merged. The core's *shape* is tuned separately: `core_gap_tolerance` widens or narrows the
variable slot the aligner may bridge (raise it for formulas with longer variable parts, lower it if
unrelated formulas start merging), and `core_identity_threshold` filters the loose end of the
distribution. Both are reported in the input summary, so a report tells you how its cores were
built.

**Where the window sits** is a separate question from how strictly it is judged, and
`core_anchors` / `core_anchor_window` / `core_max_tokens` answer it. Left to itself the aligner
maximises matched tokens, and the longest passage two mediaeval charters share is almost never the
legal act — it is the protocol they both inherited (intitulatio, salutatio, arenga). On the MOM
corpus that produced **no core below 56 tokens at all** (median 208), with 28 of 40 clusters opening
on a protocol phrase, and one papal privilege template split into four clusters of 24, 12, 6 and 2
charters purely because their recipients differed. Anchoring changes what is extracted: the window is
seeded on a performative verb, extended no further than the gap tolerance allows from it, and capped
at `core_max_tokens` — so a donation is clustered on `donamus … duas partes decimarum`, not on the
papal greeting above it, and the `Anchor` column says which verb it was. Measured on the MOM corpus
with the anchor stems supplied, anchoring moves the median core from 208 tokens to 82 (not to the
50-token ceiling: 153 of 441 pairs carry no anchor in their shared text and keep the long window),
and the formula clusters from 29 to 32 while the near-duplicate section grows from 43 to 50 charter
documents. Raise `core_anchor_window` to reach further past the verb into the surrounding clause,
lower it to insist on the operative words alone.

**Anchoring ships switched off, and the reason is measured.** A hand-written Latin-and-German anchor
list was tried first and does not hold up on the corpus it was written for: of its 38 stems, 11
(`vendidimus`, `vendimus`, `emancipauimus`, `concessimus`, `promittimus`, `assignauimus`,
`commutauimus`, `verleihen`, `verkaufen`, `ubergeben`, …) **do not occur in the corpus at all**, and
153 of the 441 admitted pairs (34.7%) share no word of it, so a third of the corpus fell back to the
unanchored window anyway. Discovering the verbs from the corpus was then tried and failed, four
independent ways, which is why the default is an empty list rather than a worse one:

* **Distributional rules do not separate the verbs from ordinary charter vocabulary.** Document
  frequency, occurrences per document, context entropy, positional spread, the IDF of a token's
  neighbours and every conjunction of them were ranked against the verbs the hand list did reach.
  The best rule selected **8676 tokens to find 45 verbs — precision 0.5%, recall 34.9%**. The verbs
  are 0.3% of the recurring vocabulary and statistically ordinary: `contulimus` has document
  frequency 416 beside `iuris` at 1093, in a mid-frequency band holding 1241 tokens of which 16 are
  verbs.
* **Phrase-level rarity does not locate the act.** Ranking a pair's shared windows by the document
  frequency of their 4-grams puts the act at relative rank **0.467 — chance**.
* **Rarity as the objective is the wrong sign.** Mean unigram IDF, summed IDF, rarest-word IDF and
  phrase DF all *anti-select* the act (median relative rank 0.57–0.70), because the act formula is
  shared by its whole act family, while the average shared passage is a one-off list of names. This
  is the same effect behind the older note that IDF alone picks bishopric enumerations.
* **Morphology is what a reader uses, and it is not separable on the surface.** Handed the ending
  inventory of the known verbs as an oracle, `-mus` / `-nt` / `-it` still selects **2215 tokens to
  find 50 verbs (2.3%)** — the endings are carried by thousands of participles and nouns.

Identifying the performative verb is a lexical or morphological task, so the anchor set is
per-corpus input rather than a constant in the code. Two calibration notes for supplying one: the
stems match as *prefixes*, so a stem that is too broad swallows the nouns derived from the verb
(`dona` would match `donatio`, changing a donation act into a donation word), and IDF is
deliberately **not** the seed's objective — it only chooses between anchors the act is already known
to be near.

`cluster_min` filters out formulas that occur only once. On a corpus where the same
charter is filed under several names, `deduplicate` is usually the more useful first move: it removes
the repeated comparisons *before* they inflate the pair counts.

---

## Recipes

### Find long, near-verbatim text reuse

Ideal for identifying direct text copying, textual transmission lineages, or structural plagiarism with minimal alterations.

```python
DEFAULT_PARAMS = {
    'input_path': './corpus',
    'file_suffix': '.txt',
    'keep_texts': 100000,
    'ngram': 15,                      # Target long sequential strings
    'n_out': 1,                       # Enforce rigid matching (only 1 drop allowed)
    'min_text_length': 150,
    'similarity_threshold': 0.85,     # High threshold bar for rigid matches
    'auto_threshold_method': 'otsu',
    'char_norm_alphabet': "abcdefghijklmnopqrstuvwxyz",
    'char_norm_strategy': 'normalize',
    'char_norm_min_freq': 2,
    'vocab_size': 8000,               # Large vocabulary to force whole-word evaluation units
    'vocab_min_word_freq': 3,
    'vocab_coverage': 0.85,
}

```

### Find rephrased or restructured text

Utilizes balanced windows while extending token dropping limits to see through heavy lexical changes and active alterations.

```python
DEFAULT_PARAMS = {
    'input_path': './corpus',
    'file_suffix': '.txt',
    'keep_texts': 100000,
    'ngram': 10,                      # Phrase-level matching properties
    'n_out': 3,                       # Higher tolerance for token shifts and insertions
    'min_text_length': 150,
    'similarity_threshold': 0.60,     # Moderate cutoff to surface paraphrased reuse
    'auto_threshold_method': 'otsu',
    'char_norm_alphabet': "abcdefghijklmnopqrstuvwxyz",
    'char_norm_strategy': 'normalize',
    'char_norm_min_freq': 2,
    'vocab_size': 'auto',             # Handled flexibly via subwords
    'vocab_min_word_freq': 3,
    'vocab_coverage': 0.85,
}

```

### Find formulaic language / arengas

The standard optimized configurations. Leverages autonomous adaptive thresholds alongside gapped footprints to harvest historical formulae.

```python
DEFAULT_PARAMS = {
    'input_path': './corpus',
    'file_suffix': '.txt',
    'keep_texts': 100000,
    'ngram': 6,                       # Standard formulaic anchor bounds
    'n_out': 1,                       # Adaptive single gap indexing
    'similarity_threshold': 'auto',   # Calibrate cut-offs automatically via Otsu
    'auto_threshold_method': 'otsu',
    'char_norm_alphabet': "abcdefghijklmnopqrstuvwxyz",
    'char_norm_strategy': 'normalize',
    'char_norm_min_freq': 1,
    'vocab_size': 'auto',             # Morphologically optimized on-the-fly
    'vocab_min_word_freq': 5,
    'vocab_coverage': 0.85,
}

```

### Find recurring legal formulas (clusters)

Group the related pairs by the formula they share rather than by overall resemblance, so the result
reads as a list of formulas — each one with the charters that use it — instead of a list of similar
pairs. The default `cluster_threshold` of `0.70` tolerates ordinary scribal variation inside a
formula and the substitution of its variable slots (recipient, place); raise it to `1.0` to see only
verbatim copies.

```python
DEFAULT_PARAMS = {
    'input_path': './corpus',
    'file_suffix': '.txt',
    'keep_texts': 100000,
    'ngram': 6,
    'n_out': 1,
    'similarity_threshold': 'auto',   # controls which pairs even reach the clustering
    'auto_threshold_method': 'otsu',
    'cluster_threshold': 0.70,        # controls which of them count as one formula
    'cluster_min': 2,
    'cluster_linkage': 'louvain',     # density-based; 'clique' for the strict guarantee
    'core_anchors': '',               # '' = unanchored; give stems to seed on the legal act
    'core_anchor_window': 15,         # how far past the verb the window may reach
    'core_max_tokens': 50,            # ceiling on the reported window; 0 = uncapped
    'min_core_specificity': 0.0,      # raise to ~10-15 to drop universal panels
    'stopwords_file': '',             # optional hand-written diplomatic stop list
    'gen_clusters': True,
    'char_norm_alphabet': "abcdefghijklmnopqrstuvwxyz",
    'char_norm_strategy': 'normalize',
    'char_norm_min_freq': 1,
    'vocab_size': 'auto',
    'vocab_min_word_freq': 5,
    'vocab_coverage': 0.85,
}

```

`similarity_threshold` and `cluster_threshold` answer two different questions and should be tuned
together: the first decides which charters are compared at all, the second decides which of those are
judged to share a formula. The report's statistics line tells you how much work the gate did
(`lev_compares`, `gate_skips`), which is the quickest way to tell whether a threshold is too strict.

### Analyse one series inside a nested corpus

Corpora such as *fsdb* nest one folder per document and reuse the same filename
everywhere. Point `input_path` at a pattern to select a series, and the reports will
still tell the documents apart, because each is named by its path relative to the
pattern's leading directory:

```python
DEFAULT_PARAMS = {
    'input_path': './fsdb/DE-LANRWR**/*.htr.txt',   # one series, one level down
    'file_suffix': '.txt',                           # ignored: the pattern decides
    'keep_texts': 100000,
    'ngram': 6,
    'n_out': 1,
    'min_text_length': 150,
    'similarity_threshold': 'auto',
    'auto_threshold_method': 'otsu',
    'char_norm_alphabet': "abcdefghijklmnopqrstuvwxyz",
    'char_norm_strategy': 'normalize',
    'char_norm_min_freq': 1,
    'vocab_size': 'auto',
    'vocab_min_word_freq': 5,
    'vocab_coverage': 0.85,
}

```

Reports then read `DE-LANRWR001/text.htr.txt` rather than an ambiguous
`text.htr.txt`. To compare two series against each other instead, put one pattern in
`input_path` and the other in `input_path2`; each side is named relative to its own root.

---

## Acknowledgements

The character normalization components are inspired by and build upon the principles found in **Anguelos Nicolaou's**  library. Anguelos's efficient character mapping was a valuable reference for this project.

The clustering component is a port of the clustering algorithm from the **KONI** project
(Korpus Nyelvtechnológiai Infrastruktúra), which groups literary formulas in Hungarian corpora. FLAME
keeps KONI's three-stage structure and its scoring metric, and re-targets it at the legal formulas of
medieval charters.

---

## Cite

### APA Style

Kovács, T. (2025). *FLAME: Formulaic Language Analysis in Medieval Expressions* (Version 1.0.0) [Computer software]. GitHub. https://github.com/kreeedit/FLAME

### BibTeX

```bibtex
@software{Kovacs_FLAME_2025,
  author = {Kovács, Tamás},
  title = {{FLAME: Formulaic Language Analysis in Medieval Expressions}},
  version = {1.0.0},
  publisher = {Zenodo},
  year = {2025},
  doi = {10.5281/zenodo.15805449},
  url = {[https://github.com/kreeedit/FLAME](https://github.com/kreeedit/FLAME)}
}

```

---

## License

This project is licensed under the **Apache 2.0 License**.
