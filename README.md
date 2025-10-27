# FLAME: Formulaic Language Analysis in Medieval Expressions
 

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**FLAME** is a Python-based tool with both **Command-Line (CLI)** and **Graphical (GUI)** interfaces, designed for identifying and analyzing formulaic language and text reuse, particularly in historical corpora like medieval charters. It uses a **Leave-N-Out (LNO) n-gram** approach, which is highly effective for detecting variant forms of expressions that differ due to scribal variations, regional dialects, or other textual modifications. It automatically learn normalization rules from the corpus itself (handling medieval ligatures and special characters), uses subword tokenization to handle rare words and morphological variants. Automatically suggest an optimal vocabulary size for the tokenizer based on the corpus's statistical properties. It perform both intra-corpus and inter-corpus comparisons, and automatically determine an optimal similarity cutoff score using Otsu's method.

A downloadable demo of the HTML output can be found in the repository (`text_comparisons_demo.html`).

<p align="center">
  <img src="flame-little-flame.gif" width="200" alt="FLAME animation" />
</p>

## How It Works

The LNO-gram approach systematically creates robust features from text. For a given sequence of words (an n-gram), it generates multiple variants by omitting a specified number of tokens. This allows the system to identify underlying similarities even if the surface forms are not identical.

Consider the medieval charter opening: *"In nomine sancte et individue trinitatis amen"*

1.  **Generate n-grams**: The tool slides a window of a specified length (e.g., 5 words) across the text.
2.  **Create LNO variants**: For each 5-gram, it creates subsequences by removing a specified number of tokens (e.g., 1). For the 5-gram `[In, nomine, sancte, et, individue]`, it would generate features like `[_, nomine, sancte, et, individue]`, `[In, _, sancte, et, individue]`, etc.
3.  **Hashing**: Each variant is converted into a unique, memory-efficient integer hash.
4.  **Similarity Calculation**: The tool calculates the cosine similarity between documents based on the frequency of these shared feature hashes.
5.  **Visualization**: Results are presented in interactive reports that highlight matching patterns in their original context.

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
-   **Adaptive Character Normalization**: Autonomously learns and applies normalization rules (e.g., `é` -> `e`) to reduce noise from character variations.
-   **Automatic Threshold Detection**: Intelligently determines the optimal similarity threshold using Otsu's method, removing the need for manual guesswork.
-   **Dual Interface**: Can be run as a powerful command-line tool or through a user-friendly Graphical User Interface (GUI).
-   **Comprehensive Reporting**: Generates multiple outputs for in-depth analysis, including interactive HTML reports and detailed TSV files.
-   **High Performance & Scalability**: Handles large corpora by using sparse matrices, iterative vocabulary building, and efficient hashing.

---

## Installation

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

4.  **Download NLTK data:** Run the following command in a Python interpreter to download the necessary 'punkt' tokenizer models.
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('punkt_tab')
    ```
---

## Usage

You can run the analysis using either the GUI or the CLI.

### Graphical User Interface (GUI)

The GUI provides an intuitive way to set all parameters and monitor the analysis progress.

<p align="center">
  <img src="flame_gui.png" width="400" />
</p>

To launch the graphical interface, run:
```bash
python flame_gui.py
```

### Command-Line Interface (CLI)

To see all available options and their defaults, run:
```bash
python flame.py --help
```

**Example:**
```bash
python flame.py --input_path ./path/to/texts --ngram 10 --n_out 1 --similarity_threshold auto
```

### All CLI Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_path` | (see code) | **Required.** Path to the primary corpus directory. |
| `input_path2`| `None` | Optional path to a second corpus for inter-corpus comparison. |
| `file_suffix`| `.txt` | File extension of texts to process. |
| `keep_texts` | `10000` | Maximum number of texts to load from each directory. |
| `ngram` | `10` | The size of the n-gram window for feature generation. |
| `n_out` | `1` | Number of tokens to "leave out" from each n-gram. |
| `min_text_length` | `150` | Minimum character length for a file to be included. |
| `similarity_threshold` | `'auto'` | Similarity cutoff. Can be a float (e.g., `0.5`) or `'auto'`. |
| `auto_threshold_method` | `'otsu'` | Method for auto-thresholding: `'otsu'` or `'percentile'`. |
| `char_norm_alphabet` | `abcdef...` | String of allowed characters for normalization. |
| `char_norm_strategy` | `'normalize'` | Strategy for handling unknown characters (e.g., Unicode decomposition). |
| `char_norm_min_freq` | `1` | Minimum frequency for the adaptive normalizer to learn a character rule. |
| `vocab_size` | `'auto'` | Target vocabulary size. Can be an integer or `'auto'` to derive from other settings. |
| `vocab_min_word_freq`| `3` | Minimum frequency for a word to be included in the vocabulary. |
| `vocab_coverage`| `0.85` | The desired vocabulary coverage of the corpus, used when `vocab_size` is `'auto'`. |
---

## Outputs

FLAME generates up to four types of output files in the directory where it is run:

1.  **`dist_mat.npz`**: A SciPy sparse matrix file containing all pairwise similarity scores. Essential for re-analysis without re-computing.
2.  **`text_comparisons_XX.html`**: Interactive HTML files for visually comparing similar document pairs. This is the primary output for analysis. Features include:
    -   Side-by-side text comparison with synchronized highlighting.
    -   Static highlighting of "almost-matching" words on the left.
    -   Dynamic, on-click highlighting of corresponding words on the right.
    -   Controls to show/hide all similarities at once.
3.  **`similarity_summary.tsv`**: A high-level summary listing each document, how many other documents it is similar to, the names of those documents, and any long matching phrases (>4 words).
4.  **`linguistic_variations.tsv`**: A detailed TSV file for linguistic analysis, logging every "similar" and "different" word pair found in the short gaps (1-3 words) between main text matches. This is invaluable for studying micro-variations.

<p align="center">
    <img src="html_comparsion.png" width="1200" alt="HTML Comparison Screenshot" />
</p>

---
## Recipes
### Find long, near-verbatim text reuse
To find long, exact or near-exact copies of text. This is ideal for checking for direct plagiarism or verbatim text reuse with only minor changes.

```Pyton
DEFAULT_PARAMS = {
    'input_path': '',
    'input_path2': '',
    'file_suffix': '.txt',
    'keep_texts': 100000,
    'ngram': 15,                      # Look for long sequences
    'n_out': 1,                       # Allow only one token to be different
    'min_text_length': 150,
    'similarity_threshold': 0.85,     # Set a very high bar for similarity
    'auto_threshold_method': 'otsu',
    'char_norm_alphabet': "abcdefghijklmnopqrstuvwxyz",
    'char_norm_strategy': 'normalize',
    'char_norm_min_freq': 2,
    'vocab_size': 8000,               # Use a large vocab to treat words as unique units
    'vocab_min_word_freq': 3,
    'vocab_coverage': 0.85,
}
```
### Find rephrased or restructured text.
This uses a medium-sized ngram window but allows for a moderate number of tokens to be different. It's less strict than the plagiarism detector and can see through changes in vocabulary and sentence structure.

```Pyton
DEFAULT_PARAMS = {
    'input_path': '',
    'input_path2': '',
    'file_suffix': '.txt',
    'keep_texts': 100000,
    'ngram': 10,                      # Look for phrase-level patterns
    'n_out': 3,                       # Allow several words to be substituted or changed
    'min_text_length': 150,
    'similarity_threshold': 0.60,     # A medium threshold to catch non-exact matches
    'auto_threshold_method': 'otsu',
    'char_norm_alphabet': "abcdefghijklmnopqrstuvwxyz",
    'char_norm_strategy': 'normalize',
    'char_norm_min_freq': 2,
    'vocab_size': 'auto',             # 'auto' is good for handling different vocabulary
    'vocab_min_word_freq': 3,
    'vocab_coverage': 0.85,
}
```

### Find formulaic languages / arengas
This is the balanced approach. It allows for significant variation (n_out) within a moderately sized window (ngram).

```Pyton
DEFAULT_PARAMS = {
    'input_path': '',
    'input_path2': '',
    'file_suffix': '.txt',
    'keep_texts': 100000,
    'ngram': 8,                       # A balanced window size
    'n_out': 3,                       # High tolerance for word substitution
    'min_text_length': 150,
    'similarity_threshold': 'auto',   # Let the algorithm find the natural threshold
    'auto_threshold_method': 'otsu',
    'char_norm_alphabet': "abcdefghijklmnopqrstuvwxyz",
    'char_norm_strategy': 'normalize',
    'char_norm_min_freq': 2,
    'vocab_size': 'auto',             # 'auto' is ideal for historical spelling variations
    'vocab_min_word_freq': 3,
    'vocab_coverage': 0.85,
}
```


## Acknowledgements

The character normalization components are inspired by and build upon the principles found in **Anguelos Nicolaou's** 
![pylelemmatize](https://github.com/anguelos/pylelemmatize) library. Anguelos's efficient character mapping was a valuable reference for this project.

---
## Cite
### APA Style
Kovács, T. (2025). *FLAME: Formulaic Language Analysis in Medieval Expressions* (Version 1.0.0) [Computer software]. GitHub. https://github.com/kreeedit/FLAME


### BibTex


```BibTex
@software{Kovacs_FLAME_2025,
  author = {Kovács, Tamás},
  title = {{FLAME: Formulaic Language Analysis in Medieval Expressions}},
  version = {1.0.0},
  publisher = {Zenodo},
  year = {2025},
  doi = {10.5281/zenodo.15805449},
  url = {https://github.com/kreeedit/FLAME}
}
```

---
## License

This project is licensed under the **Apache 2.0 License**.
