# Recipes

Full parameter sets for common tasks, as `DEFAULT_PARAMS` overrides. The [README](../README.md#recipes) lists the same recipes as one-line commands.

## Find long, near-verbatim text reuse

For direct text copying, transmission lineages and structural plagiarism with minimal alterations.

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

## Find rephrased or restructured text

Balanced windows with widened token dropping, to see through heavy lexical change.

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

## Find formulaic language / arengas

The standard optimized configuration, with autonomous thresholds and gapped footprints.

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

## Find recurring legal formulas (clusters)

Groups the related pairs by the formula they share rather than by overall resemblance, so the result
reads as a list of formulas — each with the charters that use it — instead of a list of similar pairs.
The default `cluster_threshold` of `0.70` tolerates ordinary scribal variation inside a formula and
the substitution of its variable slots; raise it to `1.0` to see only verbatim copies.

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

`similarity_threshold` and `cluster_threshold` answer two different questions and are tuned together:
the first decides which charters are compared at all, the second which of those are judged to share a
formula. The report's statistics line tells you how much work the gate did (`lev_compares`,
`gate_skips`), the quickest way to tell whether a threshold is too strict.

## Analyse one series inside a nested corpus

Corpora such as *fsdb* nest one folder per document and reuse the same filename everywhere. Point
`input_path` at a pattern to select a series; the reports still tell the documents apart, because each
is named by its path relative to the pattern's leading directory.

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

Reports then read `DE-LANRWR001/text.htr.txt` rather than an ambiguous `text.htr.txt`. To compare two
series against each other, put one pattern in `input_path` and the other in `input_path2`; each side is
named relative to its own root.
