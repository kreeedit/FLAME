# FLAME: method in detail

The full description of the five pipeline stages, condensed in the [README](../README.md#method).


Formulaic language consists of clauses that recur with variation across documents. It is a
known difficulty for historical linguistics (Korkiakangas 2024), and in medieval texts the
variation comes on top of unstandardized spelling, abbreviation, allographs, and recognition
errors. FLAME is built to tolerate that variation. It runs five stages in a fixed order. Each item below names the CLI parameter that controls it
and links the code that implements it, so you can check the description against the source. The
design rationale and the argument for each stage are in the accompanying article (Kovács, in
preparation for *Digital Scholarship in the Humanities*, §§5–9).

**1. Layered character normalization** ([`flame.py`: `Flame.load_corpus()`](../flame.py#L668),
[`AdaptiveAlphabet.learn_mappings()`](../flame.py#L253), tables
[`MUFI_ONE_TO_MANY` / `MUFI_ONE_TO_ONE`](../flame.py#L1079)). The text is lower-cased. Then ligatures
and digraph characters are expanded with a one-to-many table (æ→ae, ß→ss, þ→th, ﬁ→fi, …), and a
one-to-one table based on the MUFI recommendation (Haugen 2009) maps allographs to their base
letters (ſ→s, ꝛ→r, ꝼ→f, ᵹ→g, ƿ→w, …). The one-to-one mapping is a vectorized lookup table over the
Unicode Basic Multilingual Plane. On top of it, an adaptive learner finds the characters outside
the target alphabet (`char_norm_alphabet`, default a–z) that occur at least `char_norm_min_freq`
times. If a character's Unicode compatibility decomposition (NFKD) begins with a letter of the
alphabet, the learner maps the character to that letter (é→e, ñ→n) and prints each rule it adds.
Any character still unaccounted for becomes a space, that is, a token boundary. Inflection and
spelling variation (u/v, i/j, doubled consonants, *vredens*/*friddens*) are deliberately
**not** normalized here. The tokenizer and the fingerprint absorb them later, and the report shows
them. The optional `bigram_normalization_enabled` and `phonetic_reduction_enabled` layers fold more
aggressively. Both are off by default.

**2. Subword tokenization** ([`suggest_vocab_size_optimized()`](../flame.py#L70), BPE training in
[`Flame.load_corpus()`](../flame.py#L795)). A Byte-Pair-Encoding tokenizer (Sennrich et al. 2016;
Hugging Face `tokenizers`) is trained on the normalized corpus, and every document becomes a
sequence of subword IDs. `-vocab_size` takes an integer or `auto`. With `auto`, words with frequency
of at least `vocab_min_word_freq` are sorted, then sorted again reversed. The longest common prefix
of each adjacent pair (2–6 characters) becomes a prefix or suffix candidate, weighted by the two
words' frequencies. The vocabulary size is the number of top-ranked affixes that covers
`vocab_coverage` (default 0.85) of all affix occurrences, plus 256. It is capped at the number of
distinct words plus 256.

**3. Leave-n-out (gapped) n-gram fingerprint** ([`Flame.leave_n_out_grams()`](../flame.py#L962)). The
token sequence is cut into overlapping windows of `-ngram` tokens. From every window, FLAME keeps
each of the C(n, k) subsets of k = n − n_out positions (`-n_out` positions dropped). A window
therefore yields

&nbsp;&nbsp;&nbsp;&nbsp;C(n, n − n_out) = n! / ((n − n_out)! · n_out!) sub-grams,

and a document of L tokens yields C(n, n − n_out) · (L − n + 1). For example, there are 6 for
n = 6, n_out = 1 (the defaults) and 10 for n = 5, n_out = 2. Each sub-gram (t₀ … t₍ₖ₋₁₎) is hashed
as Σ tⱼ·Vʲ mod (2⁶¹ − 1), a Rabin–Karp-style polynomial hash (Karp and Rabin 1987) over the token
vocabulary size V. The hash is evaluated for all windows at once as one matrix product per
position subset. Robustness follows from this construction (article §7.3). A single substituted
token at position p breaks **every** contiguous n-gram whose window spans p. With n_out ≥ 1, each
window that spans p still has at least one kept-position subset that skips p, so at least one
gapped sub-gram per affected window is shared by both passages. BPE limits the number of tokens
that a spelling variant, insertion, or deletion disturbs, and the gapped windows bridge those
tokens. The price is specificity, because gapped features are shared by chance more often. That
trade-off is why `ngram` and `n_out` are exposed.

**4. Similarity and automatic threshold** ([`Flame.compute_similarity_matrix()`](../flame.py#L993),
[`Flame._determine_auto_threshold()`](../flame.py#L941)). Fingerprints become a sparse
document-by-feature count matrix. The matrix is TF-IDF weighted (scikit-learn `TfidfTransformer`,
fitted on both corpora together in two-directory mode), and the cosine similarity is computed in
sparse form, so pairs with no shared feature are never stored. With `-similarity_threshold auto`
(the default), Otsu's method (Otsu 1979) is applied to the **non-zero** scores only. This removes
the mass of pairs that share nothing, a distribution so heavily skewed that a two-class split of
the raw scores would only separate zero from non-zero. `-auto_threshold_method percentile` uses
the 99th percentile instead, and a number given to `-similarity_threshold` overrides both. Otsu's
method assumes the non-zero scores are bimodal. Where they are not, as on the Ostrowski benchmark
([`benchmarks/ostrowski2024/`](../benchmarks/ostrowski2024/README.md), threshold fixed at 0.28),
set the threshold explicitly.

**5. Alignment report** ([`SimilarityVisualizer.highlight_similarities()`](../flame.py#L1464),
[`_render_gap_html()`](../flame.py#L1432), [`classify_gap()`](../flame.py#L1127),
[`generate_linguistic_summary_tsv()`](../flame.py#L2331)). For every pair above the threshold, the
shared passages are re-aligned on the **unnormalized** display tokens. The fingerprints are lossy
and unordered, so they are not used here. The aligner is `difflib.SequenceMatcher`, a
Ratcliff–Obershelp longest-matching-block alignment (Ratcliff and Metzener 1988), run on a
lower-cased, alphanumeric-only view. Highlights are mapped back to the text as transcribed. Each
short gap between two matching blocks (at most `max_gap_words` per side) is classified on
spelling-folded forms: an **insertion** if one side is empty, an **orthographic variant** if the
fuzzy ratio is at least `fuzz_threshold` (default 0.70), and otherwise a **bridge**, where the two
texts genuinely diverge. The same classification, aggregated over all reported pairs, is written
to `linguistic_variations.tsv`, the corpus-wide table of linguistic variants. Text before the first
or after the last matching block is shown unmarked and not classified.

## References

Haugen, O. E. (Ed.) (2009). *MUFI Character Recommendation: Characters in the Official Unicode Standard and in the Private Use Area for Medieval Texts Written in the Latin Alphabet*, version 3.0. Bergen: Medieval Unicode Font Initiative.

Karp, R. M., & Rabin, M. O. (1987). Efficient randomized pattern-matching algorithms. *IBM Journal of Research and Development*, 31(2), 249–260. https://doi.org/10.1147/rd.312.0249

Korkiakangas, T. (2024). A linguist's viewpoint: Formulaic language as a challenge for historical linguistics. https://doi.org/10.5281/ZENODO.10461847

Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66.

Ratcliff, J. W., & Metzener, D. E. (1988). Pattern matching: The gestalt approach. *Dr. Dobb's Journal*, 13(7), 46–72.

Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1715–1725). Berlin: Association for Computational Linguistics.
