# Alignment report: divergence after the last matching block is not classified

**Labels:** visualizer, article

## What happens

`SimilarityVisualizer.highlight_similarities()` classifies only the gaps *between* two matching
blocks (`_render_gap_html`). Tokens after the last block are appended as plain text, and so are
tokens before the first block. They never appear as a bridge, variant, or insertion, and never
reach `linguistic_variations.tsv`.

## Example from the article (§10.5, Ostrowski Regest 5)

Dk `G1r` *eyn Stat des friddens* vs. KCh `a5r4` *eyn Stat des waren vredens* (cosine 0.2827 at
vocab 35 / ngram 5 / n_out 2). The report highlights *eyn Stat des* and prints *friddens* and
*waren vredens* unmarked. `linguistic_variations.tsv` has no row for the pair. The article says
*friddens* / *vredens* is "rendered as a similar bridge, while the inserted *waren* is flagged
separately", but the current code does not render it that way.

The other §10.5 example (Dk `G5r` / KCh `r141`) works, though as an **insertion**, not a bridge:
*ind herschapie* sits between *Alle ouerschaff* and *is van gode*. See
`benchmarks/ostrowski2024/results/alignment_regest13.png`.

## Reproduce

    cd <scratch dir with dk/ and kch/ from benchmarks/ostrowski2024/data/segments>
    python flame.py -input_path dk -input_path2 kch -min_text_length 15 -vocab_size 35 \
        -ngram 5 -n_out 2 -char_norm_min_freq 2 -similarity_threshold 0.28
    # search text_comparisons_01.html for "friddens"

## Not done

No code change. Classifying a trailing gap of at most `max_gap_words` on both sides would be
needed to match the article. Otherwise the article sentence should change.
