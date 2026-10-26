# Article (DSH draft) statements that the replication does not support as written

**Labels:** article, documentation

Found while building `benchmarks/ostrowski2024/`. No code was changed. Each item says what the code
or the replication run shows.

1. **Table 1 caption: "full 23-pair gold standard".** The figures (R = 0.63, F1 = 0.75) are 12 hits out
   of the 19 literal + weak-literal pairs. That is the right basis for comparison with the
   Passim/BLAST/TextPAIR rows, but the caption should say "19 literal and weak-literal pairs". Over
   all 23 pairs, the same 13 detections give R = 0.52 and F1 = 0.67, which is Table 2's first row.
   Labeling Table 2 "all 23 pairs" removes the apparent contradiction. The four non-literal pairs
   are Regests 4, 21 (two pairs), and 22.
2. **Table 2 "− subword (word-level, vocab ≈ 360)".** This is BPE with `-vocab_size auto` (360 units
   on this corpus), not a word-level tokenizer. It is also flame.py's default setting.
3. **§8 "cosine similarity over count vectors".** The code applies TF-IDF weighting
   (`TfidfTransformer`) before the cosine, and has done so since b6d0a22 (June 2026).
4. **§8 log-transform for Otsu.** Not implemented. See the Otsu issue draft (01).
5. **§7.2 "polynomial rolling hash".** Each sub-gram's polynomial is evaluated directly, as a
   matrix product. The values are identical; only the word "rolling" is off.
6. **§10.5 bridges.** *ind herschapie* is rendered as an insertion. Since the report fix (see
   CHANGELOG, Unreleased), *friddens* / *vredens* (Dk G1r / KCh a5r5) is an orthographic variant,
   and *friddens* / *waren vredens* (a5r4) is one bridge; *waren* is not flagged separately.
7. **§10.5 "several of FLAME's few non-background false positives…"** At the operating point
   there is one false positive (Dk H-G1v / KCh bg_0301), and it is a background segment.
8. **§10.2 Task B.** The text describes a diplomatist-annotated stratified sample, with a word-n-gram
   detector and TRACER as baselines. Table 1 uses the Ostrowski gold standard and quotes
   Passim/BLAST/TextPAIR from Ostrowski (2024a).
9. **§10 "result tables are presented as templates"** and "Data and code availability:
   [dedicated repo not ready yet]". The replication package is now `benchmarks/ostrowski2024/`.
10. **Operating point selection.** The operating point (vocab 35, ngram 5, n_out 2, threshold 0.28)
    is the best-F1 cell of a 459-configuration × 36-threshold grid, scored on the same 23 pairs.
    There is no held-out split, so Table 1's FLAME figure is an optimistic, in-sample number.
11. **§10.3 hypothesis for Task A.** The article expects contiguous n-grams to lose recall steeply
    with the perturbation rate, while FLAME's recall degrades gracefully. `perturbation.py` (r =
    0–0.30, 5 seeds, every KCh file perturbed) ran four layer settings, A0–A3. In each, FLAME is at
    0.28 and the contiguous variant at its clean best-F1 threshold, 0.12. The mean recall difference
    FLAME − contiguous lies between −0.08 and +0.06 at every rate, and is largest in A0 (mean +0.03).
    At a shared threshold of 0.28, the contiguous variant's recall in A0 is 0.30 at r = 0 and 0.16
    at r = 0.30, at precision 1.00. The tables are in `benchmarks/ostrowski2024/README.md`.
