# Clustering (recurring legal formulas)

The full documentation of the clustering stage, summarized in the [README](../README.md#clustering-recurring-legal-formulas).

Similarity says *which* charters resemble each other; clustering says *why*. For medieval charters the
interesting "why" is usually legal: the same arenga, the same disposition formula, the same
transaction recorded by two scribes. The component is a port of the three-stage clustering algorithm
of the **KONI** project (see [Acknowledgements](../README.md#acknowledgements)), re-targeted from literary
formulas to legal ones.

**How it works.** For every pair above the similarity threshold, FLAME extracts the pair's **core** by
**gapped local alignment**: a chain of matching word-runs whose gaps stay within `core_gap_tolerance`
tokens on either side. Gapped rather than contiguous, because a formula varies at its slots — a single
inserted or substituted word (`uestro auctoritate` against `auctoritate`) used to cut one formula into
two runs, and the strict rule reported whichever half was longer. On the FLAME corpus that fragmented
one papal *confirmatio* into six micro-clusters with a median core of 27 words; the aligned core spans
the same formula at a median of 106 words. Measured against a token-level Smith-Waterman
implementation on the same pairs, the chain picks the same windows at a fraction of the cost.

Cores are taken on **spelling-folded** tokens, so `vnd`/`und` and `czu`/`zu` fold together and one
formula spelled two ways stays one formula. Identical cores collapse in one pass; the remaining
*distinct* cores are compared to each other — not pairs to pairs, which would repeat the same
comparison thousands of times — and merged by edit similarity at `cluster_threshold`. *Where* that
chain opens is the anchoring question, answered by `core_anchors`; by default the aligner takes the
chain carrying the most matched words, which in a charter is the protocol rather than the act.

Because the distinct cores are the graph's nodes, membership follows the *core*, not the pair. The
pair-level near-duplicate rule is therefore evaluated first and the pairs that clear it are wired into
the graph as forced edges. The run prints both routes (`Anchoring: …`, `Duplicate links: …`), so a
report never leaves the reader guessing which built a cluster.

Whole-charter cosine decides only which pairs are worth looking at, never who shares a cluster with
whom. Both numbers are printed per pair (`PairCosine` is the whole-charter score that admitted the
pair, `PairCoreRatio` its formula similarity to the cluster's headline formula), and `Cohesion` — the
weakest core-to-core score inside a cluster — says how tight the cluster really is.

**Reading a cluster.** A cluster is not a similarity clique. In charters it means one of:

- **the same legal formula** — an arenga, a *corroboratio*, a penalty clause, word for word across unrelated documents;
- **the same legal transaction** — one act, two or more surviving copies or confirmations;
- **a shared chancery template** — a formulary or *Formularbuch* in use by one scribe, notary or issuing office.

`CoreFormula` is the formula as the charters actually spell it; `CoreFolded` is what the clustering
compared, and the evidence when two charters land in one cluster despite visibly different spelling.

**Grouping (`cluster_linkage`).** Edit similarity at a fixed threshold chains: A resembles B, B
resembles C, A and C do not. `clique` is the opposite extreme — each group is re-scored pairwise and
split into **maximal cliques**, so every member matches every other one and `Cohesion` is at or above
`cluster_threshold` by construction. Neither extreme fits a *template*, whose agreement edges do not
close into triangles everywhere: on the FLAME corpus the papal *confirmatio* survived as one connected
component that strict cliques then cut into five overlapping pieces. The default `louvain` asks the
question the diplomatist is asking — which cores are **denser among themselves** than with the rest —
so the template stays in one cluster while two formulas that merely share a phrase stay apart. `union`
restores KONI's transitive merging, the right choice when looking for *families* of related formulas
rather than exact recurrence.

`louvain` and `union` return disjoint clusters, so `ClusterID` is a single number and no pair appears
twice. `clique` *overlaps* by design and FLAME keeps the overlap rather than resolving it: a formula
between two groups appears in **both** clusters, marked `shares pair(s) with cluster N`, and
`ClusterID` in the summary TSVs becomes a comma-separated list.

**Formulas versus copies.** Not every long shared window is a formula. When it covers most of the
shorter charter, the two documents are not *witnesses to a formula* but the same charter written out
again, and the report lists those clusters in their own section so they neither pad nor distort the
formula clusters above. Both conditions are required, because either alone misreads a real case: the
share alone calls a *short* charter a duplicate when its text is mostly one formula (measured here:
eighteen distinct charters whose 160-word papal *confirmatio* is 99% of each), and the length alone
calls a long shared *section* a duplicate. A near-duplicate therefore needs a window of at least
`min_duplicate_tokens` words (default 400, longer than the longest formula in this corpus, 285 words)
**and** at least 60% of the shorter charter. Every card prints both numbers; `clusters.tsv` carries
them in `SharedTokens` and `Coverage` and marks each row `formula` or `near-duplicate` in `Kind`.

Both numbers are read from **one** pair — the widest overlap that clears the rule, or the widest
overlap overall when none does. They were once two independent maxima, so the two figures on a card
described no single pair: on the MOM corpus that happened in 11 of the 40 clusters, once pairing a
`Coverage` of 100% with a `SharedTokens` from a different, longer pair. No `Kind` flipped, but the
verdict is assembled from those two numbers, so both now come from the pair that carries the label.
(`Coverage 1.000` with `Kind: formula` is not a contradiction on its own — the rule needs the word
count as well.)

The rule also decides cluster *membership*, not just the label, because a core narrowed to the legal
act can match nothing even between two copies of one charter: anchored runs on the MOM corpus left six
charter documents sharing 400+ words in **no cluster at all**, their pair alone below `cluster_min`.
A family of one pair stays below `cluster_min` and out of the cluster report; the run prints how many,
because a lone pair is a real find that only the pairs report can carry.

**How specific is a formula?** Length alone does not make a formula interesting: `salutem et
apostolicam benedictionem` and the `scripti patrocinio communimus` clause are both recurring formulas,
but only the second says anything about the charter it is in. Each cluster's headline formula is
scored by **length × rarity**:

- `CoreTokens` — how many words the formula has;
- `MeanIDF` — the average inverse document frequency of its **distinct** words, in nats, smoothed as `log((1 + N) / (1 + df))` so a word present in every charter scores `0` rather than going negative;
- `Specificity` — `CoreTokens × MeanIDF`.

`min_core_specificity` cuts below a value you choose, and `stopwords_file` zeroes out tokens you
already know are uninformative. Both act *before* cluster IDs are assigned, so cross-references from
the other TSVs stay valid.

**Watch the pair counts on a corpus with repeated files.** A corpus assembled from editions and
archival copies holds the same charter under several names — a shelfmark, an edition, a `(1)`
re-download. Since a pair is a pair of *files*, one comparison then appears as nine, and a cluster can
read "9 pair(s), 6 document(s)" while holding two texts. `DistinctTexts` and the
`(N distinct text(s))` note expose that; `deduplicate` removes the repeats upstream. The two are
independent: the note reports what is in the corpus, `deduplicate` decides what gets loaded.

**The summary block at the top of `clusters.html`** answers *on what input were these clusters
computed*: the mode and path or glob pattern, `file_suffix`, how many charters each side contributed
and how many files were found, loaded, skipped as too short, skipped as duplicates or cut off by
`keep_texts`, whether the threshold was automatic and by which method, the cluster threshold and
linkage, a timestamp, and the command line.

**Reading the charters themselves.** Unless the corpus is too large to embed (the report says so and
falls back to name lists), every charter in every cluster appears as a collapsible block with its
**complete text** and the shared formula in `<mark>`. The highlight maps the same `(start, size)` span
the clustering used back through the tokeniser, so it marks the words the algorithm matched rather
than a re-search for the string; a document carrying two related formulas has both marked.

**Tuning.** `cluster_threshold` sets how strictly "the same formula" is judged: `1.0` clusters only
verbatim-identical cores, `0.70` (the default) also tolerates the substitution of the template's
variable slots — a different recipient, a different place — so one template stays one cluster, and
`0.65` or below starts merging formulas that merely share a phrase. The default was lowered from
`0.85` after measuring both on the MOM corpus: at `0.85` a single papal privilege template whose
recipients differed fragmented into four clusters, at `0.70` it is one cluster and no other cluster of
four or more members merged. The core's *shape* is tuned separately: `core_gap_tolerance` widens or
narrows the slot the aligner may bridge, and `core_identity_threshold` filters the loose end of the
distribution. Both are reported in the input summary.

**Anchoring.** The window can be seeded on a performative verb through `core_anchors` — `donamus`,
`contulimus`, `confirmamus` — grown in both directions and capped at `core_max_tokens`, so a donation
is clustered on `donamus … duas partes decimarum` rather than on the papal greeting above it, and the
`Anchor` column names the verb. Supplied on the MOM corpus, anchoring moves the median core from 208
tokens to 82 (not to the 50-token ceiling: 153 of 441 pairs carry no anchor in their shared text and
keep the long window) and the formula clusters from 29 to 32, with the near-duplicate section growing
from 43 to 50 charter documents.

It ships **off** because the anchor set cannot be derived from the corpus. A hand-written
Latin-and-German list was tried first and does not hold up on the corpus it was written for: of its 38
stems, 11 (`vendidimus`, `vendimus`, `emancipauimus`, `concessimus`, `promittimus`, `assignauimus`,
`commutauimus`, `verleihen`, `verkaufen`, `ubergeben`, …) **do not occur in the corpus at all**, and
153 of the 441 admitted pairs (34.7%) share no word of it, so a third of the corpus fell back to the
unanchored window anyway. Automatic discovery was then tried and failed four ways:

| Attempt | Result |
| --- | --- |
| Distributional rules: document frequency, occurrences per document, context entropy, positional spread, neighbour IDF, and every conjunction of them | best rule selected **8676 tokens to find 45 verbs — precision 0.5%, recall 34.9%** |
| Phrase-level rarity: ranking a pair's shared windows by the document frequency of their 4-grams | the act lands at relative rank **0.467 — chance** |
| Rarity as the window's objective: mean, summed and rarest-word unigram IDF, and phrase DF | all **anti-select** the act (median relative rank 0.57–0.70) |
| Morphology, with the ending inventory of the known verbs handed over as an oracle | `-mus` / `-nt` / `-it` still selected **2215 tokens to find 50 verbs (2.3%)** |

The verbs are 0.3% of the recurring vocabulary and statistically ordinary — `contulimus` has document
frequency 416 beside `iuris` at 1093, in a mid-frequency band holding 1241 tokens of which 16 are
verbs — and rarity as an objective is the wrong sign, because the act formula is shared by its whole
act family while the average shared passage is a one-off list of names. Identifying a performative
verb is therefore a lexical or morphological task, and the anchor set is per-corpus input rather than
a constant in the code. Two notes for supplying one: stems match as *prefixes*, so a stem that is too
broad swallows the nouns derived from the verb (`dona` matches `donatio`, turning a donation act into
a donation word), and IDF is not the seed's objective — it only chooses between anchors the act is
already known to be near.

`cluster_min` filters out formulas that occur only once. On a corpus where the same charter is filed
under several names, `deduplicate` is usually the more useful first move: it removes the repeated
comparisons *before* they inflate the pair counts.

## Output column layout

> **Note — column layout changed.** `similarity_summary.tsv` and `linguistic_variations.tsv` carry a
> `ClusterID` column while `gen_clusters` is on (its default). A script reading those files by column
> *position* must be updated: the summary's `ClusterID` is the **second** column (right after
> `DocumentFilename`), the linguistic register's is the **third** (after `File_2`, before
> `Variation_Type`). Reading by **header name** is unaffected. `clusters.tsv` is a new file, with
> columns `ClusterID, Size, Anchor, CoreFormula, CoreFolded, Documents, DistinctTexts, CoreTokens,
> MeanIDF, Specificity, Cohesion, Members, PairCosine, PairCoreRatio, SharedTokens, Coverage, Kind`.
> Turning the option off restores the older layout, so long as it is turned off in the GUI or by
> setting the default in `flame.py` — the command line cannot switch a boolean *off*, since `-flag`
> arguments are presence switches (see [Command-Line Interface](CLI.md#command-line-interface-cli)).
