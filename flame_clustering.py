"""Clustering for FLAME -- grouping text pairs by the formula they share.

Ported from KONI (../KONI/app/flame_pure.py: ``_core_sequence``, ``_shingles``,
``cluster_pairs``), where the same machinery collapses the 20-30 near-identical
rows a formulaic-reuse search produces into a single cluster.

What a cluster means changes with the corpus, and the shift is deliberate:
in KONI the repeated unit is a formulaic phrase across a literary tradition; in
medieval charters it is the **legal formula or legal transaction** two documents
carry -- the same dispositive clause, the same corroboratio, the same
notification formula. The mechanism is identical; only the reading of the result
differs, and the report is worded for the charter case.

The one adaptation the charter corpus forces is in `build_token_pairs`: the
signature is taken on FLAME's *folded* tokens (see `fold_for_compare` in
flame.py), not on the raw ones. Medieval spelling varies inside a single
formula -- vnd/und, cz/tz, i/j/y, u/v/w, doubled letters, Dehnungs-h -- so a
signature taken on raw tokens would split one legal formula into as many
clusters as it has spellings. Folding first is what makes the cluster come out
as "the same formula despite the orthography".

This module is deliberately stdlib-only apart from rapidfuzz, and knows nothing
about FLAME's engine: it is handed already-tokenized, already-folded input and
returns plain dicts. That keeps it testable without a corpus or a model.
"""

import html
from difflib import SequenceMatcher
from typing import Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

from rapidfuzz.distance import Levenshtein

# Character k-gram length for the Jaccard prefilter (see _shingles).
SHINGLE_K = 4

# Gapped alignment defaults (see align_core). A formula's variable slot is a word
# or two -- a name, a place, a case ending -- so a tolerance of eight tokens
# already spans the widest slot seen in the corpus while refusing to bridge the
# distance between two genuinely different formulas.
GAP_TOLERANCE = 8
MIN_CORE_TOKENS = 12

# Anchored ("seed-and-extend") core extraction. A charter's legal act is carried
# by a performative verb -- donamus, contulimus, confirmamus -- while the longest
# shared run is almost always the protocol around it: intitulatio, salutatio,
# arenga. Maximising matched tokens therefore finds the template, not the act,
# and measured on the MOM corpus it produced no core below 56 tokens at all
# (median 208) while 28 of 40 clusters opened on a protocol phrase. Seeding the
# alignment on these verbs and bounding the extension around them is what puts
# the window on the act instead.
#
# The anchors are supplied by the caller (`core_anchors` in flame.py) and are
# matched as *prefixes* against the folded tokens, because mediaeval spelling
# varies the ending (donamus / donauimus / donauerunt) and folding keeps the
# corpus's own u-for-v spelling (contulimus, uendidimus). There is deliberately
# no built-in list here: see the measured reason at `core_anchors` in flame.py,
# and `align_core`'s anchoring paragraph for the fallback a pair without one
# takes.
ANCHOR_WINDOW = 15
MAX_CORE_TOKENS = 50


# --- signatures ---------------------------------------------------------------

def iter_token_pairs(display_tokens: Sequence[str], fold: Callable[[str], str]
                     ) -> Iterator[Tuple[int, str, str]]:
    """The kept tokens as (index in `display_tokens`, display, folded).

    The index is what lets a core found in pair space be pointed back at the
    words of the *original* text: `core_sequence` reports a core by its position
    among the kept tokens, which is not the position among the tokenizer's
    tokens, because punctuation was dropped. Rebuilding a readable charter means
    putting that punctuation back, so both indices are needed.
    """
    for index, token in enumerate(display_tokens):
        if not token.isalnum():
            continue
        folded = fold(token.lower())
        if not folded or any(char.isspace() for char in folded):
            continue
        yield index, token, folded


def build_token_pairs(display_tokens: Sequence[str],
                      fold: Callable[[str], str]) -> List[Tuple[str, str]]:
    """Aligns display tokens with their folded forms: [(display, folded), ...].

    Punctuation is dropped (`token.isalnum()`), which mirrors what the
    highlighter does in SimilarityVisualizer.highlight_similarities, so a core
    reported here names exactly the words the side-by-side view highlights.

    A token is also dropped when folding leaves nothing behind, or leaves
    whitespace: `fold` maps every character outside a-z to a space, so a token
    carrying digits or non-Latin letters ("a1b" -> "a b") folds to something
    that is no longer a single comparable word. Such a token is not formulaic
    Latin, and keeping it would let two folded forms match at a space boundary
    that means nothing.
    """
    return [(display, folded) for _index, display, folded in iter_token_pairs(display_tokens, fold)]


def core_sequence(pairs1: Sequence[Tuple[str, str]],
                  pairs2: Sequence[Tuple[str, str]]) -> Tuple[str, str, int, int]:
    """The pair's *formula core*: its longest contiguous run of matching words.

    Returns (core_folded, core_display, start, length); an all-empty core when
    the two texts share no word, which the caller reads as "no formula".

    Two adaptations from KONI's ``_core_sequence``, both simplifications:

    * KONI walks each fuzzy block tracking runs of consecutive word indices,
      because its blocks may hold non-contiguous matches. `difflib`'s
      ``get_matching_blocks()`` already returns *maximal contiguous* blocks, so
      the longest block IS the longest run -- no inner loop needed.
    * Ties go to the first maximum, matching KONI's "first maximum wins", which
      is what makes the result deterministic across runs. Blocks arrive in
      diagonal order, so this is a positional rule, not a claim that the first
      match is the strongest one.

    The comparison runs on the folded side; the returned ``core_display`` is the
    same span in the original spelling, so a diplomatist can read the formula as
    the charter writes it.
    """
    folded1 = [folded for _display, folded in pairs1]
    folded2 = [folded for _display, folded in pairs2]
    if not folded1 or not folded2:
        return "", "", 0, 0

    matcher = SequenceMatcher(None, folded1, folded2, autojunk=False)
    best_start = best_size = 0
    for a, _b, size in matcher.get_matching_blocks():
        if size > best_size:
            best_size, best_start = size, a
    if not best_size:
        return "", "", 0, 0

    core = pairs1[best_start:best_start + best_size]
    return (" ".join(folded for _display, folded in core),
            " ".join(display for display, _folded in core),
            best_start, best_size)


class CoreAlignment(NamedTuple):
    """A pair's formula core, aligned in both documents.

    `start`/`size` address the window among *pairs1*'s kept tokens, `start2`/
    `size2` the same window among *pairs2*'s. Both come from one alignment, so
    they name the same formula by construction: the caller no longer has to
    re-find the core from the other side and hope it lands on the same words.

    `identity` is the gap-only ratio 2*matched/(span1+span2) -- the same
    normalisation as `_lev_ratio`, so it is commensurable with the similarity
    threshold. `gaps` counts the unmatched words *inside* the window (the
    variable slots the formula carries), and `degraded` says the window failed
    `min_tokens`/`identity_floor` and fell back to the strict contiguous core.

    `overlap` is how many words the pair shares over the WHOLE aligned span,
    before `max_tokens` narrows the reported window. The two are different
    questions and the caller needs both: the reported window is the *formula*,
    while the near-duplicate test asks how much of the shorter charter the pair
    covers, which a 50-token window can no longer answer. `anchor` names the
    performative verb the window was seeded on, or is empty for a pair whose
    shared text carries none (see `core_anchors` in flame.py).
    """
    folded: str
    display: str
    start: int
    size: int
    start2: int
    size2: int
    identity: float
    gaps: int
    degraded: bool
    overlap: int = 0
    anchor: str = ""


EMPTY_ALIGNMENT = CoreAlignment("", "", 0, 0, 0, 0, 0.0, 0, True)


def _anchor_hits(folded: Sequence[str], anchors: Sequence[str]) -> List[Tuple[int, str]]:
    """Every position a performative verb occupies in a folded token sequence.

    Prefix matching, because the ending is what varies (donamus / donauimus /
    donauerunt) and because folding leaves the corpus's own u-for-v spelling
    alone (contulimus, uendidimus). The matched *token* is kept alongside the
    index: it is what the report names the cluster's formula by.
    """
    return [(index, token) for index, token in enumerate(folded)
            if any(token.startswith(anchor) for anchor in anchors)]


def _nearest_block(blocks: Sequence[Tuple[int, int, int]], index: int, side: int) -> int:
    """The block closest to a position, measured inside the block, not to its start.

    An anchor verb the other charter spells differently is not matched, so it
    falls in a gap and belongs to no block at all. Attaching it to the nearest
    block is what keeps such a pair anchored instead of silently falling back.
    """
    best: Optional[Tuple[int, int]] = None
    for k, (a, b, size) in enumerate(blocks):
        here = a if side == 1 else b
        distance = (0 if here <= index < here + size
                    else min(abs(index - here), abs(index - (here + size - 1))))
        if best is None or distance < best[0]:
            best = (distance, k)
    return best[1]


def _seed_blocks(blocks: Sequence[Tuple[int, int, int]],
                 hits1: Sequence[Tuple[int, str]], hits2: Sequence[Tuple[int, str]]
                 ) -> List[Tuple[int, int, int, str]]:
    """One seed per anchor occurrence: (block, its index in each sequence, the token).

    A matching block pairs tokens one to one, so the anchor's offset inside the
    block is the same on both sides and the verb can be located in *both*
    charters from one hit. An anchor outside the block it attached to is clamped
    into it, which can put the window a word or two off -- and is why the anchor
    the report names is the token the diplomatist reads in the charter, not an
    offset they would have to trust.
    """
    seeds = []
    for index, token in hits1:
        k = _nearest_block(blocks, index, 1)
        a, b, size = blocks[k]
        offset = max(0, min(index - a, size - 1))
        seeds.append((k, a + offset, b + offset, token))
    for index, token in hits2:
        k = _nearest_block(blocks, index, 2)
        a, b, size = blocks[k]
        offset = max(0, min(index - b, size - 1))
        seeds.append((k, a + offset, b + offset, token))
    return seeds


def _extend(blocks: Sequence[Tuple[int, int, int]], seed: int,
            gap_tolerance: int) -> List[int]:
    """The maximal chain of blocks reachable from a seed by gaps within tolerance.

    Both directions, one pass each, because the blocks arrive in increasing order
    in both sequences: the moment the gap to the next block exceeds the
    tolerance, no later one can qualify either. This is the "extend" half of
    seed-and-extend, and the chain it returns is the formula the seed sits in --
    it cannot wander off to the protocol, because the protocol is further away
    than the tolerance.
    """
    chain = [seed]
    k = seed
    while k > 0:
        a, b, _size = blocks[k]
        pa, pb, psize = blocks[k - 1]
        if a - (pa + psize) > gap_tolerance or b - (pb + psize) > gap_tolerance:
            break
        k -= 1
        chain.append(k)
    k = seed
    while k < len(blocks) - 1:
        a, b, size = blocks[k]
        na, nb, _nsize = blocks[k + 1]
        if na - (a + size) > gap_tolerance or nb - (b + size) > gap_tolerance:
            break
        k += 1
        chain.append(k)
    return sorted(chain)


def _anchored_chain(blocks: Sequence[Tuple[int, int, int]],
                    hits1: Sequence[Tuple[int, str]], hits2: Sequence[Tuple[int, str]],
                    gap_tolerance: int, max_tokens: int,
                    idf: Optional[Callable[[str], float]], folded1: Sequence[str]
                    ) -> Optional[Tuple[float, int, str, List[int], Tuple[int, int], Tuple[int, int]]]:
    """The best anchor's chain: (score, span, token, chain, window1, window2).

    Every anchor occurrence is extended into its own chain and the chains compete
    on the summed IDF of the window they would produce. Which verb carries the
    act is not fixed -- a donation says `donamus` in the dispositio and
    `confirmamus` in the sanctio -- so a charter offers several, and the window
    belongs on the one surrounded by the vocabulary of the transaction rather
    than by chancery filler.

    IDF decides *here* and nowhere else, and the measurement is why: making
    rarity the objective of the alignment itself selects the enumerations of
    proper names, because a list of bishoprics scores 7.18 against a formula's
    2.58, and such a list is the one passage no two charters share -- the exact
    opposite of what a cluster is for. The anchor keeps the window on the act;
    IDF only chooses among the anchors the act is already known to be near.

    Without an `idf` callable every chain scores 0.0 and the tie-break is the
    total span and then the anchors' own order, so the result stays deterministic
    for a caller that has no corpus statistics to hand.
    """
    best = None
    for k, anchor1, anchor2, token in _seed_blocks(blocks, hits1, hits2):
        chain = _extend(blocks, k, gap_tolerance)
        first, last = blocks[chain[0]], blocks[chain[-1]]
        start, start2 = first[0], first[1]
        size = last[0] + last[2] - start
        size2 = last[1] + last[2] - start2
        window1 = _anchor_window(anchor1, start, size, max_tokens)
        window2 = _anchor_window(anchor2, start2, size2, max_tokens)
        score = (sum(idf(t) for t in folded1[window1[0]:window1[0] + window1[1]])
                 if idf else 0.0)
        candidate = (score, window1[1] + window2[1], token, chain, window1, window2)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best


def _anchor_window(anchor: int, start: int, size: int, max_tokens: int) -> Tuple[int, int]:
    """The reported window: `max_tokens` wide, centred on the anchor, inside the span.

    Centring is what makes the cap mean something. Trimming the chain to its
    first `max_tokens` tokens would cut the window at whichever end the alignment
    happened to start, and would throw the anchor away whenever the verb sits
    late in the formula; centring keeps the verb in the middle of what the report
    marks. `max_tokens` of 0 leaves the span alone, which is the pre-anchor
    behaviour and the caller's way of asking for it.
    """
    if max_tokens <= 0 or size <= max_tokens:
        return start, size
    half = max_tokens // 2
    return max(start, min(anchor - half, start + size - max_tokens)), max_tokens


def _best_anchor(blocks: Sequence[Tuple[int, int, int]], chain: Sequence[int],
                 hits1: Sequence[Tuple[int, str]], hits2: Sequence[Tuple[int, str]],
                 start: int, size: int, start2: int, size2: int,
                 max_tokens: int, idf: Optional[Callable[[str], float]],
                 folded1: Sequence[str]) -> Optional[Tuple[float, str, Tuple[int, int], Tuple[int, int]]]:
    """The anchor the window is reported on: the one whose words are the rarest.

    Which verb carries the act is not fixed -- a donation says `donamus` in the
    dispositio and `confirmamus` in the sanctio -- so a charter offers several,
    and the window should sit on the one surrounded by the vocabulary of the
    transaction rather than by chancery filler. The score is the summed IDF of
    the words the anchor's window would contain.

    IDF decides *here* and nowhere else, and the measurement is why: maximising
    mean IDF over the whole alignment -- making rarity the objective -- selects
    the enumerations of proper names, because a list of bishoprics scores 7.18
    against a formula's 2.58 and is the one passage no two charters share. The
    anchor is what keeps the window on the act; IDF only chooses between the
    candidate anchors the act is already known to be near.
    """
    candidates = [(index, token, 1) for index, token in hits1 if start <= index < start + size]
    candidates += [(index, token, 2) for index, token in hits2 if start2 <= index < start2 + size2]
    best: Optional[Tuple[float, str, Tuple[int, int], Tuple[int, int]]] = None
    for index, token, side in candidates:
        other = _map_anchor(blocks, chain, index, side)
        if other is None:
            continue
        anchor1, anchor2 = (index, other) if side == 1 else (other, index)
        window1 = _anchor_window(anchor1, start, size, max_tokens)
        window2 = _anchor_window(anchor2, start2, size2, max_tokens)
        score = sum(idf(t) for t in folded1[window1[0]:window1[0] + window1[1]]) if idf else 0.0
        if best is None or score > best[0]:
            best = (score, token, window1, window2)
    return best


def align_core(pairs1: Sequence[Tuple[str, str]], pairs2: Sequence[Tuple[str, str]],
               gap_tolerance: int = GAP_TOLERANCE, min_tokens: int = MIN_CORE_TOKENS,
               identity_floor: float = 0.0, anchors: Sequence[str] = (),
               anchor_window: int = ANCHOR_WINDOW, max_tokens: int = 0,
               idf: Optional[Callable[[str], float]] = None) -> CoreAlignment:
    """The pair's formula core as a *gapped local alignment*.

    Why not the longest contiguous run (`core_sequence`): a mediaeval formula is
    a template with variable slots, so a single inserted or substituted word
    ("uestro auctoritate" against "auctoritate") splits one formula into two
    runs, and the strict rule reports whichever half is longer. Chaining the
    runs back together is the whole point here.

    The chain: `difflib` returns the maximal contiguous match blocks between the
    two folded token sequences; a core is the span of a subsequence of those
    blocks whose consecutive gaps stay within `gap_tolerance` tokens on *both*
    sides, and among the admissible chains the one carrying the most matched
    tokens wins. That is a Smith-Waterman local alignment restricted to exact
    blocks -- measured against a token-level Smith-Waterman on the real corpus it
    picks the same windows (median 107 tokens, identical to SW's median) at a
    fraction of the cost, which matters because the corpus run calls this once
    per admitted pair.

    Ties go to the longer window and then to the earliest one, so the result is
    deterministic across runs and across hash seeds.

    A window that fails `min_tokens` (its shorter side) or `identity_floor`
    degrades to `core_sequence`'s contiguous core and is flagged, so a weak pair
    still clusters exactly as it did before this change and the report can say
    how many pairs were served by the old rule.

    **Anchoring.** With `anchors` given, the chain is restricted to blocks near a
    performative verb (`_anchored_blocks`) and then narrowed to `max_tokens`
    around the best of them (`_best_anchor`). Maximising matched tokens finds the
    longest shared run, which in a mediaeval charter is the protocol -- measured
    on the MOM corpus, no core came out below 56 tokens and 28 of 40 clusters
    opened on a protocol phrase -- so the verb is what tells the aligner where
    the legal act is. A pair whose shared text carries no anchor keeps the plain
    chain and reports an empty `anchor`; that is a fallback, not a failure, and
    the caller counts it.
    """
    folded1 = [folded for _display, folded in pairs1]
    folded2 = [folded for _display, folded in pairs2]
    if not folded1 or not folded2:
        return EMPTY_ALIGNMENT

    matcher = SequenceMatcher(None, folded1, folded2, autojunk=False)
    blocks = [(a, b, size) for a, b, size in matcher.get_matching_blocks() if size]
    if not blocks:
        return EMPTY_ALIGNMENT

    # Seed: the anchor verbs, if any, decide where the window will sit. A pair
    # whose shared text carries none falls through to the chain below and is the
    # fallback the report counts -- measured on the MOM corpus, 59.4% of the real
    # pairs carry one, so the fallback is a minority path, not a rare one.
    hits1 = _anchor_hits(folded1, anchors) if anchors else []
    hits2 = _anchor_hits(folded2, anchors) if anchors else []

    # Max-weight chain, one pass: best[k] is the matched-token count of the best
    # chain ending at block k. Blocks arrive in increasing order in *both*
    # sequences, so as the predecessor index decreases both gaps only grow --
    # once one exceeds the tolerance no earlier block can qualify either, and the
    # scan stops. That keeps this O(blocks) instead of O(blocks^2).
    best = [0] * len(blocks)
    prev = [-1] * len(blocks)
    for k, (a, b, size) in enumerate(blocks):
        best[k] = size
        for p in range(k - 1, -1, -1):
            pa, pb, psize = blocks[p]
            if a - (pa + psize) > gap_tolerance or b - (pb + psize) > gap_tolerance:
                break
            if best[p] + size > best[k]:
                best[k] = best[p] + size
                prev[k] = p
    end = max(range(len(blocks)), key=lambda k: (best[k], blocks[k][2], -k))

    chain = []
    k = end
    while k != -1:
        chain.append(k)
        k = prev[k]
    chain.reverse()
    first, last = blocks[chain[0]], blocks[chain[-1]]
    start, start2 = first[0], first[1]
    size = last[0] + last[2] - start
    size2 = last[1] + last[2] - start2
    matched = sum(blocks[k][2] for k in chain)
    identity = 2 * matched / (size + size2) if (size + size2) else 0.0

    if min(size, size2) < min_tokens or identity < identity_floor:
        # Degrade to the strict rule: `core_sequence`'s longest single block. The
        # window in the second document is looked up from the other side and
        # kept only if it lands on the same words -- a single block leaves no
        # alignment to inherit the second span from, and marking the wrong
        # passage of a charter is worse than marking nothing. A single block
        # matched end to end, so its identity is 1.0 by definition.
        core = core_sequence(pairs1, pairs2)
        reverse = core_sequence(pairs2, pairs1)
        start2 = size2 = 0
        if reverse[0] == core[0]:
            start2, size2 = reverse[2], reverse[3]
        return CoreAlignment(core[0], core[1], core[2], core[3], start2, size2,
                             1.0, 0, True, min(core[3], size2), "")

    # How much the pair shares over the whole aligned span. This is the
    # *max-weight* chain's span, taken before anchoring replaces the chain: the
    # near-duplicate test asks how much of the shorter charter the pair covers,
    # and neither the formula window nor the anchor's own (shorter) chain can
    # answer it -- measuring the duplicate test on the anchored chain would drop
    # a copied-out charter below the 400-token threshold purely because the copy
    # happens to contain a performative verb.
    overlap = min(size, size2)
    anchor_label = ""
    if hits1 or hits2:
        picked = _anchored_chain(blocks, hits1, hits2, gap_tolerance, max_tokens, idf, folded1)
        if picked is not None:
            # The anchor's own chain replaces the max-weight one: that is the
            # "seed and extend" answer, and the only one that cannot walk back to
            # the protocol, because the protocol lies further from the verb than
            # the gap tolerance reaches.
            _score, _span, anchor_label, chain, (start, size), (start2, size2) = picked
            # The window moved, so its identity has to be re-measured on what the
            # window now contains: the max-weight chain's count included the words
            # the window no longer covers, and reporting that number would credit
            # the formula with matches the report does not mark.
            matched = sum(max(0, min(a + bsize, start + size) - max(a, start))
                          for k in chain for a, _b, bsize in [blocks[k]])
            identity = 2 * matched / (size + size2) if (size + size2) else 0.0

    window = pairs1[start:start + size]
    return CoreAlignment(
        " ".join(folded for _display, folded in window),
        " ".join(display for display, _folded in window),
        start, size, start2, size2,
        identity, size + size2 - 2 * matched, False, overlap, anchor_label)


def _lev_ratio(a: str, b: str) -> float:
    """KONI's `levenshtein_ratio`: `1 - dist / (len(a) + len(b))`.

    NOT `rapidfuzz.distance.Levenshtein.normalized_similarity`, which divides by
    `max(len(a), len(b))` instead -- a different metric, and a *lower* score for
    unequal lengths ('abcd' vs 'abce' is 0.875 here but 0.75 there). The default
    threshold is calibrated against KONI, and the length-window prune in
    `cluster_pairs` already reasons in this convention
    (`|la-lb| / (la+lb) > 1 - threshold`), so using the other normalisation would
    make the two halves of the same filter disagree. `rapidfuzz.fuzz.ratio`
    (Indel) is a third metric again; do not substitute it either.
    """
    if not a and not b:
        return 1.0
    return 1.0 - Levenshtein.distance(a, b) / (len(a) + len(b))


def _shingles(s: str, k: int = SHINGLE_K) -> frozenset:
    """Character k-grams -- a cheap proxy for edit similarity.

    A string of length exactly k would yield the single k-gram `s` itself, so
    two such strings differing in one character would share NO gram and the
    Jaccard gate would drop a pair the edit ratio accepts ('abcd' vs 'abce':
    ratio 0.875 but zero common 4-grams). Step down to k-1 in that case so short
    cores can still overlap. Cores produced by `core_sequence` are word
    sequences and in practice always longer than k, so this only guards direct
    callers and short inputs.
    """
    if len(s) < k:
        return frozenset((s,))
    if len(s) == k:
        k -= 1
    return frozenset(s[i:i + k] for i in range(len(s) - k + 1))


def maximal_cliques(adjacency: Dict[int, set]) -> List[frozenset]:
    """Every maximal clique of the graph, by Bron-Kerbosch with pivoting.

    A strict cluster has to satisfy a *pairwise* claim -- every core in it within
    `threshold` of every other -- and that claim is exactly a clique. Union-find
    computes the transitive closure instead, which also merges A with C when only
    A~B and B~C hold. Measured on the MOM corpus, that chained one cluster of 10
    cores whose 45 core pairs were below the 0.85 threshold 28 times. The clique
    answer is stricter on purpose: a cluster that survives is one whose every
    member carries the same formula, not one merely reachable from it.

    Maximal cliques overlap, so a core can belong to several; this function only
    enumerates them, and the caller decides where each pair lands. The worst case
    is exponential, but the clusters are small (ten cores was the largest
    measured) and `cluster_pairs` refuses to refine a group past
    `max_cores_per_cluster` anyway.
    """
    cliques: List[frozenset] = []
    nodes = sorted(adjacency)

    def expand(current: set, candidates: set, excluded: set) -> None:
        if not candidates and not excluded:
            cliques.append(frozenset(current))
            return
        pivot = max(candidates | excluded, key=lambda u: len(adjacency[u] & candidates))
        for node in sorted(candidates - adjacency[pivot]):
            expand(current | {node}, candidates & adjacency[node], excluded & adjacency[node])
            candidates = candidates - {node}
            excluded = excluded | {node}

    expand(set(), set(nodes), set())
    return cliques


def _pairwise_similarity(cores: Sequence[str], core_ids: Sequence[int]
                         ) -> Dict[Tuple[int, int], float]:
    """Exact similarity for every unordered pair of the given cores.

    No Jaccard gate and no `max_lev` cap here, unlike the stage-2 loop: this runs
    over the few cores of one already-formed group, and the point of the strict
    mode is a *complete* pairwise matrix rather than a gated approximation of it.
    """
    return {(a, b): _lev_ratio(cores[a], cores[b])
            for index, a in enumerate(core_ids) for b in core_ids[index + 1:]}


def _build_cluster(members: List[int], cores: Sequence[str], uid: Sequence[int]) -> Dict:
    """Assembles one cluster dict from its pair indices.

    The first member is the cluster's reference: its core is the headline
    `CoreFormula`, and every other pair carries `core_ratio`, that pair's core
    similarity to the headline core. `core_ratio` is not the pair's cosine -- the
    cosine is what admitted the pair to the comparison, `core_ratio` is what the
    clustering made of it -- and it is where a chain shows up, as values below
    `threshold` for pairs that hang off the cluster only through a third one.

    `cohesion` is the minimum similarity over all *distinct core* pairs in the
    cluster. A single-core cluster is 1.0 by definition: there is one core, so
    nothing can disagree with it.
    """
    core_ids = sorted({uid[p] for p in members})
    similarity = _pairwise_similarity(cores, core_ids)
    reference = uid[members[0]]
    return {
        "size": len(members),
        "members": members,
        "n_cores": len(core_ids),
        "core_ratio": [1.0 if uid[p] == reference
                       else similarity[(min(uid[p], reference), max(uid[p], reference))]
                       for p in members],
        "cohesion": min(similarity.values() or [1.0]),
        "shared_with": [],
    }


def _split_into_cliques(core_ids: List[int], cores: Sequence[str], members: List[int],
                        uid: Sequence[int], threshold: float) -> List[Dict]:
    """Splits one union-find group into its maximal cliques of cores.

    A maximal clique is the largest set of cores that agree pairwise, and those
    sets overlap: in A~B~C with A and C too far apart, the cliques are {A,B} and
    {B,C}, and B's pair belongs to both. Overlap is kept rather than resolved,
    because the alternative -- giving each pair to one clique only -- makes {B,C}
    shrink to C alone and drop below `min_size`, so a pair that *did* clear the
    threshold vanishes from the report. A charter carrying two related formulas is
    exactly what the ClusterID column of the summary TSVs already expresses with
    a comma-separated list, so the shape is not new.

    Cores left in no clique of at least two are genuinely unpaired under the
    strict reading, and their pairs fall out of the clustering: that is the chains
    being broken, not a loss.
    """
    similarity = _pairwise_similarity(cores, core_ids)
    adjacency: Dict[int, set] = {core: set() for core in core_ids}
    for (a, b), ratio in similarity.items():
        if ratio >= threshold:
            adjacency[a].add(b)
            adjacency[b].add(a)

    cliques = [c for c in maximal_cliques(adjacency) if len(c) >= 2]
    cliques.sort(key=lambda c: (-len(c), min(c)))
    return [_build_cluster([p for p in members if uid[p] in clique], cores, uid)
            for clique in cliques]


# --- document rendering -------------------------------------------------------
# The cluster report shows the formula *in the charter it was taken from*, so the
# reader can see what the word list is. That needs the punctuation back, which
# `build_token_pairs` threw away, and it needs the marked run in the right place,
# which is why the span is carried through as token indices rather than as text.

# A token starting with one of these is glued to the previous one; a token that
# IS one of the openers does not take a space after it. The rules are a small,
# fixed approximation of a detokenizer: enough for Latin charter prose, and --
# unlike a real detokenizer -- cheap to keep bit-stable, which the tests rely on.
_NO_SPACE_BEFORE = set('.,;:!?)]}»›"’”')
_NO_SPACE_AFTER = set('([{«‹"‘“')


def render_tokens(tokens: Sequence[str], spans: Sequence[Tuple[int, int]] = ()) -> str:
    """Rebuilds readable, HTML-escaped text from tokens, marking `spans`.

    `spans` are (start, end) half-open ranges of *token* indices; every token
    inside one is wrapped in `<mark>`, including punctuation that happens to sit
    between two marked words. Marking a core word by word would otherwise break
    the highlight at every comma the tokenizer split off.
    """
    marked = [False] * len(tokens)
    for start, end in spans:
        for index in range(max(0, start), min(len(tokens), end)):
            marked[index] = True

    out: List[str] = []
    previous = ""
    for index, token in enumerate(tokens):
        if previous:
            if not (token[:1] in _NO_SPACE_BEFORE or previous in _NO_SPACE_AFTER):
                out.append(" ")
        if marked[index] and (index == 0 or not marked[index - 1]):
            out.append("<mark>")
        out.append(html.escape(token))
        if marked[index] and (index == len(tokens) - 1 or not marked[index + 1]):
            out.append("</mark>")
        previous = token
    return "".join(out)


# --- clustering ---------------------------------------------------------------

def _louvain_groups(U: int, edges: Sequence[Tuple[int, int, float]],
                    uid: Sequence[int], n: int) -> List[List[int]]:
    """Density-based communities over the core graph, as member-pair lists.

    The strict linkage (`clique`) demands that every core in a cluster agree with
    every other one, and `union` demands only a path between them. Measured on
    the MOM corpus neither fits a corpus-wide formula: the papal *confirmatio*
    ran through seventeen charters as one connected component that strict cliques
    then cut into five, because a clique is an all-or-nothing object and a real
    template varies at its slots, so the "agree with everyone" edges do not
    close into a triangle everywhere.

    Louvain optimises modularity instead -- roughly, it looks for groups that are
    denser inside than between -- so it can return the formula as one community
    while still separating two formulas that happen to share a phrase. Edge
    weight is the core ratio itself, so a strong match pulls harder than a
    marginal one.

    Determinism: node order is the core order (which follows first appearance in
    the pair list), edge order is the scan order above, and the algorithm's own
    randomness is pinned by the seed. The same corpus therefore gives the same
    communities on every run, which the report's cluster ids depend on.
    """
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "The 'louvain' linkage needs networkx, which is not installed. "
            "Install it (pip install networkx) or choose -cluster_linkage "
            "'clique' / 'union'.") from exc

    graph = nx.Graph()
    graph.add_nodes_from(range(U))
    for ia, ib, ratio in edges:
        graph.add_edge(ia, ib, weight=ratio)
    communities = nx.algorithms.community.louvain_communities(
        graph, weight="weight", seed=0)

    # A community is a set of cores; a cluster is the pairs whose core is in it.
    core_to_community: Dict[int, int] = {}
    for community_id, community in enumerate(sorted(communities, key=lambda c: min(c))):
        for core_id in community:
            core_to_community[core_id] = community_id

    by_community: Dict[int, List[int]] = {}
    for p in range(n):
        by_community.setdefault(core_to_community[uid[p]], []).append(p)
    return [by_community[c] for c in sorted(by_community)]


def cluster_pairs(sigs: Sequence[str], threshold: float = 0.70, min_size: int = 2,
                  max_lev: int = 200000, linkage: str = "clique",
                  max_cores_per_cluster: int = 30,
                  forced_groups: Sequence[Sequence[int]] = ()) -> Dict[str, object]:
    """Group pairs that carry the same formulaic core into one cluster.

    Three stages, exactly as in KONI:

      1. pairs whose core is *verbatim* identical collapse in one dict pass.
         Formulaic reuse is overwhelmingly of this kind, and this stage is exact
         and complete -- it cannot lose recall;
      2. the *distinct* cores are then compared against each other rather than
         the pairs against each other. A repeated formula makes the pair-wise
         loop re-run the same core-vs-core test over and over: measured in KONI
         on a 3966-pair self-compare, the pair-wise loop performed 3222
         Levenshtein comparisons covering only 215 distinct core pairs (93%
         repeats). Comparing cores removes that redundancy at zero recall cost;
      3. within stage 2, candidates survive a sorted-length window and a 4-gram
         Jaccard gate before the Levenshtein DP runs.

    Both stage-3 gates are *heuristic*: the length window is sound, but the
    Jaccard gate can drop a core pair whose edit ratio clears `threshold` while
    its 4-grams diverge. Rejections are counted in `gate_skips` so a caller can
    see how much was filtered, and `truncated` flags the separate `max_lev` cap
    on the DP. This function does NOT claim to compute the exact transitive
    closure of the "ratio >= threshold" graph -- it computes it over the pairs
    the gate lets through. A low `gate_skips` is evidence the gate did little; a
    high one is a reason to re-run with the gate loosened.

    The ratio is KONI's own `levenshtein_ratio` (1 - dist / (len(a) + len(b))),
    reimplemented in `_lev_ratio` on top of rapidfuzz's raw distance, so the
    threshold keeps the meaning KONI calibrated it with. KONI shipped 0.85; the
    default here is 0.70, measured on the MOM corpus (see `flame.DEFAULT_PARAMS`
    for the numbers). Two nearby metrics have to be kept
    out: `rapidfuzz.distance.Levenshtein.normalized_similarity` divides by
    max(len(a), len(b)) and `rapidfuzz.fuzz.ratio` is Indel similarity -- both
    would silently move the threshold.

    Cost: the gate loop is O(U^2) in the number of *distinct* cores (U) and is
    NOT bounded by `max_lev`, which bounds only the DP.

    `linkage` decides what a cluster is:

    * `"union"` -- KONI's own answer. Union-find over the cores, so the clusters
      are the connected components of the "ratio >= threshold" graph. Two pairs
      with nothing in common end up together when a third is near both: the
      classic chaining artifact, and on the MOM corpus it produced one cluster of
      10 cores whose 45 core pairs failed the threshold 28 times.
    * `"clique"` -- each group is re-examined with every core pair scored
      *exactly* (no gate, no cap) and split into maximal cliques, so a cluster
      holds only cores that all agree with each other pairwise. Groups bigger
      than `max_cores_per_cluster` are left to union-find rather than refined,
      and counted in `n_unrefined`, so a pathological corpus degrades into
      KONI's behaviour instead of into a long-running clique search.
    * `"louvain"` (the default) -- modularity communities over the *complete*
      threshold graph (`_louvain_groups`). Cliques tear a real template apart,
      because a formula varies at its slots and "agrees with everyone" then fails
      to close into triangles; union-find chains unrelated formulas through a
      shared phrase. Louvain asks which cores are denser among themselves than
      with the rest, which is the question the diplomatist is actually asking.
      Unlike the other two, this linkage needs the whole edge set, so stage 2
      keeps every qualifying edge instead of the union-find spanning subset.

    Either way each cluster carries `core_ratio` (per pair, against the cluster's
    reference core) and `cohesion` (the worst core pair inside it), so a reader
    can tell a tight cluster from a chained one without re-running anything.
    """
    n = len(sigs)
    empty_stats = {"n_pairs": 0, "n_cores": 0, "n_clusters": 0, "n_clustered": 0,
                   "n_singletons": 0, "gate_skips": 0, "lev_compares": 0,
                   "truncated": False, "linkage": linkage,
                   "n_groups_split": 0, "n_unrefined": 0, "n_overlapping": 0,
                   "n_forced_pairs": 0, "n_forced_groups": 0, "n_forced_lone": 0}
    if n == 0:
        return {"clusters": [], "pair_cluster": [], "stats": dict(empty_stats)}

    # Collapse to distinct cores. `uid` maps each pair to its core's id; the
    # union-find then runs over cores, and pair indices are folded in at the end.
    uid_of_core: Dict[str, int] = {}
    uid = [0] * n
    for p, s in enumerate(sigs):
        i = uid_of_core.get(s)
        if i is None:
            i = uid_of_core[s] = len(uid_of_core)
        uid[p] = i
    cores = [None] * len(uid_of_core)
    for s, i in uid_of_core.items():
        cores[i] = s
    U = len(cores)
    # Forced duplicate edges: pair indices that the caller's own duplicate rule
    # already qualifies (see `duplicate_groups`), grouped into charter families.
    # Cluster membership is decided from the *cores*, and a core narrowed to the
    # legal act can match nothing even between two copies of one charter --
    # measured on the MOM corpus, six charter documents sharing 400+ words fell
    # out of the report this way, each pair left alone below `min_size`. The
    # caller's rule reads the pair's whole shared span, which no window touches,
    # so it is the caller's verdict that decides, not the core's similarity.
    n_forced_pairs = sum(len(g) for g in forced_groups)
    n_forced_lone = sum(1 for g in forced_groups if len(g) == 1)

    uparent = list(range(U))

    def ufind(x: int) -> int:
        while uparent[x] != x:
            uparent[x] = uparent[uparent[x]]
            x = uparent[x]
        return x

    def uunion(a: int, b: int) -> None:
        ra, rb = ufind(a), ufind(b)
        if ra != rb:
            # Smaller root wins, so cluster ids stay deterministic regardless of
            # the order the unions happen to arrive in.
            uparent[max(ra, rb)] = min(ra, rb)

    # stage 2 -- near-duplicate cores, prefiltered
    by_len = sorted(range(U), key=lambda i: len(cores[i]))
    sh = [_shingles(s) for s in cores]
    lev_compares = 0
    gate_skips = 0
    truncated = False
    # Every core pair that clears the threshold, kept as an edge list. The
    # union-find paths below only need connectivity, but a community algorithm
    # reads the graph's whole structure -- and the `ufind` short-circuit below
    # would hand it a spanning subset of the edges instead, which is a different
    # graph with different communities.
    edges: List[Tuple[int, int, float]] = []
    # The forced duplicate families go in first, at full weight: by the caller's
    # own rule these texts are near-identical, so no similarity measure of the
    # narrowed cores should be allowed to keep them apart. A star rather than a
    # clique -- same connectivity, fewer edges.
    for group in forced_groups:
        ids = sorted({uid[p] for p in group if 0 <= p < n})
        for other in ids[1:]:
            edges.append((ids[0], other, 1.0))
            if linkage != "louvain":
                uunion(ids[0], other)
    for a in range(U):
        ia = by_len[a]
        la = len(cores[ia])
        ha = sh[ia]
        if la < 2:
            continue
        for b in range(a + 1, U):
            ib = by_len[b]
            lb = len(cores[ib])
            # |la-lb| alone puts the ratio out of reach, and lengths are
            # non-decreasing here -> no further candidate can qualify either.
            if (lb - la) / (la + lb) > 1.0 - threshold:
                break
            if linkage != "louvain" and ufind(ia) == ufind(ib):
                continue
            hb = sh[ib]
            inter = len(ha & hb)
            # 4-gram Jaccard as a cheap proxy for edit similarity. The `not
            # inter` short-circuit is not a separate decision: inter == 0 fails
            # the ratio test below too, it just avoids the division.
            if not inter or 2 * inter / (len(ha) + len(hb)) < threshold * 0.5:
                gate_skips += 1
                continue
            if lev_compares >= max_lev:
                truncated = True
                break
            lev_compares += 1
            ratio = _lev_ratio(cores[ia], cores[ib])
            if ratio >= threshold:
                edges.append((ia, ib, ratio))
                if linkage != "louvain":
                    uunion(ia, ib)
        if truncated:
            break

    # fold cores -> pairs, then let the linkage decide how strict to be
    if linkage == "louvain":
        groups = _louvain_groups(U, edges, uid, n)
    else:
        by_root: Dict[int, List[int]] = {}
        for p in range(n):
            by_root.setdefault(ufind(uid[p]), []).append(p)
        groups = list(by_root.values())

    clusters: List[Dict] = []
    n_split = n_unrefined = 0
    for members in groups:
        core_ids = sorted({uid[p] for p in members})
        if linkage == "clique" and 1 < len(core_ids) <= max_cores_per_cluster:
            split = _split_into_cliques(core_ids, cores, members, uid, threshold)
            if len(split) > 1:
                n_split += 1
            clusters.extend(split)
        else:
            if linkage == "clique" and len(core_ids) > max_cores_per_cluster:
                n_unrefined += 1
            clusters.append(_build_cluster(members, cores, uid))

    clusters = [c for c in clusters if c["size"] >= min_size]
    clusters.sort(key=lambda c: (-c["size"], c["members"][0]))
    for cid, c in enumerate(clusters):
        c["id"] = cid

    # A pair can sit in more than one strict cluster (see _split_into_cliques). For
    # the ClusterID column there is room for one, so the pair takes the first
    # cluster that claimed it -- ids are ordered largest-first, so that is the
    # biggest group it belongs to. `shared_with` names the others, and the report
    # says so rather than leaving two clusters looking unrelated.
    pair_cluster = [-1] * n
    for c in clusters:
        for i in c["members"]:
            if pair_cluster[i] == -1:
                pair_cluster[i] = c["id"]
    membership = [0] * n
    for c in clusters:
        for i in c["members"]:
            membership[i] += 1
    clustered = sum(1 for count in membership if count)
    n_overlapping = sum(1 for count in membership if count > 1)
    if n_overlapping:
        owners: Dict[int, List[int]] = {}
        for c in clusters:
            for i in c["members"]:
                owners.setdefault(i, []).append(c["id"])
        for c in clusters:
            c["shared_with"] = sorted({cid for i in c["members"]
                                       for cid in owners[i] if cid != c["id"]})
    return {
        "clusters": clusters,
        "pair_cluster": pair_cluster,
        "stats": {
            "n_pairs": n,
            "n_cores": U,
            "n_clusters": len(clusters),
            "n_clustered": clustered,
            "n_singletons": n - clustered,
            "gate_skips": gate_skips,
            "lev_compares": lev_compares,
            "truncated": truncated,
            "linkage": linkage,
            "n_groups_split": n_split,
            "n_unrefined": n_unrefined,
            "n_overlapping": n_overlapping,
            # How many pairs the caller's duplicate rule wired into the graph
            # itself, in how many charter families. `n_forced_lone` counts
            # one-pair families, which `min_size` still drops -- a single pair
            # is not a cluster, however plainly the two texts are copies.
            "n_forced_pairs": n_forced_pairs,
            "n_forced_groups": len(forced_groups),
            "n_forced_lone": n_forced_lone,
        },
    }


def cluster_documents(clusters: Sequence[Dict], pair_docs: Sequence[Tuple[int, int]],
                      sides: Sequence[int] = (0, 1)) -> List[List[int]]:
    """The documents each cluster covers, sorted, derived from its member pairs.

    A diplomatist asks "which charters carry this formula", not "which pairs do",
    so the report needs the document set. It is derived rather than clustered
    directly: two charters can share a formula through a third without their own
    pair clearing the similarity threshold, and deriving from the pairs keeps the
    cluster exactly as consistent as the evidence behind it.

    `sides` selects which end of each pair contributes. In a self-comparison both
    ends name the same corpus, so the default unions them; comparing two corpora,
    the ends are different documents and the caller maps one side at a time.
    """
    out: List[List[int]] = []
    for cluster in clusters:
        docs = set()
        for pair_index in cluster["members"]:
            for side in sides:
                docs.add(pair_docs[pair_index][side])
        out.append(sorted(docs))
    return out


# --- report -------------------------------------------------------------------

def cluster_id_map(clusters: Sequence[Dict], pair_docs: Sequence[Tuple[int, int]],
                   doc_count: int, sides: Sequence[int] = (0, 1)) -> List[str]:
    """Per-document ClusterID cell: a comma-separated list, or 'None'.

    A document can carry several formulas, so the cell is a list. A document in
    no cluster gets the literal 'None' rather than an empty cell, so a reader can
    tell "clustering ran and found nothing here" from "this column is empty" --
    the two mean very different things when the corpus is being triaged.

    `sides` works as in `cluster_documents`: (0, 1) for a self-comparison, one
    side at a time when the two corpora are distinct.
    """
    per_doc: List[List[int]] = [[] for _ in range(doc_count)]
    for cluster in clusters:
        for pair_index in cluster["members"]:
            for side in sides:
                doc = pair_docs[pair_index][side]
                if 0 <= doc < doc_count:
                    per_doc[doc].append(cluster["id"])
    return [",".join(str(c) for c in sorted(set(ids))) if ids else "None" for ids in per_doc]


# The report embeds every charter it names, so a reader who has only the HTML can
# check a formula against the text it came from. A corpus large enough to make
# that a multi-hundred-megabyte page falls back to plain names; the header says
# which of the two happened rather than leaving a broken link to discover.
MAX_EMBED_CHARS = 6000000

# What separates a shared formula from a copied-out charter. Both conditions have
# to hold, because either one alone misreads a real case:
#
# * the share alone calls a *short* charter duplicate when its whole text happens
#   to BE a formula. Measured on the MOM corpus, the 18 charters sharing the papal
#   "scripti patrocinio communimus" confirmatio carry a 160-word window that is
#   99% of each of them -- they are distinct charters, not copies, and the whole
#   point of the formula report is that they belong together;
# * the length alone calls a long shared *section* a duplicate.
#
# A formula is a part of a charter, and a diplomatic formula is short: the longest
# one measured here is 285 words (the same papal confirmatio, at its most
# stereotyped, is 160). So a window has to be longer than any plausible formula
# AND cover most of the shorter charter before the report calls it a copy rather
# than a find. Both numbers are reported on every card, and both are parameters,
# so a reader who disagrees can move the line and see what changes.
MAX_CORE_FRACTION = 0.6
MIN_DUPLICATE_TOKENS = 400


def _input_summary(input_info: Optional[Dict], stats: Dict, similarity_threshold: float,
                   cluster_threshold: float, cluster_min: int,
                   max_core_fraction: float = MAX_CORE_FRACTION,
                   min_duplicate_tokens: int = MIN_DUPLICATE_TOKENS) -> List[Tuple[str, str]]:
    """The report's input summary as (label, value) rows.

    Written from `input_info` when the caller supplies one: only flame.py knows
    how many files the corpus held, what was skipped on the way in, and whether
    the similarity threshold was chosen by the user or by the engine. The
    thresholds the report was handed are always shown, so a direct caller (a
    test, another script) still gets a truthful header rather than a blank one.
    """
    info = input_info or {}
    rows: List[Tuple[str, str]] = []

    def add(label: str, value) -> None:
        if value not in (None, "", False):
            rows.append((label, str(value)))

    add("Mode", "two corpora compared" if info.get("mode") == "two"
        else ("single corpus, compared against itself" if info else None))
    if info.get("input"):
        add("Input", info["input"] + (" (glob pattern)" if info.get("pattern") else ""))
    if info.get("input2"):
        add("Second input", info["input2"])
    if info.get("suffix"):
        add("File suffix", info["suffix"])

    charters = list(info.get("charters") or [])
    if len(charters) == 1:
        add("Charters compared", f"{charters[0]} document(s)")
    elif charters:
        add("Charters compared", f"{charters[0]} + {charters[1]} document(s), "
                                 f"{sum(charters)} in total")
    found = list(info.get("files_found") or [])
    if found:
        add("Files found", " + ".join(str(n) for n in found))
    short = sum(info.get("files_short") or [])
    dupes = sum(info.get("files_duplicate") or [])
    add("Skipped as too short", f"{short} file(s), under "
        f"{info.get('min_text_length', '?')} characters" if short else None)
    add("Skipped as duplicate text", dupes if dupes else None)
    add("Duplicate handling", "on (-deduplicate)" if info.get("deduplicate")
        else ("off, so identical files count as separate documents" if info else None))
    add("Limit", f"-keep_texts {info['keep_texts']} reached, the corpus was truncated"
        if info.get("limit_reached") else None)

    add("Similarity threshold", f"{similarity_threshold:.4f}"
        + (f" (set automatically, {info['threshold_source']})" if info.get("threshold_source") else ""))
    add("Core similarity threshold", f"{cluster_threshold:.4f}")
    linkage = info.get("linkage") or "louvain"
    add("Linkage", {
        "clique": "strict -- maximal cliques, every core in a cluster is within the "
                  "threshold of every other",
        "union": "union-find -- KONI's original rule, which can chain unrelated cores together",
        "louvain": "Louvain communities -- cores denser among themselves than with the rest, "
                   "which keeps one template in one piece across its variable slots",
    }.get(linkage, linkage))
    add("Core alignment", info.get("alignment"))
    add("Anchoring", info.get("anchors"))
    counts = info.get("anchor_counts")
    if counts:
        anchored, total = counts
        add("Anchored pairs", f"{anchored} of {total} "
                              f"({total - anchored} kept the unanchored window)")
    # Taken from the parameter, not from `info`: like the two thresholds above,
    # this is a number the report was handed, and a direct caller that supplies no
    # `input_info` still gets a truthful header.
    add("Near-duplicate threshold", f"{min_duplicate_tokens}+ shared word(s) covering "
                                    f"{max_core_fraction:.0%} of the shorter charter")
    add("Minimum cluster size", f"{cluster_min} pair(s)")
    add("Pair admission", "FLAME's TF-IDF cosine over the whole charters (the similarity "
                          "threshold above); the clustering itself runs on the cores")
    add("Generated", info.get("generated"))
    add("Command", info.get("command"))
    return rows


def duplicate_groups(pair_docs: Sequence[Tuple[int, int]], overlaps: Sequence[int],
                     lengths1: Sequence[int], lengths2: Sequence[int],
                     max_core_fraction: float = MAX_CORE_FRACTION,
                     min_duplicate_tokens: int = MIN_DUPLICATE_TOKENS) -> List[List[int]]:
    """Pair indices that pass the near-duplicate rule, grouped into charter families.

    The very two conditions `is_near_duplicate` applies to a cluster, applied
    instead to a single pair: it shares at least `min_duplicate_tokens` words and
    those cover at least `max_core_fraction` of the shorter text. Both are read
    from the pair's *whole* shared span (`overlaps`, the aligner's `overlap`),
    never from the reported window -- the anchor ceiling narrows that to the legal
    act, and a copy shares far more than an act.

    Grouped transitively through shared documents, because a duplicate is a
    relation between charters rather than between pairs: `(a, b)` and `(a, c)`
    name one family of copies. The groups are what `cluster_pairs` takes as
    `forced_groups` -- a pair that qualifies here is wired into the graph by that
    alone, so a core narrowed to the act cannot lose it (see the note in
    `cluster_pairs`).

    A family of one pair is returned too, and reported: it is a genuine duplicate
    find that `min_size` cannot turn into a cluster, and the caller should say so
    rather than let it vanish.
    """
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    qualifying: List[int] = []
    for p, (i, j) in enumerate(pair_docs):
        if p >= len(overlaps):
            break
        shared = overlaps[p]
        shorter = min(lengths1[i], lengths2[j])
        if not shorter:
            continue
        if shared >= min_duplicate_tokens and shared / shorter >= max_core_fraction:
            qualifying.append(p)
            union(i, j)
    families: Dict[int, List[int]] = {}
    for p in qualifying:
        families.setdefault(find(pair_docs[p][0]), []).append(p)
    return [families[root] for root in sorted(families)]


def is_near_duplicate(record: Dict, max_core_fraction: float = MAX_CORE_FRACTION,
                      min_duplicate_tokens: int = MIN_DUPLICATE_TOKENS) -> bool:
    """Whether a cluster is one charter written out again rather than a formula.

    See `MAX_CORE_FRACTION` for why both conditions are required. `shared_tokens`
    is the window the cluster's pairs actually share (the narrower side of each
    pair, the widest of those); older records that predate the measure fall back
    to `core_tokens`, which is the reference core's length.
    """
    tokens = record.get("shared_tokens") or record.get("core_tokens") or 0
    return (record.get("coverage", 0.0) >= max_core_fraction
            and tokens >= min_duplicate_tokens)


def generate_cluster_report(records: Sequence[Dict], stats: Dict,
                            similarity_threshold: float, cluster_threshold: float,
                            cluster_min: int, input_info: Optional[Dict] = None,
                            max_core_fraction: float = MAX_CORE_FRACTION,
                            min_duplicate_tokens: int = MIN_DUPLICATE_TOKENS) -> None:
    """Writes clusters.tsv and clusters.html from prepared records.

    Each record is built by the caller, which is the only place that knows how a
    document index maps to a name (and, when two corpora are compared, which side
    it came from)::

        {"id", "size", "core_display", "core_folded",
         "documents": [name, ...], "distinct_texts": int,
         "members": [(name1, name2, cosine), ...],
         "core_ratio": [float, ...], "cohesion": float,
         "core_tokens": int, "mean_idf": float, "specificity": float,
         "doc_texts": {name: {"tokens": [...], "spans": [(start, end), ...]}}}

    A record carries both spellings of the core on purpose. `core_display` is the
    formula as the charters write it, and `core_folded` is what the clustering
    actually compared; when two charters land in one cluster despite visibly
    different spelling, the folded form is the evidence for why.

    `distinct_texts` counts how many *different* texts the cluster's documents
    hold. It is not redundant with `len(documents)`: a corpus that keeps one
    charter under three names (a shelfmark, an edition and a re-download) turns a
    single comparison into nine pairs, and a bare "6 documents" would read as six
    independent witnesses. Where the two numbers differ, the pair count is
    inflated by repeated copies of the same text.

    `doc_texts` is optional and only supplies the embedded reading view; a record
    without it still produces a complete report, with the document names as plain
    text instead of links.
    """
    # The per-pair cosine is the TF-IDF similarity that let the pair into the
    # comparison at all. It is NOT the number that formed the cluster: that one is
    # `cluster_threshold` applied to the cores, reported beside it as PairCoreRatio,
    # and the two can differ a lot (a pair can reach a cluster with a modest cosine
    # when its shared formula is nearly verbatim).
    # The two kinds are separated once, here, so the TSV and the HTML can never
    # disagree about which clusters are formulas.
    formula_records = [r for r in records
                       if not is_near_duplicate(r, max_core_fraction, min_duplicate_tokens)]
    duplicate_records = [r for r in records
                         if is_near_duplicate(r, max_core_fraction, min_duplicate_tokens)]

    tsv_rows = ["ClusterID\tSize\tAnchor\tCoreFormula\tCoreFolded\tDocuments\tDistinctTexts\t"
                "CoreTokens\tMeanIDF\tSpecificity\tCohesion\tMembers\tPairCosine\tPairCoreRatio\t"
                "SharedTokens\tCoverage\tKind\n"]
    for r in records:
        members = "; ".join(f"{n1} ~ {n2} (cosine {cosine:.4f})" for n1, n2, cosine in r["members"])
        kind = ("near-duplicate" if is_near_duplicate(r, max_core_fraction, min_duplicate_tokens)
                else "formula")
        tsv_rows.append(
            f"{r['id']}\t{r['size']}\t{r.get('anchor', '')}\t{r['core_display']}\t{r['core_folded']}\t"
            f"{', '.join(r['documents'])}\t{r['distinct_texts']}\t{r['core_tokens']}\t"
            f"{r['mean_idf']:.4f}\t{r['specificity']:.1f}\t{r['cohesion']:.4f}\t{members}\t"
            f"{'; '.join(f'{cosine:.4f}' for _n1, _n2, cosine in r['members'])}\t"
            f"{'; '.join(f'{ratio:.4f}' for ratio in r['core_ratio'])}\t"
            f"{r.get('shared_tokens') or r.get('core_tokens') or 0}\t"
            f"{r.get('coverage', 0.0):.3f}\t{kind}\n")
    with open("clusters.tsv", "w", encoding="utf-8") as f:
        f.writelines(tsv_rows)
    print("Generated clusters.tsv")

    # A charter is rendered once per cluster it appears in, not once overall: the
    # highlight follows *this* cluster's core, and the same charter carries
    # different formulas in different clusters, so one shared copy would mark the
    # wrong words.
    rendered: Dict[Tuple[int, str], str] = {}
    for r in records:
        for name in r["documents"]:
            entry = (r.get("doc_texts") or {}).get(name)
            if entry:
                rendered[(r["id"], name)] = render_tokens(entry["tokens"], entry["spans"])
    embedded = sum(len(text) for text in rendered.values()) <= MAX_EMBED_CHARS

    def card(r: Dict) -> str:
        member_rows = "\n".join(
            f"      <li>{html.escape(n1)} &harr; {html.escape(n2)} "
            f"<span class='score'>cosine {cosine:.4f}</span> "
            f"<span class='ratio'>core {ratio:.4f}</span></li>"
            for (n1, n2, cosine), ratio in zip(r["members"], r["core_ratio"]))
        folded_note = ("" if r["core_folded"] == r["core_display"].lower() else
                       f"      <p class='folded'>compared as: <code>{html.escape(r['core_folded'])}</code></p>")
        # Only shown when the two counts disagree, so the common case stays clean.
        # A single-text cluster is called out plainly: it is one charter filed under
        # several names, and reads as a discovery until you know that.
        distinct = r["distinct_texts"]
        repeated_note = ""
        if distinct < len(r["documents"]):
            if distinct == 1:
                repeated_note = (f" <span class='repeated'>-- all one text, filed under "
                                 f"{len(r['documents'])} names</span>")
            else:
                repeated_note = f" <span class='repeated'>({distinct} distinct text(s))</span>"
        shared = r.get("shared_with") or []
        if shared:
            repeated_note += (f" <span class='shared'>shares pair(s) with cluster "
                              f"{', '.join(str(cid) for cid in shared)}: this formula sits "
                              f"between two groups</span>")
        # The verb the window was seeded on, said in words: it is the one-word
        # answer to "which legal act is this cluster?", and its absence is worth
        # stating too -- a cluster with no anchor is a protocol-level match, and
        # the reader should not have to infer that from an empty cell.
        anchor_note = (f"<strong>Legal act:</strong> <code>{html.escape(r['anchor'])}</code> &middot; "
                       if r.get("anchor") else
                       "<strong>Legal act:</strong> no performative verb in the shared text &middot; ")

        names = []
        bodies = []
        for index, name in enumerate(r["documents"]):
            escaped = html.escape(name)
            key = (r["id"], name)
            if embedded and key in rendered:
                anchor = f"doc-{r['id']}-{index}"
                names.append(f"<a href='#{anchor}'>{escaped}</a>")
                bodies.append(f"""      <details class="doc" id="{anchor}">
        <summary>{escaped}</summary>
        <p class="doctext">{rendered[key]}</p>
      </details>""")
            else:
                names.append(escaped)
        return f"""  <section class="cluster">
    <h2>Cluster {r['id']} <span class="size">{r['size']} pair(s), {len(r['documents'])} document(s){repeated_note}</span></h2>
    <p class="formula">{html.escape(r['core_display'])}</p>
{folded_note}
    <p class="scores">{anchor_note}<strong>Core:</strong> {r['core_tokens']} word(s) &middot;
    mean IDF {r['mean_idf']:.2f} &middot; specificity {r['specificity']:.1f} &middot;
    cohesion {r['cohesion']:.4f}</p>
    <p class="docs"><strong>Documents:</strong> {', '.join(names)}</p>
    <ul class="members">
{member_rows}
    </ul>
{chr(10).join(bodies)}
  </section>"""

    gate_note = ("The edit-distance comparison hit its cap, so this run is truncated and may be incomplete."
                 if stats.get("truncated") else
                 "The 4-gram gate rejected candidates it judged dissimilar before the edit-distance step.")
    if stats.get("n_groups_split"):
        gate_note += (f" {stats['n_groups_split']} group(s) of cores were chained under union-find and "
                      f"split into {stats['n_clusters']} strict cluster(s).")
    if stats.get("n_unrefined"):
        gate_note += (f" {stats['n_unrefined']} group(s) were too large to refine and were left as "
                      f"connected components.")
    if stats.get("n_overlapping"):
        gate_note += (f" {stats['n_overlapping']} pair(s) sit in more than one cluster: their "
                      f"formula is close to two groups that are not close to each other.")
    link_note = ("Each document name links to its full text below, with the shared formula marked."
                 if embedded and rendered else
                 "The charters are named but not embedded in this report.")

    split_note = (f"{len(formula_records)} of the {len(records)} cluster(s) share a formula; "
                  f"{len(duplicate_records)} share a window of at least {min_duplicate_tokens} word(s) "
                  f"covering {max_core_fraction:.0%} of the shorter charter and are listed as "
                  f"near-duplicate charters below."
                  if duplicate_records else
                  f"All {len(records)} cluster(s) share a formula: none shares a window of "
                  f"{min_duplicate_tokens}+ word(s) covering {max_core_fraction:.0%} of the shorter "
                  f"charter.")
    formula_cards = [card(r) for r in formula_records]
    duplicate_cards = [card(r) for r in duplicate_records]
    # Only shown when the split found something. A section that is always there
    # and always empty just teaches the reader to skip it.
    duplicate_section = (f"""
<h1>Near-duplicate charters</h1>
<p class="meta">Clusters whose shared window runs to at least {min_duplicate_tokens} word(s) and
covers at least {max_core_fraction:.0%} of the shorter charter. A formula is a <em>part</em> of a
charter -- an arenga, a clause, a corroboratio -- and a diplomatic formula is short (the longest in
this corpus is 285 words), so a window this long, covering this much of the text, is not a formula
two charters have in common but one charter written out again. Both numbers are printed on every
card below. These clusters are listed separately so they neither pad nor distort the formula
clusters above; <code>clusters.tsv</code> keeps the same cluster ids and marks each row in its
<code>Kind</code> column.</p>
{chr(10).join(duplicate_cards)}""" if duplicate_cards else "")
    summary = "\n".join(f"  <dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd>"
                        for label, value in _input_summary(
                            input_info, stats, similarity_threshold, cluster_threshold, cluster_min,
                            max_core_fraction, min_duplicate_tokens))
    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Cluster report (recurring formulas)</title>
<style>
 body {{ font-family: Georgia, serif; margin: 2rem auto; max-width: 60rem; line-height: 1.5; padding: 0 1rem; }}
 h1 {{ margin-bottom: .2rem; }}
 .meta {{ color: #555; margin-top: 0; }}
 .inputs {{ display: grid; grid-template-columns: max-content 1fr; gap: .15rem 1rem;
            margin: 1rem 0 2rem; font-size: .85rem; }}
 .inputs dt {{ color: #555; white-space: nowrap; }}
 .inputs dd {{ margin: 0; }}
 .cluster {{ border: 1px solid #ddd; border-left: 4px solid #4a6fa5; padding: 1rem 1.25rem; margin: 1.5rem 0; }}
 .cluster h2 {{ margin: 0 0 .5rem; font-size: 1.05rem; }}
 .size {{ font-weight: normal; color: #666; font-size: .85rem; }}
 .formula {{ font-style: italic; background: #fffbe6; padding: .6rem .8rem; margin: .4rem 0; }}
 .folded, .docs, .scores {{ font-size: .85rem; color: #555; }}
 .members {{ font-size: .85rem; }}
 .score {{ color: #4a6fa5; font-variant-numeric: tabular-nums; }}
 .ratio {{ color: #2f6b3a; font-variant-numeric: tabular-nums; }}
 .repeated {{ color: #a3541f; }}
 .shared {{ color: #4a6fa5; }}
 .doc {{ margin: .4rem 0 0 1rem; font-size: .9rem; }}
 .doc summary {{ cursor: pointer; color: #4a6fa5; }}
 .doctext {{ background: #fcfcfc; border: 1px solid #eee; padding: .6rem .8rem; }}
 mark {{ background: #ffe9a8; }}
</style></head><body>
<h1>Formula clusters</h1>
<p class="meta">Charters grouped by the legal formula they share.</p>
<dl class="inputs">
{summary}
</dl>
<p class="meta">{stats.get('n_pairs', 0)} pair(s) &rarr; {stats.get('n_cores', 0)} distinct core(s)
&rarr; {stats.get('n_clusters', 0)} cluster(s) covering {stats.get('n_clustered', 0)} pair(s);
{stats.get('n_singletons', 0)} pair(s) carry a core no other pair shares.
{stats.get('lev_compares', 0)} edit-distance comparison(s), {stats.get('gate_skips', 0)} rejected by the gate.
{gate_note}</p>
<p class="meta">{split_note}</p>
<p class="meta">{link_note}</p>
{chr(10).join(formula_cards) if formula_cards else '<p>No formula cluster reached the minimum size.</p>'}
{duplicate_section}
</body></html>"""
    with open("clusters.html", "w", encoding="utf-8") as f:
        f.write(document)
    print("Generated clusters.html")
