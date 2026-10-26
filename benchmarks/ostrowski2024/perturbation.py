#!/usr/bin/env python3
"""Task A of the article (Section 10.2): recall under controlled, synthetic noise.

    python benchmarks/ostrowski2024/perturbation.py

Every KCh file (reuse chains and distractors alike) is perturbed at rate r: each
word is hit with probability r by one operation drawn uniformly from those that
apply to it:

  substitution   one letter replaced by a random letter a-z
  insertion      one random letter inserted
  deletion       one letter removed (words of two letters or more)
  u/v, i/j       one u<->v or i<->j interchange
  doubling       a single consonant doubled, or a doubled one reduced
  variant        the word replaced by an attested orthographic variant

The attested variants are read off the gold standard itself: word pairs that the
Dk and KCh sides of a gold pair align one-to-one, and that flame.classify_gap
calls a 'variant' at flame.py's default fuzz_threshold. The Dk side is never
perturbed.

Two systems are compared at the threshold each reaches its best F1 on the clean
data (Table 2): FLAME at the operating point (vocab 35, ngram 5, n_out 2,
threshold 0.28) and the same pipeline with contiguous n-grams (n_out 0,
threshold 0.12). For each rate and seed the full pipeline is rerun, BPE training
included, and recall over the 23 gold pairs is recorded.

Writes results/perturbation.csv (one row per system, rate, seed) and
results/perturbation.png (mean recall with min-max band over the seeds).
"""
import argparse
import csv
import pathlib
import random
import shutil
import sys
import tempfile
from difflib import SequenceMatcher

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_benchmark as rb  # noqa: E402  (also puts the repository on sys.path)
import flame  # noqa: E402

RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SEEDS = [42, 43, 44, 45, 46]
SYSTEMS = [
    ('FLAME (gapped, n_out 2)', dict(vocab_size=35, ngram=5, n_out=2), 0.28),
    ('Contiguous (n_out 0)', dict(vocab_size=35, ngram=5, n_out=0), 0.12),
]
LETTERS = 'abcdefghijklmnopqrstuvwxyz'
CONSONANTS = set('bcdfghjklmnpqrstvwxz')
SWAPS = {'u': 'v', 'v': 'u', 'i': 'j', 'j': 'i'}


def attested_variants(dk, kch, gold):
    """word -> list of attested spellings, from one-to-one aligned words of the gold pairs."""
    table = {}
    for d_file, k_file in gold:
        a = flame.normalize_for_compare((dk / d_file).read_text(encoding='utf-8')).split()
        b = flame.normalize_for_compare((kch / k_file).read_text(encoding='utf-8')).split()
        for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if op != 'replace' or i2 - i1 != j2 - j1:
                continue
            for x, y in zip(a[i1:i2], b[j1:j2]):
                if x != y and flame.classify_gap([x], [y], flame.DEFAULT_PARAMS['fuzz_threshold'])[0] == 'variant':
                    table.setdefault(x, set()).add(y)
                    table.setdefault(y, set()).add(x)
    return {k: sorted(v) for k, v in table.items()}


def _letter_positions(word):
    return [i for i, c in enumerate(word) if c.lower() in LETTERS]


def perturb_word(word, rng, variants):
    pos = _letter_positions(word)
    ops = []
    if pos:
        ops += ['substitution', 'insertion']
    if len(pos) >= 2:
        ops.append('deletion')
    if any(c.lower() in SWAPS for c in word):
        ops.append('uv_ij')
    if any(word[i].lower() in CONSONANTS for i in pos):
        ops.append('doubling')
    if word.lower() in variants:
        ops.append('variant')
    if not ops:
        return word
    op = rng.choice(ops)
    if op == 'substitution':
        i = rng.choice(pos)
        return word[:i] + rng.choice(LETTERS) + word[i + 1:]
    if op == 'insertion':
        i = rng.choice(pos + [pos[-1] + 1])
        return word[:i] + rng.choice(LETTERS) + word[i:]
    if op == 'deletion':
        i = rng.choice(pos)
        return word[:i] + word[i + 1:]
    if op == 'uv_ij':
        i = rng.choice([k for k, c in enumerate(word) if c.lower() in SWAPS])
        return word[:i] + SWAPS[word[i].lower()] + word[i + 1:]
    if op == 'doubling':
        i = rng.choice([k for k in pos if word[k].lower() in CONSONANTS])
        if i + 1 < len(word) and word[i + 1] == word[i]:
            return word[:i] + word[i + 1:]
        return word[:i + 1] + word[i] + word[i + 1:]
    return rng.choice(variants[word.lower()])


def perturb_text(text, rate, rng, variants):
    return ' '.join(perturb_word(w, rng, variants) if rng.random() < rate else w for w in text.split())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--xml', help='local copy of the evaluation set (default: download from Zenodo)')
    a = ap.parse_args()

    xml = rb.ensure_dataset(a.xml)
    dk, kch, gold_path, _ = rb.build_corpus(xml)
    gold, links = rb.load_gold(gold_path)
    variants = attested_variants(dk, kch, gold)
    print(f"Attested variant table: {len(variants)} spellings")

    rb.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        noisy = pathlib.Path(tmp) / 'kch'
        for rate in RATES:
            for seed in SEEDS:
                rng = random.Random(seed)
                if noisy.exists():
                    shutil.rmtree(noisy)
                noisy.mkdir()
                for p in sorted(kch.glob('*.txt')):
                    (noisy / p.name).write_text(perturb_text(p.read_text(encoding='utf-8'), rate, rng, variants),
                                                encoding='utf-8')
                for name, cfg, thr in SYSTEMS:
                    an = rb.run_flame(dk, noisy, cfg)
                    r = rb.evaluate(rb.pair_scores(an), gold, links, thr)
                    rows.append(dict(system=name, rate=rate, seed=seed, threshold=thr, recall=round(r['R'], 4),
                                     precision=round(r['P'], 4), TP=r['TP'], FP=r['FP']))
            summary = '  '.join(f"{n}: R={np.mean([x['recall'] for x in rows if x['system'] == n and x['rate'] == rate]):.3f}"
                                for n, _, _ in SYSTEMS)
            print(f"rate {rate:.2f}  {summary}")

    out = rb.RESULTS_DIR / 'perturbation.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, rb.RESULTS_DIR / 'perturbation.png')
    print(f"Wrote {out.relative_to(rb.REPO)} and results/perturbation.png")


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ink, muted, grid = '#0b0b0b', '#52514e', '#e4e3df'
    colors = ['#2a78d6', '#eb6834']
    markers = ['o', 's']
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    fig.patch.set_facecolor('#fcfcfb')
    ax.set_facecolor('#fcfcfb')
    for (name, _, thr), color, marker in zip(SYSTEMS, colors, markers):
        mean = [np.mean([r['recall'] for r in rows if r['system'] == name and r['rate'] == x]) for x in RATES]
        lo = [min(r['recall'] for r in rows if r['system'] == name and r['rate'] == x) for x in RATES]
        hi = [max(r['recall'] for r in rows if r['system'] == name and r['rate'] == x) for x in RATES]
        ax.fill_between(RATES, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(RATES, mean, color=color, linewidth=2, marker=marker, markersize=6,
                markeredgecolor='#fcfcfb', markeredgewidth=1.5, label=f"{name}, threshold {thr:.2f}")
        ax.annotate(f"{mean[-1]:.2f}", (RATES[-1], mean[-1]), xytext=(6, 0), textcoords='offset points',
                    va='center', fontsize=9, color=ink)
    ax.set_xlabel('Perturbation rate r (share of KCh words edited)', color=muted)
    ax.set_ylabel('Recall over the 23 gold pairs', color=muted)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.01, RATES[-1] + 0.03)
    ax.grid(axis='y', color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted)
    ax.legend(frameon=False, loc='upper right', fontsize=9, labelcolor=ink)
    ax.set_title(f"Recall vs. synthetic noise (mean of {len(SEEDS)} seeds, band = min-max)",
                 color=ink, fontsize=11, loc='left')
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__':
    main()
