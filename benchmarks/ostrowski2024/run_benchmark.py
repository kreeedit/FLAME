#!/usr/bin/env python3
"""Reproduces Tables 1 and 2 of the FLAME article on the Ostrowski (2024b) gold standard.

    python benchmarks/ostrowski2024/run_benchmark.py            # Tables 1-2, ~1 min
    python benchmarks/ostrowski2024/run_benchmark.py --grid     # also the 459-config grid, ~10 min

Steps:
  1. fetch the evaluation set from Zenodo (or use --xml) and check its MD5;
  2. build the chain-level segments and distractors (build_segments.py);
  3. run flame.py's engine (Flame.load_corpus + compute_similarity_matrix) in
     inter-corpus mode, every Dk file against every KCh file;
  4. score detected Dk-KCh pairs against the gold standard at every threshold
     0.05-0.40 (step 0.01) and at the Otsu threshold flame.py selects itself.

Writes results/table1.csv, results/table2.csv, results/run.json (and
results/grid.csv with --grid). Nothing in flame.py is changed or monkey-patched;
the parameters below are passed exactly as the CLI would pass them.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import pathlib
import platform
import random
import subprocess
import sys
import tempfile
import time
import types
import urllib.request

os.environ.setdefault('TQDM_DISABLE', '1')

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import flame  # noqa: E402
from build_segments import build  # noqa: E402

SEED = 42
DATA_DIR = HERE / 'data'
RESULTS_DIR = HERE / 'results'
XML_NAME = 'Evaluation_Set_for_Historical_Text_Reuse_Detection.xml'
XML_URL = f'https://zenodo.org/api/records/13378221/files/{XML_NAME}/content'
XML_MD5 = '2381353fb446b3e85e6e1deed4e59c61'

# Settings shared by every run. The segments are short (median chain 77
# characters), so flame.py's default min_text_length of 150 would discard most
# of the gold standard; 15 keeps every chain and distractor. These are the
# values the article's grid search used.
FIXED = dict(min_text_length=15, char_norm_min_freq=2, vocab_min_word_freq=5,
             vocab_coverage=0.85, keep_texts=100000)

# Operating point of the article (Table 1, first row of Table 2).
OPERATING_POINT = dict(vocab_size=35, ngram=5, n_out=2)
ABLATIONS = [
    ('FLAME (vocab 35, ngram 5, n_out 2)', OPERATING_POINT),
    ('- gapping (n_out = 0, contiguous)', dict(vocab_size=35, ngram=5, n_out=0)),
    ('- fine tokens (vocab 200)', dict(vocab_size=200, ngram=5, n_out=2)),
    ('- subword (vocab_size auto)', dict(vocab_size='auto', ngram=5, n_out=2)),
]
CLI_DEFAULTS = dict(vocab_size=flame.DEFAULT_PARAMS['vocab_size'],
                    ngram=flame.DEFAULT_PARAMS['ngram'],
                    n_out=flame.DEFAULT_PARAMS['n_out'])

THRESHOLDS = [round(t, 3) for t in np.arange(0.05, 0.401, 0.01)]

# Search space of the article's grid search.
GRID_VOCAB = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 45, 60, 80, 120, 200, 'auto']
GRID_NGRAM = [2, 3, 4, 5, 6, 8, 10]
GRID_NOUT = [0, 1, 2, 4, 5]

# Regest numbers of the four non-literal gold pairs, from Ostrowski (2024c),
# Dokumentation von Parallelstellen (https://doi.org/10.5281/zenodo.14410766).
# The TEI file carries no regest numbers. They were assigned by matching each
# segment's text against the passages each regest quotes (fuzzy partial ratio
# 93-100 for all 23 pairs; the literal pairs land on Regest 13 for r141 and
# Regest 5 for a5r4/a5r5, as the article states).
REGEST_OF_NON_LITERAL = {
    ('Dk_seg_F5r2-1.txt', 'KCh_seg_a4v1-1.txt'): 4,
    ('Dk_seg_F6v1-1.txt', 'KCh_seg_b1r6-1.txt'): 21,
    ('Dk_seg_F7r-1.txt', 'KCh_seg_b1r6-1.txt'): 21,
    ('Dk_seg_F6v3-1.txt', 'KCh_seg_b1r7-1.txt'): 22,
}

# Figures reported by Ostrowski (2024a) and quoted in the article's Table 1.
# They are not recomputed here: the three tools are not part of this repository.
QUOTED_BASELINES = [
    ('Passim (Ostrowski 2024)', 1.00, 0.26, 0.42, 5),
    ('BLAST (Ostrowski 2024)', 0.38, 0.26, 0.31, 5),
    ('TextPAIR (Ostrowski 2024)', 0.75, 0.16, 0.26, 3),
]


# ----------------------------------------------------------------------------- data

def md5sum(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 16), b''):
            h.update(block)
    return h.hexdigest()


def ensure_dataset(xml=None):
    """Returns the path of a verified copy of the Ostrowski TEI file."""
    path = pathlib.Path(xml) if xml else DATA_DIR / XML_NAME
    if not path.exists():
        if xml:
            sys.exit(f"{path} does not exist")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {XML_URL}")
        urllib.request.urlretrieve(XML_URL, path)
    digest = md5sum(path)
    if digest != XML_MD5:
        sys.exit(f"{path}: MD5 {digest} does not match the Zenodo record ({XML_MD5})")
    return path


def build_corpus(xml_path, out_dir=DATA_DIR / 'segments'):
    out_dir = pathlib.Path(out_dir)
    dk, kch, gold_path = out_dir / 'dk', out_dir / 'kch', out_dir / 'gold_standard.json'
    for d in (dk, kch):
        if d.exists():
            for p in d.glob('*.txt'):
                p.unlink()
    stats = build(str(xml_path), str(dk), str(kch), str(gold_path), verbose=False)
    return dk, kch, gold_path, stats


def load_gold(path):
    """(dk_file, kch_file) -> subtype, and kch_file -> set of its dk_files (one gold link)."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    pairs, links = {}, {}
    for link in data:
        kch = f"KCh_seg_{link['source_chronicle_segment_id']}.txt"
        links[kch] = {f"Dk_seg_{d}.txt" for d in link['target_doernenkrantz_ids']}
        for d in link['target_doernenkrantz_ids']:
            pairs[(f"Dk_seg_{d}.txt", kch)] = link.get('subtype', 'unspecified')
    return pairs, links


# ----------------------------------------------------------------------------- engine

def make_args(dk, kch, overrides):
    params = dict(flame.DEFAULT_PARAMS)
    params.update(FIXED)
    params.update(input_path=str(dk), input_path2=str(kch))
    params.update(overrides)
    return types.SimpleNamespace(**params)


def run_flame(dk, kch, overrides):
    """Runs the scoring core of flame.py once and returns the analyzer (or None)."""
    args = make_args(dk, kch, overrides)
    with tempfile.TemporaryDirectory() as tmp:
        analyzer = flame.Flame(args=args, tmp_dir=tmp)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            analyzer.load_corpus()
            if not analyzer.corpus:
                return None
            analyzer.compute_similarity_matrix()
    return analyzer


def otsu_threshold(analyzer):
    with contextlib.redirect_stdout(io.StringIO()):
        return analyzer._determine_auto_threshold(method='otsu')


def pair_scores(analyzer):
    """Every non-zero (dk_file, kch_file) -> cosine score of an inter-corpus run."""
    coo = analyzer.dist_mat.tocoo()
    dk = [p.name for p in analyzer.file_paths]
    kch = [p.name for p in analyzer.file_paths2]
    return {(dk[r], kch[c]): float(s) for r, c, s in zip(coo.row, coo.col, coo.data)}


def bpe_vocab_size(analyzer):
    return analyzer.tokenizer_model.get_vocab_size() if analyzer.tokenizer_model else None


# ----------------------------------------------------------------------------- scoring

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(scores, gold, links, thr):
    """Pair-level metrics at one threshold, on three gold denominators.

    full   : all 23 gold pairs (what the article's Table 1 caption describes).
    lw     : the 19 literal + weak-literal pairs; detections of the 4
             non-literal pairs are dropped from scoring rather than counted
             as false positives.
    links  : the 19 gold links (one KCh chain and its Dk chain(s)); a link is
             recovered if any of its pairs is detected.
    """
    detected = {p for p, s in scores.items() if s >= thr
                and p[0].startswith('Dk') and p[1].startswith('KCh')}
    true = set(gold)
    tp, fp = detected & true, detected - true
    out = dict(threshold=thr, detected=len(detected), TP=len(tp), FP=len(fp), FN=len(true - tp))
    out['P'], out['R'], out['F1'] = _prf(len(tp), len(fp), len(true - tp))

    lw = {p for p, s in gold.items() if s in ('literal', 'weak-literal')}
    non = true - lw
    tp_lw = detected & lw
    fp_lw = detected - true
    out['TP_lw'] = len(tp_lw)
    out['P_lw'], out['R_lw'], out['F1_lw'] = _prf(len(tp_lw), len(fp_lw), len(lw - tp_lw))
    out['non_literal_detected'] = len(detected & non)

    hit_links = sum(1 for k, dks in links.items() if any((d, k) in detected for d in dks))
    out['links_recovered'] = hit_links
    out['R_links'] = hit_links / len(links)

    for sub in ('literal', 'weak-literal', 'non-literal'):
        tot = [p for p, s in gold.items() if s == sub]
        out[f'hit_{sub}'] = f"{sum(1 for p in tot if p in detected)}/{len(tot)}"
    out['FP_background'] = sum(1 for d, k in fp if '_bg_' in d or '_bg_' in k)
    out['false_positives'] = sorted(f"{d} <-> {k}" for d, k in fp)
    return out


def best_on_grid(scores, gold, links):
    """Best-F1 threshold on THRESHOLDS; ties go to higher precision, then the lower threshold."""
    curve = [evaluate(scores, gold, links, t) for t in THRESHOLDS]
    return max(curve, key=lambda r: (r['F1'], r['P'])), curve


# ----------------------------------------------------------------------------- provenance

def provenance(seconds, dataset_md5, corpus_stats):
    def git(*a):
        try:
            return subprocess.run(['git', *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    versions = {}
    for mod in ('numpy', 'scipy', 'sklearn', 'skimage', 'tokenizers', 'rapidfuzz', 'fargv', 'nltk', 'tqdm'):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    # Dirty means the code that produced the numbers differs from the commit; the
    # results/ files this run overwrites do not count.
    status = git('status', '--porcelain', '--untracked-files=no', '--', 'flame.py', 'flame_clustering.py',
                 *[str(p.relative_to(REPO)) for p in HERE.glob('*.py')])
    return dict(git_commit=git('rev-parse', 'HEAD'),
                git_dirty=bool(status),
                python=platform.python_version(), platform=platform.platform(),
                packages=versions, seed=SEED, wall_clock_seconds=round(seconds, 1),
                dataset=dict(doi='10.5281/zenodo.13378221', file=XML_NAME, md5=dataset_md5),
                corpus=corpus_stats, fixed_parameters=FIXED, thresholds=[THRESHOLDS[0], THRESHOLDS[-1], 0.01])


def r2(x):
    return f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--xml', help='local copy of the evaluation set (default: download from Zenodo)')
    ap.add_argument('--grid', action='store_true', help='also rerun the full grid search (~10 min)')
    a = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    xml = ensure_dataset(a.xml)
    dk, kch, gold_path, stats = build_corpus(xml)
    gold, links = load_gold(gold_path)
    print(f"Corpus: {stats}")

    runs = {}

    def run(label, cfg):
        an = run_flame(dk, kch, cfg)
        scores = pair_scores(an)
        best, curve = best_on_grid(scores, gold, links)
        otsu = otsu_threshold(an)
        runs[label] = dict(config={k: v for k, v in cfg.items()}, bpe_vocab_size=bpe_vocab_size(an),
                           best=best, otsu=evaluate(scores, gold, links, otsu),
                           at_0_28=evaluate(scores, gold, links, 0.28), scores=scores)
        b = best
        print(f"{label:40s} vocab={runs[label]['bpe_vocab_size']:>4}  best@{b['threshold']:.2f} "
              f"P={b['P']:.3f} R={b['R']:.3f} F1={b['F1']:.3f} TP={b['TP']} FP={b['FP']}  "
              f"| Otsu@{otsu:.4f} P={runs[label]['otsu']['P']:.3f} R={runs[label]['otsu']['R']:.3f} "
              f"F1={runs[label]['otsu']['F1']:.3f}")
        return runs[label]

    for label, cfg in ABLATIONS:
        run(label, cfg)
    run('FLAME CLI defaults (vocab auto, ngram 6, n_out 1)', CLI_DEFAULTS)

    op = runs[ABLATIONS[0][0]]
    defaults = runs['FLAME CLI defaults (vocab auto, ngram 6, n_out 1)']

    # ---- Table 1: both gold sets. literal_weak_19 is the subset Ostrowski's
    # baseline figures cover; all_23 adds the four non-literal pairs.
    t1 = RESULTS_DIR / 'table1.csv'
    with open(t1, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['system', 'gold_set', 'threshold', 'P', 'R', 'F1', 'pairs_found', 'gold_pairs',
                    'detected', 'FP', 'source'])
        for name, p, r, f1, n in QUOTED_BASELINES:
            w.writerow([name, 'literal_weak_19', '', r2(p), r2(r), r2(f1), n, 19, '', '', 'quoted'])

        def flame_rows(name, res, thr_fmt):
            thr = thr_fmt.format(res['threshold'])
            w.writerow([name, 'literal_weak_19', thr, r2(res['P_lw']), r2(res['R_lw']), r2(res['F1_lw']),
                        res['TP_lw'], 19, res['detected'], res['FP'], 'this run'])
            w.writerow([name, 'all_23', thr, r2(res['P']), r2(res['R']), r2(res['F1']),
                        res['TP'], 23, res['detected'], res['FP'], 'this run'])

        flame_rows('FLAME (vocab 35, ngram 5, n_out 2), best-F1 threshold', op['best'], '{:.2f}')
        flame_rows('FLAME (vocab 35, ngram 5, n_out 2), Otsu threshold', op['otsu'], '{:.4f}')
        flame_rows('FLAME CLI defaults (vocab auto, ngram 6, n_out 1), Otsu threshold', defaults['otsu'], '{:.4f}')

    # ---- Table 2: ablation on all 23 pairs, each row at its own best-F1 threshold
    t2 = RESULTS_DIR / 'table2.csv'
    with open(t2, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['configuration', 'gold_set', 'vocab_size', 'bpe_vocab_size', 'ngram', 'n_out',
                    'best_threshold', 'P', 'R', 'F1', 'TP', 'FP', 'literal', 'weak_literal', 'non_literal'])
        for label, cfg in ABLATIONS:
            rr = runs[label]
            b = rr['best']
            w.writerow([label, 'all_23', cfg['vocab_size'], rr['bpe_vocab_size'], cfg['ngram'], cfg['n_out'],
                        r2(b['threshold']), r2(b['P']), r2(b['R']), r2(b['F1']), b['TP'], b['FP'],
                        b['hit_literal'], b['hit_weak-literal'], b['hit_non-literal']])

    # ---- the non-literal pairs and whether any configuration recovers them
    non_literal = [dict(dk=d, kch=k, regest=REGEST_OF_NON_LITERAL[(d, k)],
                        cosine_at_operating_point=round(op['scores'].get((d, k), 0.0), 4))
                   for (d, k), sub in sorted(gold.items()) if sub == 'non-literal']

    # ---- qualitative examples (article Section 10.5)
    examples = {}
    for d_file, k_file in [('Dk_seg_G4v-1.txt', 'KCh_seg_r141.txt'), ('Dk_seg_G5r.txt', 'KCh_seg_r141.txt'),
                           ('Dk_seg_G1r.txt', 'KCh_seg_a5r4.txt'), ('Dk_seg_G1r.txt', 'KCh_seg_a5r5.txt')]:
        examples[f"{d_file} <-> {k_file}"] = round(op['scores'].get((d_file, k_file), 0.0), 4)

    # ---- optional grid
    grid_summary = None
    if a.grid:
        grid_summary = run_grid(dk, kch, gold, links)

    for rr in runs.values():
        del rr['scores']
    report = provenance(time.time() - t0, md5sum(xml), stats)
    report.update(runs=runs, qualitative_example_cosines=examples, non_literal_pairs=non_literal, grid=grid_summary)
    with open(RESULTS_DIR / 'run.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nWrote {t1.relative_to(REPO)}, {t2.relative_to(REPO)}, "
          f"{(RESULTS_DIR / 'run.json').relative_to(REPO)}  ({time.time() - t0:.0f}s)")


def run_grid(dk, kch, gold, links):
    configs = [(v, n, o) for v in GRID_VOCAB for n in GRID_NGRAM for o in GRID_NOUT if o < n]
    rows = []
    path = RESULTS_DIR / 'grid.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['vocab_size', 'ngram', 'n_out', 'threshold', 'detected', 'TP', 'FP', 'FN', 'P', 'R', 'F1'])
        for i, (v, n, o) in enumerate(configs, 1):
            an = run_flame(dk, kch, dict(vocab_size=v, ngram=n, n_out=o))
            if an is None or an.dist_mat is None or an.dist_mat.nnz == 0:
                continue
            scores = pair_scores(an)
            for t in THRESHOLDS:
                r = evaluate(scores, gold, links, t)
                w.writerow([v, n, o, t, r['detected'], r['TP'], r['FP'], r['FN'],
                            f"{r['P']:.4f}", f"{r['R']:.4f}", f"{r['F1']:.4f}"])
                rows.append((v, n, o, r))
            print(f"  grid {i}/{len(configs)}", end='\r')
    # First maximum in search order, ties broken by precision (as in the article's search).
    v, n, o, r = max(rows, key=lambda x: (x[3]['F1'], x[3]['P']))
    print(f"\nGrid optimum: vocab={v} ngram={n} n_out={o} thr={r['threshold']:.2f} "
          f"P={r['P']:.3f} R={r['R']:.3f} F1={r['F1']:.3f}")
    return dict(configs=len(configs), optimum=dict(vocab_size=v, ngram=n, n_out=o, threshold=r['threshold'],
                                                   P=r['P'], R=r['R'], F1=r['F1'], TP=r['TP'], FP=r['FP']))


if __name__ == '__main__':
    main()
