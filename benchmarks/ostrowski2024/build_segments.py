"""Builds the chain-level benchmark corpus from the Ostrowski (2024b) TEI file.

The annotated reuse in the evaluation set is split into `<seg type="tr">`
sub-segments linked by `prev`/`next`. A single sub-segment is a 7-49 character
fragment; the unit of reuse is the whole chain. This script

  1. rebuilds every prev/next chain and writes one file per chain, named after
     the chain head (`Dk_seg_DkF4v2-1.txt`);
  2. writes the gold standard keyed by the same head ids, so corpus and gold
     standard share one granularity and one naming scheme;
  3. writes background distractors (`Dk_bg_0001.txt`, ...): runs of `<l>` lines
     that carry no reuse annotation, merged into chunks of about
     `bg_target_len` characters, dropping chunks shorter than `bg_min_len`.

With the defaults this yields 22 Dk and 19 KCh chains, 113 Dk and 379 KCh
distractors, and 19 gold links covering 23 Dk-KCh segment pairs (article
Section 10.1). Standard library only.
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET

NS = {'tei': 'http://www.tei-c.org/ns/1.0'}
XMLID = '{http://www.w3.org/XML/1998/namespace}id'


def _clean_ref(x):
    return (x or '').lstrip('#').strip()


def _text_of(elem):
    return ' '.join(''.join(elem.itertext()).split())


def build(xml_path, out_dk, out_kch, gold_path,
          with_background=True, bg_target_len=120, bg_min_len=20, verbose=True):
    for d in (out_dk, out_kch):
        os.makedirs(d, exist_ok=True)

    root = ET.parse(xml_path).getroot()

    # Index every <seg>: id -> element, and id -> owning text ('Dk' or 'KCh').
    seg, tei_of = {}, {}
    for tei in root.findall('.//tei:TEI', NS):
        tid = tei.get(XMLID)
        for s in tei.findall('.//tei:seg', NS):
            sid = s.get(XMLID)
            if sid:
                seg[sid] = s
                tei_of[sid] = tid

    def head_of(sid):
        cur, seen = sid, set()
        while True:
            p = _clean_ref(seg[cur].get('prev'))
            if p and p in seg and p not in seen:
                seen.add(p)
                cur = p
            else:
                return cur

    def chain_ids(head):
        ids, cur, seen = [head], head, {head}
        while True:
            n = _clean_ref(seg[cur].get('next'))
            if n and n in seg and n not in seen:
                ids.append(n)
                seen.add(n)
                cur = n
            else:
                return ids

    heads = {}
    for sid, s in seg.items():
        if (s.get('type') or '').lower() == 'tr':
            h = head_of(sid)
            if h not in heads:
                heads[h] = chain_ids(h)

    written = {'Dk': 0, 'KCh': 0}
    for h, ids in heads.items():
        tid = tei_of[h]
        full = ' '.join(_text_of(seg[i]) for i in ids).strip()
        if not full:
            continue
        target = out_dk if tid == 'Dk' else out_kch
        with open(os.path.join(target, f"{tid}_seg_{h}.txt"), 'w', encoding='utf-8') as f:
            f.write(full)
        written[tid] += 1

    # The corresp attribute lives on the KCh side and points at Dk segments.
    gold, n_pairs = [], 0
    for sid, s in seg.items():
        corr = s.get('corresp')
        if not corr or tei_of[sid] != 'KCh':
            continue
        dk_heads = sorted({head_of(_clean_ref(t)) for t in corr.split() if _clean_ref(t) in seg})
        if dk_heads:
            gold.append({
                "source_chronicle_segment_id": head_of(sid),
                "target_doernenkrantz_ids": dk_heads,
                "subtype": s.get('subtype') or "unspecified",
            })
            n_pairs += len(dk_heads)
    with open(gold_path, 'w', encoding='utf-8') as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)

    bg_counts = {'Dk': 0, 'KCh': 0}
    if with_background:
        for tei in root.findall('.//tei:TEI', NS):
            tid = tei.get(XMLID)
            target = out_dk if tid == 'Dk' else out_kch
            counter, buf, buflen = 1, [], 0

            def flush():
                nonlocal counter, buf, buflen
                if buflen >= bg_min_len:
                    with open(os.path.join(target, f"{tid}_bg_{counter:04d}.txt"), 'w', encoding='utf-8') as f:
                        f.write(' '.join(buf).strip())
                    bg_counts[tid] += 1
                    counter += 1
                buf, buflen = [], 0

            for line in tei.findall('.//tei:l', NS):
                if any((sg.get('type') or '').lower() == 'tr' for sg in line.findall('.//tei:seg', NS)):
                    flush()  # an annotated line ends the background run
                    continue
                txt = _text_of(line)
                if txt:
                    buf.append(txt)
                    buflen += len(txt)
                    if buflen >= bg_target_len:
                        flush()
            flush()

    missing = [p for link in gold
               for p in [os.path.join(out_kch, f"KCh_seg_{link['source_chronicle_segment_id']}.txt")]
               + [os.path.join(out_dk, f"Dk_seg_{d}.txt") for d in link['target_doernenkrantz_ids']]
               if not os.path.exists(p)]
    if missing:
        raise RuntimeError(f"{len(missing)} gold-standard file(s) missing from the corpus, e.g. {missing[0]}")

    stats = dict(chains_dk=written['Dk'], chains_kch=written['KCh'],
                 background_dk=bg_counts['Dk'], background_kch=bg_counts['KCh'],
                 gold_links=len(gold), gold_pairs=n_pairs)
    if verbose:
        print(f"Chains: Dk={written['Dk']} KCh={written['KCh']}  "
              f"Background: Dk={bg_counts['Dk']} KCh={bg_counts['KCh']}  "
              f"Gold: {len(gold)} links / {n_pairs} pairs")
    return stats


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('xml')
    ap.add_argument('--out', default='segments', help='output directory (gets dk/, kch/, gold_standard.json)')
    ap.add_argument('--no-background', action='store_true')
    ap.add_argument('--bg-target-len', type=int, default=120)
    a = ap.parse_args()
    build(a.xml, os.path.join(a.out, 'dk'), os.path.join(a.out, 'kch'),
          os.path.join(a.out, 'gold_standard.json'),
          with_background=not a.no_background, bg_target_len=a.bg_target_len)
