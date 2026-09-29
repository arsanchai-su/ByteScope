"""Train/test leakage audit with perceptual hashing.

Writes dedup/near_duplicates.json (list of TEST paths having a near-duplicate
in TRAIN with Hamming distance <= --thr) plus summary stats.  Pass the list to
train.py --exclude to report results on the de-duplicated test set.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from bytescope.data import _find_split_dir, scan_split


def phash_all(items):
    return np.array([int(str(imagehash.phash(Image.open(p).convert("L"), hash_size=8)), 16) for p, _ in items],
                    dtype=np.uint64)


def popcount64(x):
    x = x.astype(np.uint64)
    c = np.zeros(x.shape, dtype=np.uint64)
    for _ in range(64):
        c += x & np.uint64(1)
        x >>= np.uint64(1)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--thr", type=int, default=4)
    ap.add_argument("--out", default="dedup")
    args = ap.parse_args()
    root = Path(args.data)
    tr = scan_split(_find_split_dir(root, ["train", "Train"]))
    te = scan_split(_find_split_dir(root, ["val", "Val", "test"]))
    print(f"hashing {len(tr)} train + {len(te)} test images ...")
    htr, hte = phash_all(tr), phash_all(te)
    dup, same_cls = [], 0
    min_d = []
    for i, h in enumerate(hte):
        d = popcount64(np.bitwise_xor(htr, h))
        j = int(d.argmin()); min_d.append(int(d[j]))
        if d[j] <= args.thr:
            dup.append(te[i][0])
            same_cls += te[i][1] == tr[j][1]
    out = Path(args.out); out.mkdir(exist_ok=True)
    json.dump(dup, open(out / "near_duplicates.json", "w"))
    per_cls = Counter(c for p, c in te if p in set(dup))
    summ = {"threshold": args.thr, "n_test": len(te), "n_near_dup": len(dup),
            "frac_near_dup": len(dup) / len(te), "near_dup_same_class": same_cls,
            "per_class": dict(per_cls), "min_dist_hist": dict(Counter(min_d))}
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "min_dist_hist"}, indent=1))


if __name__ == "__main__":
    main()
