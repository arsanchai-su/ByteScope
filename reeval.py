"""Re-evaluate saved test logits WITHOUT retraining:
  * malware-vs-benign detection metrics (AUROC, TPR@FPR, argmax FPR)
  * accuracy / macro-F1 on the de-duplicated test subset
Writes runs/<run>/reeval.json for every run whose protocol contains the benign class.

  python reeval.py --data /content/malevis --runs runs --dedup dedup/near_duplicates.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bytescope.data import _find_split_dir, benign_index, build_protocol, make_splits, scan_split
from bytescope.metrics import detection_metrics, family_metrics_from_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--dedup", default="dedup/near_duplicates.json")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    dup = set(json.load(open(a.dedup))) if Path(a.dedup).exists() else set()
    all_classes = sorted({c for _, c in scan_split(_find_split_dir(Path(a.data), ["train", "Train"]))})
    n = 0
    for mf in sorted(Path(a.runs).glob("*/metrics.json")):
        out = mf.parent / "reeval.json"
        npz = mf.parent / "test_logits.npz"
        if (out.exists() and not a.force) or not npz.exists():
            continue
        m = json.load(open(mf)); ar = m["args"]
        if ar["protocol"] in ("closed25", "openset"):   # superseded protocols (benign-class bug)
            continue
        classes = m["classes"]
        b = benign_index(classes)
        Z = np.load(npz)
        L, y = Z["logits_known"], Z["y_known"]
        det = Z["det_known"] if "det_known" in Z else None
        res = {}
        if b >= 0:
            res["detection"] = detection_metrics(L, y, b, det)
        proto = build_protocol(all_classes, ar["protocol"], ar["fold"], seed=0)
        _, _, te_k, _ = make_splits(a.data, proto, ar["seed"])
        if len(te_k) == len(y) and dup:
            keep = np.array([p not in dup for p, _ in te_k])
            for name, msk in (("dedup", keep), ("dup_only", ~keep)):
                if msk.any():
                    res[name] = {"n": int(msk.sum()),
                                 **family_metrics_from_pred(L[msk].argmax(1), y[msk], len(classes), present_only=True)}
            if b >= 0 and keep.any() and 0 < (y[keep] != b).sum() < keep.sum():
                res["dedup_detection"] = detection_metrics(L[keep], y[keep], b, det[keep] if det is not None else None)
        json.dump(res, open(out, "w"), indent=1)
        n += 1
    print(f"re-evaluated {n} runs")


if __name__ == "__main__":
    main()
