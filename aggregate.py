"""Collect all runs into one flat CSV plus a compact summary.json (mean/std over
seeds or folds) that is used to fill the manuscript tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bytescope.metrics import mcnemar


def flat(prefix, d, out):
    for k, v in d.items():
        if isinstance(v, dict):
            flat(f"{prefix}{k}.", v, out)
        elif isinstance(v, (int, float)) or v is None:
            out[f"{prefix}{k}"] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    rows = []
    for mf in sorted(Path(a.runs).glob("*/metrics.json")):
        m = json.load(open(mf)); ar = m["args"]
        if ar["protocol"] in ("closed25", "openset"):   # superseded (benign-class bug)
            continue
        r = {"run": mf.parent.name, "model": ar["model"], "protocol": ar["protocol"], "fold": ar["fold"],
             "aug": ar["aug"], "seed": ar["seed"], "train_time_s": m.get("train_time_s")}
        t = m["test"]
        r.update(acc=t["accuracy"], macro_f1=t["macro_f1"], macro_p=t["macro_precision"],
                 macro_r=t["macro_recall"], ece=t["ece"])
        if "Other" in t.get("per_class", {}):
            r["benign_recall"] = t["per_class"]["Other"]["recall"]
        flat("eff.", {k: v for k, v in m["efficiency"].items() if k != "device"}, r)
        if "detection" in m:
            flat("det.", m["detection"], r)
        if "test_calibrated" in m:
            flat("cal.", m["test_calibrated"], r)
        if "openset" in m:
            flat("os.", m["openset"], r)
        for extra in ("reeval", "robustness", "xai"):
            f = mf.parent / f"{extra}.json"
            if f.exists():
                d = json.load(open(f))
                if extra == "reeval" and "detection" in d:
                    det = d.pop("detection")
                    if "detection" not in m:   # old runs: use the logits-based re-evaluation
                        flat("det.", det, r)
                flat(f"{extra[:3]}.", d, r)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "all_runs.csv", index=False)
    print(f"{len(df)} runs collected")

    summ = {}
    num = [c for c in df.columns if df[c].dtype.kind in "fi" and c not in ("seed", "fold")]
    for (p, mdl, aug), g in df.groupby(["protocol", "model", "aug"]):
        key = f"{p}|{mdl}|{aug}"
        summ[key] = {"n": int(len(g))}
        for c in num:
            v = g[c].dropna()
            if len(v):
                summ[key][c] = [round(float(v.mean()), 5), round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 5)]
    # McNemar: bs2_swin vs every other full26 model, per seed
    tests = []
    for s in sorted(df.seed.unique()):
        ref = Path(a.runs) / f"bs2_swin_noOversamp_full26_aug-bytescope_s{s}" / "test_logits.npz"
        if not ref.exists():
            continue
        R = np.load(ref); y = R["y_known"]; pa = R["logits_known"].argmax(1)
        for mdl in df[df.protocol == "full26"].model.unique():
            f = Path(a.runs) / f"{mdl}_full26_aug-bytescope_s{s}" / "test_logits.npz"
            if mdl == "bs2_swin_noOversamp" or not f.exists():
                continue
            pb = np.load(f)["logits_known"].argmax(1)
            tests.append({"seed": int(s), "vs": mdl, **mcnemar(pa, pb, y)})
    summ["_mcnemar_vs_final"] = tests
    dd = Path("dedup/summary.json")
    if dd.exists():
        summ["_dedup_audit"] = {k: v for k, v in json.load(open(dd)).items() if k != "min_dist_hist"}
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print("tables written to", out)


if __name__ == "__main__":
    main()
