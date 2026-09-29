"""Run the complete (v2) experiment grid. Resumable: runs whose metrics.json
exists are skipped, so it is safe to restart after a Colab disconnect.

  python run_all.py --data /content/malevis --seeds 0,1,2 --stage all

Stages (in order): dedup, reeval, v2, baselines, mal25, osr, robustness, xai, aggregate
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BASELINES = ["resnet50", "densenet121", "effnetv2_s", "convnext_t", "vit_s", "deit_s",
             "swin_t", "coatnet_0", "maxvit_t"]
V1 = ["bytescope", "bytescope_cnnonly", "bytescope_tronly", "bytescope_concat_noOARE",
      "bytescope_concat", "bytescope_noOARE"]
V2_MAIN = ["bs2_swin", "bs2_hybrid"]
V2_ABL = ["bs2_swin_noDet", "bs2_swin_noSupCon", "bs2_swin_noMargin", "bs2_swin_noOversamp"]
MAL25 = ["densenet121", "convnext_t", "swin_t", "bytescope", "bs2_swin", "bs2_swin_noOversamp"]
OSR = ["convnext_t", "swin_t", "maxvit_t", "bytescope", "bs2_swin", "bs2_swin_noOversamp"]
FINAL = "bs2_swin_noOversamp"   # final ByteScope configuration (uniform sampling)
FINAL_ABL = ["bs2_final_noDet", "bs2_final_noSupCon", "bs2_final_noMargin"]
ROBUST = ["resnet50", "convnext_t", "swin_t", "maxvit_t", "bytescope", "bs2_swin", "bs2_hybrid",
          "bs2_swin_noOversamp"] + FINAL_ABL
XAI = ["resnet50", "convnext_t", "swin_t", "bytescope", "bs2_swin", "bs2_swin_noOversamp"]
FAILED = []


def sh(cmd):
    print(">>", " ".join(cmd), flush=True)
    ok = subprocess.run(cmd).returncode == 0
    if not ok:
        print("!! FAILED:", " ".join(cmd), flush=True)
    return ok


def train(a, model, protocol, seed, fold=0, aug="bytescope"):
    pname = protocol if protocol != "osr" else f"osr_f{fold}"
    tag = f"{model}_{pname}_aug-{aug}_s{seed}"
    if (Path(a.out) / tag / "metrics.json").exists():
        return tag
    bs = a.bs
    while bs >= 4:  # retry with a smaller batch on failure (e.g. CUDA out of memory)
        if sh([sys.executable, "train.py", "--data", a.data, "--model", model, "--protocol", protocol,
               "--fold", str(fold), "--seed", str(seed), "--epochs", str(a.epochs), "--bs", str(bs),
               "--aug", aug, "--out", a.out, "--workers", str(a.workers)]):
            keep = (protocol == "full26" and seed == 0 and (model in ROBUST or model in XAI or aug == "none"))
            ck = Path(a.out) / tag / "best.pt"
            if not keep and not a.keep_all_ckpt and ck.exists():
                ck.unlink()
            return tag
        bs //= 2
    FAILED.append(tag)
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--keep_all_ckpt", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    st = a.stage
    py = sys.executable

    if st in ("all", "dedup") and not Path("dedup/near_duplicates.json").exists():
        sh([py, "dedup_audit.py", "--data", a.data])
    if st in ("all", "reeval"):
        sh([py, "reeval.py", "--data", a.data, "--runs", a.out])
    if st in ("all", "final"):                    # final ByteScope: ablations, open set, Mal-25 first
        for s in seeds:
            for m in FINAL_ABL:
                train(a, m, "full26", s)
        for fold in range(5):
            train(a, FINAL, "osr", seeds[0], fold=fold)
        for s in seeds:
            train(a, FINAL, "mal25", s)
    if st in ("all", "v2"):                       # proposed method + its ablations first
        for s in seeds:
            for m in V2_MAIN + V2_ABL:
                train(a, m, "full26", s)
    if st in ("all", "baselines"):                # finish baselines / v1 on full26
        for s in seeds:
            for m in BASELINES + V1:
                train(a, m, "full26", s)
            train(a, "bytescope", "full26", s, aug="none")
    if st in ("all", "mal25"):
        for s in seeds:
            for m in MAL25:
                train(a, m, "mal25", s)
    if st in ("all", "osr"):
        for fold in range(5):
            for m in OSR:
                train(a, m, "osr", seeds[0], fold=fold)
    if st in ("all", "robustness"):
        for m in ROBUST:
            run = Path(a.out) / f"{m}_full26_aug-bytescope_s{seeds[0]}"
            if (run / "best.pt").exists() and not (run / "robustness.json").exists():
                sh([py, "robustness.py", "--data", a.data, "--run", str(run), "--workers", str(a.workers)])
    if st in ("all", "xai"):
        for m in XAI:
            run = Path(a.out) / f"{m}_full26_aug-bytescope_s{seeds[0]}"
            if (run / "best.pt").exists() and not (run / "xai.json").exists():
                sh([py, "xai.py", "--data", a.data, "--run", str(run)])
    if st in ("all", "reeval", "aggregate"):
        sh([py, "reeval.py", "--data", a.data, "--runs", a.out, "--force"])
        sh([py, "aggregate.py", "--runs", a.out])
    if FAILED:
        print("RUNS THAT FAILED EVEN AT SMALL BATCH:", FAILED, flush=True)


if __name__ == "__main__":
    main()
