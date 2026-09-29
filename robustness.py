"""Evasion-style robustness evaluation in the image domain.

Perturbations (applied to test images of a trained run):
  benign_pad@f : append rows taken from a *benign training* image (fraction f
                 of the original height) at the bottom, then resize back.
                 Image-domain proxy for appending benign overlay / padding
                 bytes to a PE file (cf. padding attacks on MalConv).
  header_occ@f : zero the top fraction f of rows (header destroyed/obfuscated).
  byte_noise@p : replace a fraction p of pixels with uniform random bytes.

Usage:
  python robustness.py --data <root> --run runs/bytescope_full26_aug-bytescope_s0
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from bytescope.data import (IMAGENET_MEAN, IMAGENET_STD, ByteImageDataset, benign_index, build_protocol,
                            build_transforms, is_benign, make_splits)
from bytescope.metrics import (calibrated_predictions, closed_set_metrics, malware_score,
                               threshold_for_fpr)
from bytescope.models import build_any

TARGET_VAL_FPR = 0.02


class Perturbed(Dataset):
    def __init__(self, items, classes, kind, level, donors, img=224, seed=0):
        self.items, self.kind, self.level, self.donors, self.img = items, kind, level, donors, img
        self.cidx = {c: i for i, c in enumerate(classes)}
        self.rng = random.Random(seed)
        self.donor_idx = [self.rng.randrange(len(donors)) if donors else 0 for _ in items]
        self.post = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, cls = self.items[i]
        im = Image.open(path).convert("RGB")
        w, h = im.size
        if self.kind == "benign_pad" and self.level > 0:
            d = Image.open(self.donors[self.donor_idx[i]]).convert("RGB").resize((w, h))
            add = int(round(h * self.level))
            canvas = Image.new("RGB", (w, h + add))
            canvas.paste(im, (0, 0))
            canvas.paste(d.crop((0, 0, w, add)), (0, h))
            im = canvas.resize((w, h), Image.BILINEAR)
        elif self.kind == "header_occ" and self.level > 0:
            a = np.array(im)
            a[: int(round(h * self.level))] = 0
            im = Image.fromarray(a)
        elif self.kind == "byte_noise" and self.level > 0:
            a = np.array(im)
            r = np.random.default_rng(i)
            m = r.random(a.shape[:2]) < self.level
            a[m] = r.integers(0, 256, (m.sum(), 3), dtype=np.uint8)
            im = Image.fromarray(a)
        return self.post(im), self.cidx[cls]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    run = Path(args.run)
    meta = json.load(open(run / "metrics.json"))
    a = meta["args"]
    classes = meta["classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_any(a["model"], len(classes), pretrained=False, img_size=a["img"]).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.eval()

    all_classes = sorted(set(classes) | set(meta["unknown_classes"]))
    proto = build_protocol(all_classes, a["protocol"], a["fold"], seed=0)
    tr, va, te_k, _ = make_splits(args.data, proto, a["seed"])
    b_idx = benign_index(classes)
    donors = [p for p, c in tr if is_benign(c)]
    malware_test = [it for it in te_k if not is_benign(it[1])]

    def infer(loader):
        L, D, Y = [], [], []
        with torch.no_grad():
            for x, y in loader:
                L.append(model(x.to(device)).float().cpu()); Y.append(y)
                if getattr(model, "last_det", None) is not None:
                    D.append(model.last_det.float().cpu())
        return torch.cat(L).numpy(), (torch.cat(D).numpy() if D else None), torch.cat(Y).numpy()

    # common operating point for every model: 2% FPR on the validation split
    tau = None
    if b_idx >= 0:
        dl_va = DataLoader(ByteImageDataset(va, classes, build_transforms(a["img"], False)),
                           batch_size=args.bs, num_workers=args.workers)
        Lv, Dv, Yv = infer(dl_va)
        tau = threshold_for_fpr(malware_score(Lv, b_idx, Dv)[Yv == b_idx], TARGET_VAL_FPR)

    grid = {"clean": [0.0], "benign_pad": [0.1, 0.25, 0.5, 1.0],
            "header_occ": [0.05, 0.1, 0.2], "byte_noise": [0.01, 0.05, 0.1]}
    results = {"tau_val_fpr2": tau}
    for kind, levels in grid.items():
        for lv in levels:
            ds = Perturbed(malware_test, classes, "benign_pad" if kind == "clean" else kind, lv, donors, a["img"])
            dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers)
            L, D, Y = infer(dl)
            m = closed_set_metrics(L, Y, classes)
            r = {"family_accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
            if b_idx >= 0:
                r["evasion_rate_argmax"] = float((L.argmax(1) == b_idx).mean())
                pred = calibrated_predictions(L, b_idx, D, tau)
                r["evasion_rate"] = float((pred == b_idx).mean())
            results[f"{kind}@{lv}"] = r
            print(kind, lv, r, flush=True)
    json.dump(results, open(run / "robustness.json", "w"), indent=1)


if __name__ == "__main__":
    main()
