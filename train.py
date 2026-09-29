"""Train + evaluate one (model, protocol, seed) run.

Example:
  python train.py --data /content/malevis --model bs2_swin --protocol full26 --seed 0

Models: baselines (resnet50, swin_t, ...), ByteScope-v1 variants (bytescope*),
and ByteScope-v2 detection-aware models (bs2_*; see bytescope/models.py:V2).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from bytescope.data import (ByteImageDataset, _find_split_dir, benign_index, build_protocol,
                            build_transforms, make_splits, scan_split)
from bytescope.metrics import (calibrated_predictions, closed_set_metrics, detection_metrics,
                               family_metrics_from_pred, malware_score, openset_metrics,
                               openset_metrics_knn, threshold_for_fpr)
from bytescope.models import V2, build_any

TARGET_VAL_FPR = 0.02   # operating point chosen on the validation split


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def supcon_loss(z, y, t=0.1):
    """Supervised contrastive loss on L2-normalised embeddings (single view).
    Always computed in float32 with autocast disabled (the -1e9 mask overflows fp16)."""
    with torch.autocast(device_type=z.device.type, enabled=False):
        z = z.float()
        sim = z @ z.t() / t
        n = z.shape[0]
        eye = torch.eye(n, device=z.device, dtype=torch.bool)
        pos = (y[:, None] == y[None, :]) & ~eye
        sim = sim.masked_fill(eye, -1e9)
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        cnt = pos.sum(1)
        ok = cnt > 0
        if not ok.any():
            return z.sum() * 0
        return -((logp * pos).sum(1)[ok] / cnt[ok]).mean()


def balanced_bce(logit, target):
    pos, neg = target > 0.5, target <= 0.5
    l = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    parts = [l[m].mean() for m in (pos, neg) if m.any()]
    return sum(parts) / len(parts)


@torch.no_grad()
def predict(model, loader, device, amp, want_emb=False):
    model.eval()
    L, Y, D, E = [], [], [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            if want_emb:
                out, emb = model(x.to(device, non_blocking=True), return_embedding=True)
            else:
                out = model(x.to(device, non_blocking=True))
        L.append(out.float().cpu()); Y.append(y)
        det = getattr(model, "last_det", None)
        if det is not None:
            D.append(det.float().cpu())
        if want_emb:
            E.append(F.normalize(emb.float(), dim=1).cpu())
    res = {"logits": torch.cat(L).numpy(), "y": torch.cat(Y).numpy(),
           "det": torch.cat(D).numpy() if D else None}
    if want_emb:
        res["emb"] = torch.cat(E).numpy()
    return res


def efficiency(model, device, img_size, runs=50):
    n_params = sum(p.numel() for p in model.parameters())
    gflops = None
    try:
        from thop import profile
        m = copy.deepcopy(model).to("cpu").eval()
        macs, _ = profile(m, inputs=(torch.randn(1, 3, img_size, img_size),), verbose=False)
        gflops = 2 * macs / 1e9
    except Exception as e:  # noqa
        print("FLOPs profiling skipped:", e)
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(runs):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return {"params_M": n_params / 1e6, "GFLOPs": gflops,
            "latency_ms_bs1": 1000 * (time.perf_counter() - t) / runs, "device": str(device)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="bs2_swin")
    ap.add_argument("--protocol", default="full26", choices=["closed25", "mal25", "full26", "openset", "osr"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--aug", default="bytescope", choices=["bytescope", "none"])
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--exclude", default=None, help="json list of test paths to drop (dedup)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--knn_k", type=int, default=50)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    cfg = V2.get(args.model)

    train_dir = _find_split_dir(Path(args.data), ["train", "Train", "training"])
    classes_all = sorted({cls for _, cls in scan_split(train_dir)})
    proto = build_protocol(classes_all, args.protocol, args.fold, seed=0)
    exclude = set(json.load(open(args.exclude))) if args.exclude else None
    tr, va, te_k, te_u = make_splits(args.data, proto, args.seed, exclude)
    classes = proto.known
    b_idx = benign_index(classes)
    tag = f"{args.model}_{proto.name}_aug-{args.aug}_s{args.seed}"
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{tag}] train={len(tr)} val={len(va)} test_known={len(te_k)} test_unknown={len(te_u)} "
          f"benign_idx={b_idx} device={device}")

    ttr = build_transforms(args.img, True, args.aug)
    tte = build_transforms(args.img, False)
    ds_tr = ByteImageDataset(tr, classes, ttr)
    sampler = None
    if cfg and cfg["oversample"] > 1 and b_idx >= 0:
        w = [cfg["oversample"] if c == classes[b_idx] else 1.0 for _, c in tr]
        sampler = WeightedRandomSampler(w, num_samples=len(tr), replacement=True,
                                        generator=torch.Generator().manual_seed(args.seed))
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=sampler is None, sampler=sampler,
                       num_workers=args.workers, pin_memory=amp, drop_last=True)
    mk = lambda items, t: DataLoader(ByteImageDataset(items, classes, t), batch_size=args.bs, shuffle=False,
                                     num_workers=args.workers, pin_memory=amp)
    dl_va, dl_te = mk(va, tte), mk(te_k, tte)

    model = build_any(args.model, len(classes), not args.no_pretrained, args.img).to(device)
    back, new = [], []
    for n, p in model.named_parameters():
        is_back = (n.startswith(("cnn.", "tr.", "m.", "body.cnn.", "body.tr.", "body.m."))
                   and not any(k in n for k in ("head", "classifier", "fc.")))
        (back if is_back else new).append(p)
    opt = torch.optim.AdamW([{"params": back, "lr": args.lr}, {"params": new, "lr": args.lr * 10}],
                            weight_decay=args.wd)
    steps = args.epochs * len(dl_tr)
    warm = len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler(enabled=amp)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best, bad, hist = -1, 0, []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tl = 0.0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                if cfg:
                    logits, zn = model(x, return_embedding=True)
                    if cfg["margin"] > 0:
                        logits = logits - model.scale * cfg["margin"] * F.one_hot(y, logits.shape[1])
                    loss = crit(logits.float(), y)
                    if cfg["supcon"] > 0:
                        loss = loss + cfg["supcon"] * supcon_loss(zn.float(), y)
                    if model.last_det is not None and b_idx >= 0:
                        loss = loss + balanced_bce(model.last_det.float(), (y != b_idx).float())
                else:
                    loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item()
        pv = predict(model, dl_va, device, amp)
        mv = closed_set_metrics(pv["logits"], pv["y"], classes)
        hist.append({"epoch": ep, "train_loss": tl / max(len(dl_tr), 1), "val_macro_f1": mv["macro_f1"],
                     "val_acc": mv["accuracy"]})
        print(hist[-1], flush=True)
        if mv["macro_f1"] > best:
            best, bad = mv["macro_f1"], 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop"); break
    train_time = time.time() - t0

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    want_emb = bool(te_u)
    pv = predict(model, dl_va, device, amp)
    pt = predict(model, dl_te, device, amp, want_emb)
    res = {"args": vars(args), "classes": classes, "unknown_classes": proto.unknown, "benign_idx": b_idx,
           "history": hist, "train_time_s": train_time,
           "test": closed_set_metrics(pt["logits"], pt["y"], classes),
           "efficiency": efficiency(model, device, args.img)}
    save = {"logits_known": pt["logits"], "y_known": pt["y"], "logits_val": pv["logits"], "y_val": pv["y"]}
    if pt["det"] is not None:
        save["det_known"], save["det_val"] = pt["det"], pv["det"]
    if b_idx >= 0:
        sv = malware_score(pv["logits"], b_idx, pv["det"])
        tau = threshold_for_fpr(sv[pv["y"] == b_idx], TARGET_VAL_FPR)
        res["detection"] = detection_metrics(pt["logits"], pt["y"], b_idx, pt["det"], tau)
        pred = calibrated_predictions(pt["logits"], b_idx, pt["det"], tau)
        res["test_calibrated"] = family_metrics_from_pred(pred, pt["y"], len(classes))
    if te_u:
        dl_u = mk(te_u, tte)
        pu = predict(model, dl_u, device, amp, True)
        res["openset"] = openset_metrics(pt["logits"], pu["logits"])
        bank = predict(model, mk(tr, tte), device, amp, True)["emb"]
        res["openset"]["knn"] = openset_metrics_knn(bank, pt["emb"], pu["emb"], args.knn_k)
        save["logits_unknown"] = pu["logits"]
    np.savez_compressed(out / "test_logits.npz", **save)
    json.dump(res, open(out / "metrics.json", "w"), indent=1)
    t = res["test"]
    msg = f"TEST acc={t['accuracy']:.4f} macroF1={t['macro_f1']:.4f} ECE={t['ece']:.4f}"
    if "detection" in res:
        d = res["detection"]
        msg += f" | det AUROC={d['AUROC']:.4f} TPR@1%FPR={d['TPR@FPR1%']:.4f} argmaxFPR={d['argmax_FPR']:.4f}"
    if te_u:
        msg += f" | open-set knnAUROC={res['openset']['knn']['AUROC']:.4f} energyAUROC={res['openset']['energy']['AUROC']:.4f}"
    print(msg, flush=True)


if __name__ == "__main__":
    main()
