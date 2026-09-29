"""Grad-CAM + quantitative faithfulness (deletion / insertion AUC) and the
Header Attribution Ratio (share of CAM mass in the top 10% of rows, i.e. the
PE header region).

Usage:
  python xai.py --data <root> --run runs/bytescope_full26_aug-bytescope_s0 --n 500 --save_examples 12
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from bytescope.data import build_protocol, build_transforms, make_splits
from bytescope.models import V2, build_any

TARGET_LAYERS = {  # module whose output is the explained feature map
    "bytescope": "merge", "bytescope_noOARE": "merge", "bytescope_concat": "merge",
    "bytescope_concat_noOARE": "merge", "bytescope_cnnonly": "merge", "bytescope_tronly": "merge",
    "resnet50": "m.layer4", "densenet121": "m.features", "convnext_t": "m.stages.3",
    "effnetv2_s": "m.conv_head", "swin_t": "m.layers.3", "coatnet_0": "m.stages.3",
    "maxvit_t": "m.stages.3", "swin_t_v2": "m.layers.3",
}


class GradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        layer = dict(model.named_modules())[layer_name]
        self.act = self.grad = None
        layer.register_forward_hook(self._fh)

    def _fh(self, m, i, o):
        self.act = o
        if o.requires_grad:  # skipped during the no-grad deletion/insertion passes
            o.register_hook(lambda g: setattr(self, "grad", g))

    def __call__(self, x, cls=None):
        self.model.zero_grad()
        out = self.model(x)
        cls = out.argmax(1) if cls is None else cls
        out.gather(1, cls[:, None]).sum().backward()
        a, g = self.act, self.grad
        if a.dim() == 4 and a.shape[1] < a.shape[-1] and a.shape[1] == a.shape[2]:  # NHWC (swin)
            a, g = a.permute(0, 3, 1, 2), g.permute(0, 3, 1, 2)
        w = g.mean((2, 3), keepdim=True)
        cam = F.relu((w * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.amin((1, 2), keepdim=True)
        cam = cam / (cam.amax((1, 2), keepdim=True) + 1e-8)
        return cam.detach(), cls.detach(), out.detach()


@torch.no_grad()
def del_ins_auc(model, x, cam, cls, steps=20):
    """RISE-style deletion (lower = better) and insertion (higher = better)."""
    b, _, h, w = x.shape
    order = cam.flatten(1).argsort(1, descending=True)
    base = torch.zeros_like(x)
    blur = F.avg_pool2d(x, 11, 1, 5)
    n = h * w
    dels, ins = [], []
    for k in np.linspace(0, n, steps + 1).astype(int):
        mask = torch.zeros(b, n, device=x.device)
        if k > 0:
            mask.scatter_(1, order[:, :k], 1.0)
        mask = mask.view(b, 1, h, w)
        xd = x * (1 - mask) + base * mask
        xi = blur * (1 - mask) + x * mask
        dels.append(F.softmax(model(xd), 1).gather(1, cls[:, None])[:, 0])
        ins.append(F.softmax(model(xi), 1).gather(1, cls[:, None])[:, 0])
    auc = lambda s: torch.trapz(torch.stack(s, 1), dx=1.0 / steps, dim=1)
    return auc(dels).cpu().numpy(), auc(ins).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--save_examples", type=int, default=12)
    args = ap.parse_args()
    run = Path(args.run)
    meta = json.load(open(run / "metrics.json"))
    a, classes = meta["args"], meta["classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_any(a["model"], len(classes), pretrained=False, img_size=a["img"]).to(device).eval()
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    layer = ("body." + TARGET_LAYERS[V2[a["model"]]["backbone"]]) if a["model"] in V2 else TARGET_LAYERS[a["model"]]
    cammer = GradCAM(model, layer)

    all_classes = sorted(set(classes) | set(meta["unknown_classes"]))
    _, _, te, _ = make_splits(args.data, build_protocol(all_classes, a["protocol"], a["fold"], seed=0), a["seed"])
    random.Random(0).shuffle(te)
    te = te[: args.n]
    tf = build_transforms(a["img"], False)
    cidx = {c: i for i, c in enumerate(classes)}
    D, I, HAR = [], [], []
    ex_dir = run / "cam_examples"; ex_dir.mkdir(exist_ok=True)
    saved = 0
    for s in range(0, len(te), args.bs):
        batch = te[s:s + args.bs]
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in batch]).to(device)
        cam, cls, _ = cammer(x)
        cammer.act = cammer.grad = None
        d, i = del_ins_auc(model, x, cam, cls)
        D += d.tolist(); I += i.tolist()
        rows = cam.sum(2)  # B,H
        top = max(1, int(0.1 * rows.shape[1]))
        HAR += (rows[:, :top].sum(1) / (rows.sum(1) + 1e-8)).cpu().tolist()
        for j, (p, c) in enumerate(batch):
            if saved >= args.save_examples:
                break
            im = Image.open(p).convert("RGB").resize((a["img"], a["img"]))
            heat = (cam[j].cpu().numpy() * 255).astype(np.uint8)
            hm = Image.fromarray(heat).convert("L")
            col = Image.merge("RGB", (hm, Image.new("L", hm.size, 0), Image.eval(hm, lambda v: 255 - v)))
            Image.blend(im, col, 0.45).save(ex_dir / f"{saved:02d}_{c}_pred-{classes[cls[j]]}.png")
            saved += 1
    res = {"n": len(D), "deletion_auc_mean": float(np.mean(D)), "deletion_auc_std": float(np.std(D)),
           "insertion_auc_mean": float(np.mean(I)), "insertion_auc_std": float(np.std(I)),
           "header_attr_ratio_mean": float(np.mean(HAR)), "uniform_baseline_HAR": 0.1}
    print(res)
    json.dump(res, open(run / "xai.json", "w"), indent=1)


if __name__ == "__main__":
    main()
