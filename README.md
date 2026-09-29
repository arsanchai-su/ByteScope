# ByteScope

**Detection-Aware Learning for Robust Visual Malware Recognition under Leakage-Audited, Open-World Evaluation**

Official code for the manuscript by Arsanchai Sukkuea, Pensiri Akkajit, Nanda Kumar Tharu and Nat Wannawas
(Walailak University). ByteScope adds a detection-aware learning head to a vision backbone: a scaled
cosine-margin classifier, a supervised contrastive embedding and a binary benign-versus-malware head
calibrated at a fixed false-positive rate. The repository also contains the leakage-audited, open-world
evaluation protocol used in the paper (near-duplicate audit, FPR-controlled detection, family-disjoint
open-set folds, evasion proxies and explanation faithfulness).

## Repository layout

```
ByteScope/
├── bytescope/
│   ├── models.py        # ByteScope (detection-aware head), hybrid v1 architecture, timm baselines
│   ├── data.py          # splits and protocols (mal25 / full26 / osr folds) and augmentation
│   └── metrics.py       # accuracy, macro-F1, ECE, TPR@FPR, open-set AUROC/FPR95, McNemar test
├── train.py             # trains and evaluates one run
├── dedup_audit.py       # pHash near-duplicate audit of the test split
├── reeval.py            # re-evaluates trained runs on the de-duplicated test subset
├── robustness.py        # evasion proxies: benign padding, header occlusion, byte noise
├── xai.py               # Grad-CAM, deletion/insertion AUC, header attribution ratio (HAR)
├── aggregate.py         # collects all runs into tables (CSV / LaTeX / JSON)
├── run_all.py           # resumable driver for the complete experiment grid
├── notebooks/
│   └── ByteScope_Colab.ipynb   # one-click Google Colab version (self-contained)
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/arsanchai-su/ByteScope.git
cd ByteScope
conda create -n bytescope python=3.11 -y && conda activate bytescope
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Data

MaleVis is available from its authors for academic use at
<https://web.cs.hacettepe.edu.tr/~selman/malevis/>. Download `malevis_train_val_224x224.zip` and unzip it;
the folder must contain `train/<class>/*.png` and `val/<class>/*.png`.

* The official **val** split is used **only as the test set**.
* 15% of the official train split (stratified, seeded) is held out as the validation set for early
  stopping and threshold calibration.

## Reproducing the paper

```bash
python run_all.py --data /path/to/malevis --seeds 0,1,2 --stage all
```

The driver is resumable: runs whose `runs/<tag>/metrics.json` already exists are skipped. Individual stages
can be run with `--stage dedup | final | v2 | baselines | mal25 | osr | robustness | xai | reeval | aggregate`.
If the GPU runs out of memory, the driver retries automatically with a smaller batch; you can also pass `--bs 16`.

A single run:

```bash
python train.py --data /path/to/malevis --model bs2_swin_noOversamp --protocol full26 --seed 0
```

### Google Colab

Open `notebooks/ByteScope_Colab.ipynb` in Colab, select a GPU runtime and choose *Runtime → Run all*.
The notebook embeds the code, downloads MaleVis into Google Drive and saves all results to
`MyDrive/ByteScope/`, so it can be restarted after a disconnect.

### Model names used in the code

| Name in paper | `--model` |
|---|---|
| ByteScope (final configuration, uniform sampling) | `bs2_swin_noOversamp` |
| ByteScope ablations (w/o detection head / SupCon / margin) | `bs2_final_noDet`, `bs2_final_noSupCon`, `bs2_final_noMargin` |
| ByteScope + R (benign oversampling) | `bs2_swin` |
| Hybrid architecture (v1) and its variants | `bytescope`, `bytescope_cnnonly`, `bytescope_tronly`, `bytescope_concat`, … |
| Baselines | `resnet50`, `densenet121`, `effnetv2_s`, `convnext_t`, `vit_s`, `deit_s`, `swin_t`, `coatnet_0`, `maxvit_t` |

Protocols: `mal25` (25 malware families), `full26` (25 families + benign) and `osr` (five family-disjoint
open-set folds, `--fold 0..4`).

## Outputs

| Result | File |
|---|---|
| Near-duplicate audit and list | `dedup/summary.json`, `dedup/near_duplicates.json` |
| Main comparison and ablations | `tables/all_runs.csv`, `tables/main_table.tex` |
| Unknown-family rejection | `tables/openset_table.tex` |
| Robustness (evasion proxies) | `tables/all_runs.csv` (columns `rob_*`), `runs/<run>/robustness.json` |
| Explanation faithfulness | `tables/all_runs.csv` (`del_auc`, `ins_auc`, `HAR`), `runs/<run>/xai.json` |
| Significance tests | `tables/mcnemar.csv` |
| Grad-CAM examples | `runs/<run>/cam_examples/*.png` |

## Citation

If you use this code or the MaleVis near-duplicate list, please cite:

```bibtex
@article{sukkuea2026bytescope,
  title   = {ByteScope: Detection-Aware Learning for Robust Visual Malware Recognition under
             Leakage-Audited, Open-World Evaluation},
  author  = {Sukkuea, Arsanchai and Akkajit, Pensiri and Tharu, Nanda Kumar and Wannawas, Nat},
  journal = {Machine Learning and Knowledge Extraction},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code is released under the MIT License (see `LICENSE`). MaleVis is distributed by its original authors
under their own terms and is not included in this repository.

## Acknowledgments

This research was funded by Walailak University under the New Researcher Development scheme,
grant number WU69283.
