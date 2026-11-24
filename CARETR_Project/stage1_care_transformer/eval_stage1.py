
import os
import sys
import json
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from timm import create_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import models.care_model  # noqa: F401  (registers CARETrans_S0)

MODEL_PATH = os.path.join(ROOT, "checkpoints", "caret_s0_448_bal_best.pth")
DATA_ROOT = os.path.join(ROOT, "data", "grids_448")
DEFAULT_OUT = os.path.join(ROOT, "results", "stage1", "stage1_448_bal_eval_meta.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256

# ImageFolder alphabetical order: F=0 (non-crack), T=1 (crack)
IMG_SIZE = 448

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.450, 0.400], [0.229, 0.225, 0.225]),
])
SWEEP_THRS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def load_model(path=MODEL_PATH):
    model = create_model('CARETrans_S0', pretrained=False, num_classes=2)
    for m in model.modules():
        if hasattr(m, 'drop_path_prob'):
            m.drop_path_prob = 0.05
    if hasattr(model, 'head') and hasattr(model.head, 'fc'):
        model.head.fc = nn.Sequential(nn.Dropout(0.45), model.head.fc)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def infer_probs(model, loader):
    """返回 (gts, probs): 真实标签 与 裂纹概率 softmax[:,1]"""
    gts, probs = [], []
    for x, y in tqdm(loader, desc="infer", ncols=80):
        out = model(x.to(DEVICE))
        if isinstance(out, tuple):
            out = out[0]
        p = F.softmax(out, dim=1)[:, 1]
        gts.extend(y.numpy().tolist())
        probs.extend(p.cpu().numpy().tolist())
    return gts, probs


def set_img_size(size):
    """重建 transform(评测 448 数据时必须调用, 否则会被缩回 224)"""
    global transform
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.450, 0.400], [0.229, 0.225, 0.225]),
    ])


def metrics_at(gts, probs, thr):
    """返回 dict: acc / macro_f1 / crack_f1 / crack_p / crack_r / noncrack_f1"""
    preds = [1 if p >= thr else 0 for p in probs]
    cm = confusion_matrix(gts, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    cp = tp / (tp + fp) if tp + fp else 0.0
    cr = tp / (tp + fn) if tp + fn else 0.0
    cf1 = 2 * cp * cr / (cp + cr) if cp + cr else 0.0
    np_ = tn / (tn + fn) if tn + fn else 0.0
    nr = tn / (tn + fp) if tn + fp else 0.0
    nf1 = 2 * np_ * nr / (np_ + nr) if np_ + nr else 0.0
    return {"acc": acc, "macro_f1": (cf1 + nf1) / 2, "crack_f1": cf1,
            "crack_p": cp, "crack_r": cr, "noncrack_f1": nf1}


def evaluate_split(model, split, thr, data_root=DATA_ROOT, balance=False, seed=42):
    """Single-split evaluation; returns (result_dict, gts, probs, nF, nT)."""
    folder = os.path.join(data_root, split)
    ds = datasets.ImageFolder(folder, transform=transform)
    nF_all = sum(1 for _, y in ds.samples if y == 0)   # non-crack (original)
    nT = sum(1 for _, y in ds.samples if y == 1)       # crack
    if balance:
        # Downsample non-crack grids to match the crack count (files are kept on disk).
        import random as _r
        from torch.utils.data import Subset
        idx_f = [i for i, (_, y) in enumerate(ds.samples) if y == 0]
        idx_t = [i for i, (_, y) in enumerate(ds.samples) if y == 1]
        rs = _r.Random(seed)
        rs.shuffle(idx_f)
        n_keep = min(len(idx_f), len(idx_t))
        sel = sorted(idx_f[:n_keep] + idx_t)
        print(f"    [1:1] {split}: F {n_keep}/{nF_all} (kept on disk) + T {len(idx_t)} = {len(sel)}")
        ds = Subset(ds, sel)
        nF = n_keep
    else:
        nF = nF_all
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    gts, probs = infer_probs(model, loader)

    preds = [1 if p >= thr else 0 for p in probs]
    cm = confusion_matrix(gts, preds, labels=[0, 1])
    correct = int(cm[0][0] + cm[1][1])
    total = len(gts)
    f_acc = 100 * cm[0][0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
    t_acc = 100 * cm[1][1] / cm[1].sum() if cm[1].sum() > 0 else 0.0
    report = classification_report(gts, preds, target_names=["F", "T"],
                                   digits=4, output_dict=True)
    m = metrics_at(gts, probs, thr)

    # ---- core metrics ----
    result = {
        "set": split,
        "total": total,
        "correct": correct,
        "acc": round(100 * correct / total, 4),
        "F_acc": round(f_acc, 4),
        "T_acc": round(t_acc, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
     
        "n_F_noncrack": nF,
        "n_T_crack": nT,
        "F_T_ratio": round(nF / max(nT, 1), 3),
        "all_F_baseline_acc": round(100 * nF / total, 4),
        "crack_precision": round(m["crack_p"], 4),
        "crack_recall": round(m["crack_r"], 4),
        "crack_f1": round(m["crack_f1"], 4),
        "macro_f1": round(m["macro_f1"], 4),
    }
    return result, gts, probs, nF, nT


def main():
    global BATCH_SIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE,
                    help="评 448 数据时必须设为 448")
    ap.add_argument("--splits", default="train,valid,test")
    ap.add_argument("--balance-f-to-t", type=int, default=1,
                    help="1=downsample non-crack grids to 1:1 per split (files are kept on disk)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--sweep-on", default="valid",
                    help="在哪个 split 上做阈值扫描(默认 valid, 不应用 test)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    BATCH_SIZE = args.batch_size
    set_img_size(args.img_size)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    print(f"模型: {args.model}")
    print(f"数据: {args.data_root}")
    print(f"设备: {DEVICE}   阈值: {args.thr}   输入: {args.img_size}   评估集: {splits}")

    model = load_model(args.model)
    results, cache = [], {}
    for s in splits:
        r, gts, probs, nF, nT = evaluate_split(model, s, args.thr, args.data_root,
                                               bool(args.balance_f_to_t), args.seed)
        results.append(r)
        cache[s] = (gts, probs)
        print(f"\n=== {s}: {r['total']} 格 (F={nF}, T={nT}, 1:{r['F_T_ratio']}) ===")
        print(f"  准确率 = {r['acc']}%   (全预测'非裂纹'基线 = {r['all_F_baseline_acc']}%)")
        print(f"  F/非裂纹 acc={r['F_acc']}%   T/裂纹 acc={r['T_acc']}%")
        print(f"  裂纹 P={r['crack_precision']}  R={r['crack_recall']}  F1={r['crack_f1']}")
        print(f"  宏平均 F1 = {r['macro_f1']}")
        print(f"  混淆矩阵 = {r['confusion_matrix']}")

    # ---- 阈值扫描(默认只在 valid 上, 避免污染 test) ----
    sweep = []
    if args.sweep_on in cache:
        gts, probs = cache[args.sweep_on]
        print(f"\n[阈值扫描] 在 {args.sweep_on} 上")
        print(f"{'thr':>5} | {'acc%':>7} | {'crack P':>8} | {'crack R':>8} | "
              f"{'crack F1':>9} | {'macro F1':>9}")
        print("-" * 66)
        for t in SWEEP_THRS:
            m = metrics_at(gts, probs, t)
            sweep.append({"thr": t, "acc": round(100 * m["acc"], 4),
                          "crack_p": round(m["crack_p"], 4),
                          "crack_r": round(m["crack_r"], 4),
                          "crack_f1": round(m["crack_f1"], 4),
                          "macro_f1": round(m["macro_f1"], 4)})
            print(f"{t:>5.2f} | {100 * m['acc']:>7.2f} | {m['crack_p']:>8.4f} | "
                  f"{m['crack_r']:>8.4f} | {m['crack_f1']:>9.4f} | {m['macro_f1']:>9.4f}")
        best = max(sweep, key=lambda x: x["macro_f1"])
        print("-" * 66)

        print(f"best threshold by macro F1 = {best['thr']}  (macro F1={best['macro_f1']})")

    meta = {
        "model": os.path.basename(args.model),
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": args.data_root,
        "threshold": args.thr,
        "note": "Evaluation on the image-level split; train/valid/test images are disjoint; "
                "the threshold sweep is performed on valid only.",
        "results": results,
        "threshold_sweep": sweep,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
