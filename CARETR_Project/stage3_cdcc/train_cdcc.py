
import os
import sys
import csv
import json
import time
import random
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.cdcc_model import CDCC, PositionAgnosticMLP

TYPE_MAP = {"grid": 0, "horizontal": 1, "vertical": 2}


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VecDataset(Dataset):
    def __init__(self, root, is_train=True):
        self.samples, self.is_train = [], is_train
        for cls, lab in TYPE_MAP.items():
            d = os.path.join(root, cls)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.endswith(".pt"):
                    self.samples.append((os.path.join(d, f), lab))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, lab = self.samples[i]
        v = torch.load(p, weights_only=True)["vector"].float()
        if self.is_train:                     # 轻量扰动(与既有实现一致)
            v = v + torch.randn_like(v) * 0.01
            v = v * random.uniform(0.96, 1.04)
        return v, torch.tensor(lab, dtype=torch.long)


def build_model(kind, conv_kernels=8, hidden=256, drop=0.25):
    if kind == "cdcc":
        return CDCC(in_ch=1, conv_kernels=conv_kernels, hidden_dim1=hidden,
                    hidden_dim2=hidden, num_classes=3, drop=drop)
    if kind == "agnostic":
        return PositionAgnosticMLP(input_dim=48, hidden_dim1=hidden,
                                   hidden_dim2=hidden, num_classes=3, drop=drop)
    raise ValueError(f"unknown model kind: {kind}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    for x, y in loader:
        preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
        gts.extend(y.tolist())
    cm = confusion_matrix(gts, preds, labels=[0, 1, 2]).tolist()
    acc = 100 * (cm[0][0] + cm[1][1] + cm[2][2]) / max(len(gts), 1)
    report = classification_report(gts, preds, target_names=list(TYPE_MAP),
                                   digits=4, output_dict=True, zero_division=0)
    return acc, cm, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["pnr", "nopnr"])
    ap.add_argument("--model", default="cdcc", choices=["cdcc", "agnostic"])
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data", "cdcc_data"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "stage3"))
    ap.add_argument("--ckpt-dir", default=os.path.join(ROOT, "checkpoints"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--conv-kernels", type=int, default=8, help="CDCC 卷积输出通道数")
    ap.add_argument("--hidden", type=int, default=256, help="MLP 隐层宽度")
    ap.add_argument("--drop", type=float, default=0.25)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--tag-suffix", default="", help="输出文件名后缀, 避免覆盖")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"{args.model}_{args.arm}{args.tag_suffix}"
    root = os.path.join(args.data_root, args.arm)
    train_dir, valid_dir = os.path.join(root, "train"), os.path.join(root, "valid")

    full = VecDataset(train_dir, is_train=True)
    print(f"=== {tag} ===  数据: {train_dir}  共 {len(full)} 条", flush=True)

    if os.path.isdir(valid_dir) and len(os.listdir(valid_dir)) > 0:
        gen = torch.Generator().manual_seed(args.seed)
        n_tr = int((1 - args.val_ratio) * len(full))
        tr_ds, va_ds = random_split(full, [n_tr, len(full) - n_tr], generator=gen)
        assert isinstance(va_ds.dataset, VecDataset)
        va_ds.dataset.is_train = False
        print(f"由 train 切分: Train={len(tr_ds)} Valid={len(va_ds)}", flush=True)
    else:
        tr_ds = full
        va_ds = None

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0) \
        if va_ds is not None else None

    model = build_model(args.model, args.conv_kernels, args.hidden, args.drop).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_par:,}  conv_kernels={args.conv_kernels} hidden={args.hidden} "
          f"drop={args.drop} wd={args.weight_decay}", flush=True)

    labels = [lab for _, lab in full]
    cw = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=labels)
    cw = torch.tensor(cw, dtype=torch.float32).to(device)
    print(f"类权重: {cw.cpu().numpy().round(3).tolist()}", flush=True)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)

    best_vloss, best_state = float("inf"), None
    history = []
    for ep in range(args.epochs):
        model.train()
        tl = 0.0
        for x, y in tqdm(tr_loader, desc=f"E{ep+1}/{args.epochs}", ncols=80):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            tl += loss.item()
        tl /= max(len(tr_loader), 1)

        if va_loader is not None:
            model.eval()
            vl, n = 0.0, 0
            with torch.no_grad():
                for x, y in va_loader:
                    x, y = x.to(device), y.to(device)
                    vl += criterion(model(x), y).item() * y.size(0)
                    n += y.size(0)
            vl /= max(n, 1)
            va, _, _ = evaluate(model, va_loader, device)
            scheduler.step(vl)
            history.append([ep + 1, tl, vl, va])
            if vl < best_vloss:
                best_vloss = vl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"E{ep+1} TL={tl:.4f} VL={vl:.4f} ValAcc={va:.2f}%", flush=True)
        else:
            history.append([ep + 1, tl, float("nan"), float("nan")])

    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    weight_path = os.path.join(args.ckpt_dir, f"best_{tag}.pth")
    torch.save(model.state_dict(), weight_path)

    # ---- 训练集 / 验证集 指标 ----
    results = []
    tr_loader_eval = DataLoader(full, batch_size=args.batch_size, shuffle=False, num_workers=0)
    for name, loader in (("train", tr_loader_eval), ("valid", va_loader)):
        if loader is None:
            continue
        acc, cm, report = evaluate(model, loader, device)
        per_class = {c: round(100 * cm[i][i] / max(sum(cm[i]), 1), 4)
                     for i, c in enumerate(TYPE_MAP)}
        results.append({"set": name, "total": sum(sum(r) for r in cm), "acc": round(acc, 4),
                        "per_class_acc": per_class, "confusion_matrix": cm,
                        "classification_report": report})
        print(f"\n[{name}] acc={acc:.2f}%  逐类={per_class}", flush=True)

    # ---- 单张图推理耗时 (论文 III.E 的 XX ms) ----
    model.eval()
    x1 = torch.rand(1, 48, device=device)
    with torch.no_grad():
        for _ in range(20):
            model(x1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts = []
        for _ in range(100):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
    t = torch.tensor(ts)
    print(f"单张图耗时: {t.mean():.3f} ± {t.std():.3f} ms", flush=True)

    meta = {
        "model": tag, "arm": args.arm, "arch": args.model,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "weight": os.path.basename(weight_path),
        "params": n_par,
        "input": "48-dim 单通道 6x8 网格图 (论文 Eq.10/11)",
        "time_per_image_ms": round(t.mean().item(), 4),
        "time_std_ms": round(t.std().item(), 4),
        "config": vars(args),
        "results": results,
    }
    mp = os.path.join(args.out_dir, f"{tag}_eval_meta.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, f"{tag}_train_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_acc_%"])
        w.writerows(history)
    print(f"\n[OK] 权重: {weight_path}\n     指标: {mp}", flush=True)


if __name__ == "__main__":
    main()
