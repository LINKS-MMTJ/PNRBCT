
import os
import sys
import csv
import json
import random
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.pnr_model import PositionAwareNoiseRemover


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

DATASET_PATH = os.environ.get("PNR_DATASET", os.path.join(ROOT, "data", "cache", "pnr_dataset.pt"))
OUT_DIR = os.environ.get("PNR_OUT_DIR", os.path.join(ROOT, "results", "stage2"))
SAVE_NAME = os.environ.get("PNR_CHECKPOINT", os.path.join(ROOT, "checkpoints", "best_pnr.pth"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 60
LR = 1e-3
PROB_LIMIT = 0.85   # 决策集门控: 仅对 prob<此值的裂纹格做保留/去除决策

# ---- 决策集 1:1 平衡 (以噪声格数为准, 下采样"有效裂纹格") ----
BALANCE = int(os.environ.get("PNR_BALANCE", "1"))   # 1=平衡(默认) 0=用原始决策集
SEED = 42

data = torch.load(DATASET_PATH, weights_only=False)
X, y, truth = data["X"], data["y"], data["truth"]   # y = 噪声标签(1=噪声)
print(f"数据集: {X.shape[0]} 张图, 噪声格 {int(y.sum())}/{int(y.numel())}", flush=True)

act_low = ((X[:, 0].view(-1, 48) == 1) & (X[:, 1].view(-1, 48) < PROB_LIMIT))
n_noise_low = int((act_low & y.view(-1, 48).bool()).sum())
print(f"低置信决策集(prob<{PROB_LIMIT}): {int(act_low.sum())} 格, 其中噪声 {n_noise_low} "
      f"({100 * n_noise_low / max(int(act_low.sum()), 1):.1f}%), 覆盖全部噪声的 "
      f"{100 * n_noise_low / max(int(y.sum()), 1):.0f}%", flush=True)

full = TensorDataset(X, y, truth)
if "splits" in data:
    # 沿用图片级 manifest 划分(train/valid), 与 Stage1 用同一套划分, 不再随机切分
    from torch.utils.data import Subset
    is_tr = torch.tensor([s == "train" for s in data["splits"]])
    tr_idx = torch.nonzero(is_tr).flatten().tolist()
    va_idx = torch.nonzero(~is_tr).flatten().tolist()
    train_ds = Subset(full, tr_idx)
    val_ds = Subset(full, va_idx) if va_idx else Subset(full, tr_idx)
    print(f"[划分] 使用 manifest: train={len(train_ds)}, valid={len(val_ds)}", flush=True)
else:
    generator = torch.Generator().manual_seed(42)
    n_train = int(0.8 * len(full))
    train_ds, val_ds = random_split(full, [n_train, len(full) - n_train], generator=generator)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

# ---- 决策集掩码: 门控内的裂纹格 ----
act_low_all = ((X[:, 0].view(-1, 48) == 1) & (X[:, 1].view(-1, 48) < PROB_LIMIT))
noise_all = y.view(-1, 48).bool()
keep = act_low_all.clone()          # 默认: 用全部决策集

if BALANCE:
    import random as _rnd
    all_splits = data.get("splits")
    keep = torch.zeros_like(act_low_all)
    for sp_name, rows in (("train", tr_idx), ("valid", va_idx)):
        if not rows:
            continue
        sel = act_low_all[rows]                    # (n_img, 48)
        tgt = noise_all[rows]
        i_noise = (sel & tgt).nonzero()
        i_valid = (sel & ~tgt).nonzero()
        n = min(len(i_noise), len(i_valid))
        g = torch.Generator().manual_seed(SEED)
        pn = torch.randperm(len(i_noise), generator=g)[:n]
        pv = torch.randperm(len(i_valid), generator=g)[:n]
        for rr, cc in i_noise[pn].tolist():
            keep[rows[rr], cc] = True
        for rr, cc in i_valid[pv].tolist():
            keep[rows[rr], cc] = True
        print(f"  [{sp_name}] 决策集 {int(sel.sum())} 格 -> 1:1 平衡后 {2*n} 格 "
              f"(噪声 {n} + 有效裂纹 {n})", flush=True)

full = TensorDataset(X, y, truth, keep)
# 重新构建 Subset(索引不变, 只是多了第 4 个张量)
if "splits" in data:
    train_ds = Subset(full, tr_idx)
    val_ds = Subset(full, va_idx) if va_idx else Subset(full, tr_idx)
else:
    train_ds = Subset(full, list(range(len(full))))
    val_ds = Subset(full, list(range(len(full))))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = PositionAwareNoiseRemover().to(DEVICE)
criterion = nn.CrossEntropyLoss()      # Eq.(8) 2 类
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-3)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6)


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def val_metrics(loader):
    """返回 val_loss, 噪声 P/R/F1, 裂纹误检 P/R/F1, 真实裂格误删率, 去噪前后网格精度"""
    model.eval()
    total_loss, n_act = 0.0, 0
    tp = fp = fn = 0                       # 噪声检测
    m_tp = m_fp = m_fn = 0                 # 裂纹格被误判为噪声
    crack_total = crack_removed = 0
    acc_before = acc_after = n_grids = 0
    with torch.no_grad():
        for xb, yb, tb, kb in loader:
            xb, yb, tb, kb = xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE), kb.to(DEVICE)
            tb = tb.view(-1, 6, 8)
            label = xb[:, 0]
            G1 = xb[:, 1]
            k = model(xb)                                   # (B, 48, 2)
            sel = kb.view(-1).bool()                        # 平衡掩码(仅用于损失)
            if sel.any():
                target = yb.view(-1)[sel].long()
                total_loss += criterion(k.view(-1, 2)[sel], target).item()
                n_act += int(sel.sum())

            # 门控集(完整, 与推理一致) —— 指标在这个集合上算
            act_low = ((label.view(-1, 48) == 1) & (G1.view(-1, 48) < PROB_LIMIT)).view(-1)
            pred_noise = (k.argmax(-1) == 1)                # Eq.(9)
            remove = (act_low.view(-1, 6, 8) & pred_noise.view(-1, 6, 8))
            cleaned = label.clone()
            for i in range(xb.size(0)):
                cleaned[i, remove[i]] = 0.0

            # 噪声检测 P/R/F1 (决策集口径)
            pn = remove.view(-1)
            gt_noise = yb.view(-1).bool() & act_low
            tp += int((pn & gt_noise).sum())
            fp += int((pn & ~gt_noise).sum())
            fn += int((~pn & gt_noise).sum())

            # 裂纹网格误检 P/R/F1: 正类 = 被误当作噪声移除的真实裂纹格
            gt_crack = (tb == 1).view(-1) & act_low
            m_tp += int((pn & gt_crack).sum())
            m_fp += int((pn & ~gt_crack).sum())
            m_fn += int((~pn & gt_crack).sum())

            crack_total += int((tb == 1).sum())             # 分母 = 全部真实裂纹格
            crack_removed += int((pn & (tb.view(-1) == 1)).sum())
            acc_before += int((label.view(-1, 6, 8) == tb).sum())
            acc_after += int((cleaned == tb).sum())
            n_grids += tb.numel()

    np_, nr, nf = _prf(tp, fp, fn)
    mp_, mr, mf = _prf(m_tp, m_fp, m_fn)
    return (total_loss / max(n_act, 1), np_, nr, nf, mp_, mr, mf,
            100 * crack_removed / max(crack_total, 1),
            100 * acc_before / n_grids, 100 * acc_after / n_grids)


best_acc_after, best_state, best_metrics = -1.0, None, None
early_stop_counter, patience = 0, 15
history = []

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss, n_act = 0.0, 0
    for xb, yb, tb, kb in tqdm(train_loader, desc=f"E{epoch+1} train", ncols=80):
        xb, yb, tb, kb = xb.to(DEVICE), yb.to(DEVICE), tb.to(DEVICE), kb.to(DEVICE)
        k = model(xb)
        sel = kb.view(-1).bool()
        if not sel.any():
            continue
        loss = criterion(k.view(-1, 2)[sel], yb.view(-1)[sel].long())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
        train_loss += loss.item()
        n_act += int(act_low.sum())
    train_loss /= max(n_act, 1)

    vloss, vp, vr, vf1, mp, mr, mf1, vrm, vacc_b, vacc_a = val_metrics(val_loader)
    scheduler.step(vloss)
    print(f"E{epoch+1:02d} TL={train_loss:.4f} VL={vloss:.4f} | "
          f"噪声 P={vp:.3f} R={vr:.3f} F1={vf1:.3f} | "
          f"裂格误检 P={mp:.3f} R={mr:.3f} F1={mf1:.3f} | 误删={vrm:.2f}% | "
          f"网格精度 {vacc_b:.2f}%→{vacc_a:.2f}%", flush=True)
    history.append([epoch + 1, train_loss, vloss, vp, vr, vf1, mp, mr, mf1, vrm,
                    vacc_b, vacc_a])

    if vacc_a > best_acc_after:
        best_acc_after = vacc_a
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_metrics = (vp, vr, vf1, mp, mr, mf1, vrm, vacc_b, vacc_a)
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("[STOP] Early stop", flush=True)
            break

model.load_state_dict(best_state)
os.makedirs(os.path.dirname(SAVE_NAME), exist_ok=True)
torch.save(model.state_dict(), SAVE_NAME)

model.to(DEVICE).eval()
full_loader = DataLoader(full, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
_, fp_, fr_, ff1, fmp, fmr, fmf1, frm, facc_b, facc_a = val_metrics(full_loader)
bp, br, bf1, bmp, bmr, bmf1, brm, bb, ba = best_metrics

meta = {
    "model": os.path.basename(SAVE_NAME),
    "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "train_n": len(train_ds), "val_n": len(val_ds),
    "loss": "CrossEntropyLoss 2 类 (q=0 有效裂纹格 / q=1 噪声格), 仅在低置信决策集上计算",
    "prob_limit": PROB_LIMIT,
    "best_epoch_val": {
        "noise_precision": round(bp, 4), "noise_recall": round(br, 4), "noise_f1": round(bf1, 4),
        "crack_misdetection_precision": round(bmp, 4),
        "crack_misdetection_recall": round(bmr, 4),
        "crack_misdetection_f1": round(bmf1, 4),
        "true_crack_removed_%": round(brm, 4),
        "grid_acc_before_%": round(bb, 4), "grid_acc_after_%": round(ba, 4),
    },
    "full_dataset": {
        "noise_precision": round(fp_, 4), "noise_recall": round(fr_, 4), "noise_f1": round(ff1, 4),
        "crack_misdetection_precision": round(fmp, 4),
        "crack_misdetection_recall": round(fmr, 4),
        "crack_misdetection_f1": round(fmf1, 4),
        "true_crack_removed_%": round(frm, 4),
        "grid_acc_before_%": round(facc_b, 4), "grid_acc_after_%": round(facc_a, 4),
    },
    "balance_decision_set": bool(BALANCE),
    "note": "PNR(8×2×2conv+ReLU + 2 类 MLP 头) 训练目标 = 噪声标签 (Stage1 预测为裂纹 且 真值为非裂纹); "
            "决策集 = 低置信裂纹格(prob<0.85), 高置信格直接保留; dropout 0.3 + wd 5e-3; "
            "最优 epoch 按去噪后网格精度选取",
}
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "pnr_eval_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

if history:
    with open(os.path.join(OUT_DIR, "pnr_train_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "noise_precision", "noise_recall",
                    "noise_f1", "crack_mis_p", "crack_mis_r", "crack_mis_f1",
                    "true_crack_removed_%", "grid_acc_before_%", "grid_acc_after_%"])
        w.writerows(history)

print(f"\n[OK] {SAVE_NAME} (best val grid acc after={best_acc_after:.2f}%)", flush=True)
print(f"[全数据集] 噪声 P={fp_:.3f} R={fr_:.3f} F1={ff1:.3f} | "
      f"裂格误检 P={fmp:.3f} R={fmr:.3f} F1={fmf1:.3f} | 误删={frm:.2f}% | "
      f"网格精度 {facc_b:.2f}% → {facc_a:.2f}%", flush=True)
