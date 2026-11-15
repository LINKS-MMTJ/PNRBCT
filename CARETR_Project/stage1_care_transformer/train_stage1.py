
import os
import sys
import csv
import random
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as v2
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from timm import create_model
from sklearn.metrics import f1_score
from tqdm import tqdm
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import models.care_model  # noqa: F401  (注册 CARETrans_S0)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
   
    def __init__(self, patience=80, save_path='best.pth'):
        self.patience = patience
        self.counter = 0
        self.best_score = None            # 宏平均 F1, 越大越好
        self.early_stop = False
        self.best_val_loss = float('inf')
        self.save_path = save_path
        self.best_epoch = 0

    def __call__(self, macro_f1, val_loss, model, epoch):
        import math
        if math.isnan(macro_f1):
            self.counter += 1
            print(f'  [WARN] NaN macro-F1! {self.counter}/{self.patience}', flush=True)
            if self.counter >= self.patience:
                self.early_stop = True
            return
        if self.best_score is None or macro_f1 > self.best_score + 1e-6:
            self.best_score = macro_f1
            self.save_checkpoint(macro_f1, val_loss, model, epoch)
            self.counter = 0
        else:
            self.counter += 1
            print(f'  EarlyStopping: {self.counter}/{self.patience}', flush=True)
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, macro_f1, val_loss, model, epoch):
        print(f'  Macro-F1 improved -> {macro_f1:.4f} (val_loss={val_loss:.4f}). Saving...',
              flush=True)
        torch.save(model.state_dict(), self.save_path)
        self.best_val_loss = val_loss
        self.best_epoch = epoch + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data", "grids_448"),
                    help="448x448 grid dataset (train/valid/test, each with F/ and T/)")
    ap.add_argument("--save-dir", default=os.path.join(ROOT, "checkpoints"),
                    help="directory for the best checkpoint")
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results", "stage1"),
                    help="directory for training metrics and curves")
    ap.add_argument("--tag", default="caret_s0_448_bal")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--weight-decay", type=float, default=0.03)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--grad-checkpoint", type=int, default=1,
                    help="1=enable gradient checkpointing (saves memory, costs time)")
    ap.add_argument("--pin-memory", type=int, default=0, help="1=on 0=off")
    ap.add_argument("--persistent-workers", type=int, default=1, help="1=on 0=off")
    ap.add_argument("--prefetch-factor", type=int, default=1)
    ap.add_argument("--img-size", type=int, default=448,
                    help="input grid size; must match the dataset resolution")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balance-f-to-t", type=int, default=1,
                    help="1=训练时把 F 下采样到与 T 相同(1:1), 文件不删除 (默认); 0=用逆频权重")
    ap.add_argument("--use-class-weight", type=int, default=0,
                    help="1=损失用逆频类权重; 0=不加(默认, 与 1:1 采样配套)")
    args = ap.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.tag}_best.pth")
    device = torch.device("cuda")

    print(f"=== Stage 1: Binary CARE Transformer ===  data: {args.data_root}", flush=True)
    print(f"batch={args.batch_size}  lr={args.lr}  epochs={args.epochs}  "
          f"warmup={args.warmup_epochs}  wd={args.weight_decay}  img={args.img_size}", flush=True)

    transform_train = v2.Compose([
        v2.Resize((args.img_size, args.img_size), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=10),
        v2.ColorJitter(brightness=0.15, contrast=0.15),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.450, 0.400], [0.229, 0.225, 0.225]),
    ])
    transform_val = v2.Compose([
        v2.Resize((args.img_size, args.img_size), antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.450, 0.400], [0.229, 0.225, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(args.data_root, "train"), transform=transform_train)
    valid_ds = datasets.ImageFolder(os.path.join(args.data_root, "valid"), transform=transform_val)
    print(f"Train: {len(train_ds)}   Valid: {len(valid_ds)}", flush=True)

    # ---- 可选: 按 1:1 下采样非裂纹(文件不删除, 只是不参与训练) ----
    if args.balance_f_to_t:
        from torch.utils.data import Subset
        tg = train_ds.targets
        idx_f = [i for i, t in enumerate(tg) if t == 0]
        idx_t = [i for i, t in enumerate(tg) if t == 1]
        rs = random.Random(args.seed)
        rs.shuffle(idx_f)
        n_keep = min(len(idx_f), len(idx_t))
        idx_f = sorted(idx_f[:n_keep])
        sel = sorted(idx_f + idx_t)
        print(f"[1:1 下采样] F {len(idx_f)} + T {len(idx_t)} = {len(sel)} "
              f"(原 F 共 {len([t for t in tg if t == 0])} 个, 未删除, 仅不参与训练)", flush=True)
        train_ds = Subset(train_ds, sel)

    dl_kw = dict(num_workers=args.num_workers,
                 pin_memory=bool(args.pin_memory),
                 persistent_workers=bool(args.persistent_workers) and args.num_workers > 0)
    if args.num_workers > 0:
        dl_kw["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **dl_kw)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, **dl_kw)
    print(f"DataLoader: workers={args.num_workers} pin_memory={bool(args.pin_memory)} "
          f"persistent={dl_kw['persistent_workers']} prefetch={args.prefetch_factor}", flush=True)

    model = create_model('CARETrans_S0', pretrained=False, num_classes=2).to(device)
    if hasattr(model, 'head') and hasattr(model.head, 'fc'):
        model.head.fc = nn.Sequential(nn.Dropout(0.45), model.head.fc)
    if args.grad_checkpoint:
        model.set_grad_checkpointing(True)
        print("梯度检查点: ON (用重计算换显存)", flush=True)

    # 逆频类权重选项: 不重采样, 用权重补偿类别不平衡
    counter = Counter(train_ds.dataset.targets if hasattr(train_ds, "dataset") else train_ds.targets)
    n_used = len(train_ds)
    print(f"训练用样本 {n_used}  (F={counter[0]}, T={counter[1]})", flush=True)
    if args.use_class_weight and not args.balance_f_to_t:
        w = torch.tensor([1.0 / counter[0], 1.0 / counter[1]], dtype=torch.float32).to(device)
        w = w / w.sum()
        print(f"类权重 w=[{w[0]:.4f}, {w[1]:.4f}]  比值 wT/wF={w[1]/w[0]:.3f}", flush=True)
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=args.label_smoothing)
    else:
        print("类权重: 不加(1:1 已平衡)", flush=True)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                            weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    print(f"Steps/epoch: {steps_per_epoch}  Total: {total_steps}  Warmup: {warmup_steps}", flush=True)

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    scaler = torch.amp.GradScaler('cuda')
    early = EarlyStopping(patience=args.patience, save_path=save_path)
    best_acc = 0.0
    best_macro_f1 = 0.0
    history = []
    csv_path = os.path.join(args.results_dir, f"{args.tag}_train_metrics.csv")
    t_start = datetime.datetime.now()

    for epoch in range(args.epochs):
        model.train()
        running_loss, total = 0.0, 0
        pbar = tqdm(train_loader, desc=f"E{epoch+1}/{args.epochs}", ncols=90)
        for inputs, lbls in pbar:
            inputs, lbls = inputs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, lbls)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = running_loss / total

        model.eval()
        val_loss, correct, num_val = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, lbls in valid_loader:
                inputs, lbls = inputs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = criterion(outputs, lbls)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(1)
                correct += preds.eq(lbls).sum().item()
                num_val += inputs.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(lbls.cpu().tolist())
        val_loss /= num_val
        acc = 100.0 * correct / num_val
        crack_f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        best_acc = max(best_acc, acc)
        best_macro_f1 = max(best_macro_f1, macro_f1)
        print(f"E{epoch+1} | TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
              f"ValAcc={acc:.2f}% | crackF1={crack_f1:.4f} | macroF1={macro_f1:.4f} | "
              f"Best macroF1={best_macro_f1:.4f}", flush=True)
        history.append([epoch + 1, train_loss, val_loss, acc, crack_f1, macro_f1, best_macro_f1])
        early(macro_f1, val_loss, model, epoch)

        with open(csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["epoch", "train_loss", "val_loss", "val_acc_%",
                         "crack_f1", "macro_f1", "best_macro_f1"])
            wr.writerows(history)

        if early.early_stop:
            print(f"[STOP] Early stop at epoch {epoch+1}", flush=True)
            break

    el = (datetime.datetime.now() - t_start).total_seconds() / 60
    print(f"\n[DONE] 最优 epoch={early.best_epoch}  Best macroF1={early.best_score:.4f}  "
          f"Best ValAcc={best_acc:.2f}%  (ValLoss={early.best_val_loss:.4f})  "
          f"耗时 {el:.1f} min", flush=True)
    print(f"       权重: {save_path}", flush=True)

    # 训练曲线
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = [h[0] for h in history]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
        ax1.plot(ep, [h[1] for h in history], color="#2a78d6", lw=2, label="TrainLoss")
        ax1.plot(ep, [h[2] for h in history], color="#eb6834", lw=2, label="ValLoss")
        ax1.set_ylabel("Loss"); ax1.legend(frameon=False)
        ax1.set_title(f"{args.tag} (image-level split)", loc="left")
        # history 列: [epoch, train_loss, val_loss, val_acc, crack_f1, macro_f1, best_macro_f1]
        ax2.plot(ep, [h[3] for h in history], color="#12855c", lw=2,
                 label="Validation accuracy")
        ax2.plot(ep, [h[5] for h in history], color="#2a78d6", lw=2,
                 label="Validation macro-F1")
        ax2.plot(ep, [h[6] for h in history], color="#4a3aa7", lw=2, ls="--",
                 label="Best macro-F1")
        ax2.set_ylabel("Accuracy (%)"); ax2.set_xlabel("Epoch"); ax2.legend(frameon=False)
        for ax in (ax1, ax2):
            ax.set_axisbelow(True); ax.grid(axis="y", color="#e1e0d9", lw=0.8)
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.margins(x=0.01)
        plt.savefig(os.path.join(args.results_dir, f"{args.tag}_curve.png"), dpi=200,
                    bbox_inches="tight")
        print("[OK] 曲线已保存", flush=True)
    except Exception as e:
        print(f"[WARN] 画曲线失败: {e}", flush=True)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
