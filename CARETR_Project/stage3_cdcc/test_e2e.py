
import os
import sys
import json
import time
import argparse
import datetime
from collections import Counter

import torch
from sklearn.metrics import confusion_matrix, classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.cdcc_model import CDCC

TYPE_MAP = {"grid": 0, "horizontal": 1, "vertical": 2}
INV = {v: k for k, v in TYPE_MAP.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["pnr", "nopnr"])
    ap.add_argument("--cache", default=os.path.join(ROOT, "data", "cache", "stage1_g_cache_448_bal.pt"))
    ap.add_argument("--pnr-model", default=os.path.join(ROOT, "checkpoints", "best_pnr.pth"))
    ap.add_argument("--cdcc-model", default=None)
    ap.add_argument("--arch", default="cdcc", choices=["cdcc"])
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "stage3"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cdcc_path = args.cdcc_model or os.path.join(args.out_dir, f"best_{args.arch}_{args.arm}.pth")
    print(f"=== 端到端 arm={args.arm} arch={args.arch} ===\nCDCC: {cdcc_path}", flush=True)

    cdcc = CDCC().to(device)
    cdcc.load_state_dict(torch.load(cdcc_path, map_location=device))
    cdcc.eval()

    pnr = None
    if args.arm == "pnr":
        from models.pnr_model import PositionAwareNoiseRemover
        pnr = PositionAwareNoiseRemover().to(device)
        pnr.load_state_dict(torch.load(args.pnr_model, map_location=device))
        pnr.eval()

    entries = [e for e in torch.load(args.cache, weights_only=False)["entries"]
               if e["split"] == "e2e_test"]
    print(f"测试图: {len(entries)} 张  {dict(Counter(e['cls'] for e in entries))}", flush=True)

    # 预热
    with torch.no_grad():
        _g = entries[0]["G"].to(device)
        for _ in range(10):
            _v = (pnr.clean(_g) if pnr is not None else _g[0]).flatten().unsqueeze(0)
            cdcc(_v)
    if device.type == "cuda":
        torch.cuda.synchronize()

    gts, preds, ts, removed_total = [], [], [], 0
    with torch.no_grad():
        for e in entries:
            G = e["G"].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if pnr is not None:
                grid, m = pnr.clean(G, return_mask=True)
                removed_total += int(m.sum())
            else:
                grid = G[0]
            vec = grid.flatten().unsqueeze(0)
            pred = cdcc(vec).argmax(1).item()
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
            gts.append(TYPE_MAP[e["cls"]])
            preds.append(pred)

    cm = confusion_matrix(gts, preds, labels=[0, 1, 2])
    acc = 100 * (cm[0][0] + cm[1][1] + cm[2][2]) / max(len(gts), 1)
    per_class = {c: round(100 * cm[i][i] / max(sum(cm[i]), 1), 4) for i, c in enumerate(TYPE_MAP)}
    report = classification_report(gts, preds, target_names=list(TYPE_MAP),
                                   digits=4, output_dict=True, zero_division=0)
    macro_f1 = report["macro avg"]["f1-score"]
    t = torch.tensor(ts)

    print(f"\n准确率 = {acc:.2f}%   宏平均 F1 = {macro_f1:.4f}", flush=True)
    print(f"逐类正确率: {per_class}", flush=True)
    print(f"混淆矩阵: {cm.tolist()}", flush=True)
    print(f"单张图耗时: {t.mean():.2f} ± {t.std():.2f} ms", flush=True)
    if pnr is not None:
        print(f"PNR 共移除网格: {removed_total}", flush=True)

    meta = {
        "arm": args.arm, "arch": args.arch,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cache": os.path.basename(args.cache),
        "cdcc_model": os.path.basename(cdcc_path),
        "pnr_model": os.path.basename(args.pnr_model) if pnr is not None else None,
        "n_images": len(entries),
        "acc": round(acc, 4), "macro_f1": round(macro_f1, 4),
        "per_class_acc": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "time_per_image_ms": round(t.mean().item(), 4),
        "time_std_ms": round(t.std().item(), 4),
        "pnr_removed_grids": removed_total if pnr is not None else None,
    }
    out = os.path.join(args.out_dir, f"e2e_{args.arch}_{args.arm}_eval_meta.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n元数据已保存: {out}", flush=True)


if __name__ == "__main__":
    main()
