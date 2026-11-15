
import os
import sys
import json
import argparse
from collections import Counter

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_ROWS, GRID_COLS = 6, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "data", "cache", "stage1_g_cache_448_bal.pt"))
    ap.add_argument("--anno-root", default=os.path.join(ROOT, "data", "annotations"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "cache", "pnr_dataset.pt"))
    ap.add_argument("--out-stats", default=os.path.join(ROOT, "results", "stage2", "pnr_dataset_stats.json"))
    args = ap.parse_args()

    entries = torch.load(args.cache, weights_only=False)["entries"]
    print(f"G 缓存: {len(entries)} 条", flush=True)

    X, y, ids, classes, truths, splits = [], [], [], [], [], []
    per_class, skipped = {}, []

    for e in entries:
        if e["split"] not in ("train", "valid"):
            continue
        json_path = None
        for ct in ("grid", "horizontal", "vertical"):
            p = os.path.join(args.anno_root, ct, ct, f"{e['id']}.json")
            if os.path.exists(p):
                json_path = p
                break
        if json_path is None:
            skipped.append(f"{e['cls']}/{e['id']}")
            continue
        with open(json_path, encoding="utf-8") as f:
            cells = json.load(f)["grid"]
        truth = torch.zeros(GRID_ROWS, GRID_COLS, dtype=torch.int32)
        for c in cells:
            truth[c["r"]][c["c"]] = c["label"]

        pred = e["G"][0]
        noise = ((pred == 1) & (truth == 0)).float()
        X.append(e["G"]); y.append(noise.flatten())
        ids.append(e["id"]); classes.append(e["cls"])
        truths.append(truth.flatten()); splits.append(e["split"])

        st = per_class.setdefault(e["cls"], {"n": 0, "pred_crack": 0, "true_crack": 0,
                                             "noise": 0, "grid_correct": 0, "grid_total": 0})
        st["n"] += 1
        st["pred_crack"] += int((pred == 1).sum())
        st["true_crack"] += int((truth == 1).sum())
        st["noise"] += int((noise == 1).sum())
        st["grid_correct"] += int((pred == truth).sum())
        st["grid_total"] += 48

    X = torch.stack(X); y = torch.stack(y)
    torch.save({"X": X, "y": y, "truth": torch.stack(truths),
                "ids": ids, "classes": classes, "splits": splits}, args.out)

    stats = {
        "n_images": len(ids),
        "split_counts": dict(Counter(splits)),
        "skipped_no_json": skipped,
        "total_grids": int(y.numel()),
        "noise_grids": int(y.sum()),
        "noise_rate": round(float(y.mean()), 6),
        "stage1_grid_acc": round(100 * sum(s["grid_correct"] for s in per_class.values())
                                 / sum(s["grid_total"] for s in per_class.values()), 4),
        "per_class": per_class,
    }
    with open(args.out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"PNR 数据集: {len(ids)} 张图, 噪声格 {int(y.sum())}/{int(y.numel())} "
          f"({100 * float(y.mean()):.2f}%)", flush=True)
    print(f"split 分布: {dict(Counter(splits))}", flush=True)
    print(f"Stage1 网格级精度(相对真值): {stats['stage1_grid_acc']}%", flush=True)
    for cls, s in per_class.items():
        print(f"  {cls}: {s['n']}图  预测裂格={s['pred_crack']}  真值裂格={s['true_crack']}  "
              f"噪声格={s['noise']}  网格精度={100 * s['grid_correct'] / s['grid_total']:.2f}%",
              flush=True)
    if skipped:
        print(f"无 JSON 真值跳过 {len(skipped)} 张: {skipped}", flush=True)
    print(f"已保存: {args.out} / {args.out_stats}", flush=True)


if __name__ == "__main__":
    main()
