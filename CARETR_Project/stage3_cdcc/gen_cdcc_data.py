
import os
import sys
import argparse
from collections import Counter

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TYPE_MAP = {"grid": 0, "horizontal": 1, "vertical": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "data", "cache", "stage1_g_cache_448_bal.pt"))
    ap.add_argument("--arm", required=True, choices=["pnr", "nopnr"])
    ap.add_argument("--pnr-model", default=os.path.join(ROOT, "checkpoints", "best_pnr.pth"))
    ap.add_argument("--out-root", default=os.path.join(ROOT, "data", "cdcc_data"))
    args = ap.parse_args()

    entries = torch.load(args.cache, weights_only=False)["entries"]
    print(f"G 缓存: {len(entries)} 条", flush=True)

    pnr = None
    if args.arm == "pnr":
        from models.pnr_model import PositionAwareNoiseRemover
        pnr = PositionAwareNoiseRemover()
        pnr.load_state_dict(torch.load(args.pnr_model, map_location="cpu"))
        pnr.eval()
        print(f"PNR: {args.pnr_model}", flush=True)

    out_root = os.path.join(args.out_root, args.arm)
    counts = Counter()
    removed = 0
    for e in entries:
        split = e["split"]
        split = "e2e" if split == "e2e_test" else split   # end-to-end test split
        if split not in ("train", "valid", "e2e"):
            continue                                       # Stage1 自己的 test 不用于 CDCC
        G = e["G"]
        if pnr is not None:
            grid, mask = pnr.clean(G, return_mask=True)    # (6,8) 单通道
            removed += int(mask.sum())
        else:
            grid = G[0]                                    # 检测标签通道
        vec = grid.flatten().float()                       # 48 维
        d = os.path.join(out_root, split, e["cls"])
        os.makedirs(d, exist_ok=True)
        torch.save({"vector": vec, "label": TYPE_MAP[e["cls"]]},
                   os.path.join(d, f"{e['id']}.pt"))
        counts[split] += 1

    print(f"\n[OK] arm={args.arm} -> {out_root}", flush=True)
    for s in ("train", "valid", "e2e"):
        per = {ct: len(os.listdir(os.path.join(out_root, s, ct)))
               for ct in TYPE_MAP if os.path.isdir(os.path.join(out_root, s, ct))}
        print(f"  {s}: {counts[s]} 条  {per}", flush=True)
    if pnr is not None:
        print(f"  PNR 共移除网格: {removed}", flush=True)


if __name__ == "__main__":
    main()
