
import os
import sys
import json

import torch

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CARETR_Project")
sys.path.insert(0, ROOT)
from models.pnr_model import PositionAwareNoiseRemover

CACHE = os.path.join(ROOT, "data", "cache", "stage1_g_cache_448_bal.pt")
PNR_P = os.path.join(ROOT, "checkpoints", "best_pnr.pth")
ANNO = os.path.join(ROOT, "data", "annotations")
OUT_DIR = os.path.join(ROOT, "results", "stage2")
PROB_LIMIT = 0.85


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "test"
    entries = [e for e in torch.load(CACHE, weights_only=False)["entries"]
               if e["split"] == target]
    n_img = len(entries)

    pnr = PositionAwareNoiseRemover()
    pnr.load_state_dict(torch.load(PNR_P, map_location="cpu"))
    pnr.eval()

    tp = fp = fn = 0
    m_tp = 0
    crack_all = noise_all = act_all = gate_all = 0
    acc_b = acc_a = n_grid = 0
    skipped = 0

    with torch.no_grad():
        for e in entries:
            jp = None
            for ct in ("grid", "horizontal", "vertical"):
                p = os.path.join(ANNO, ct, ct, f"{e['id']}.json")
                if os.path.exists(p):
                    jp = p
                    break
            if jp is None:
                skipped += 1
                continue
            t = torch.zeros(6, 8, dtype=torch.int32)
            for c in json.load(open(jp, encoding="utf-8"))["grid"]:
                t[c["r"]][c["c"]] = c["label"]

            G = e["G"]
            label, prob = G[0], G[1]
            _, remove = pnr.clean(G, return_mask=True)

            actionable = (label == 1)
            gate = actionable & (prob < PROB_LIMIT)
            gt_noise = (t == 0)

            act_all += int(actionable.sum())
            gate_all += int(gate.sum())
            noise_all += int((gt_noise & actionable).sum())
            crack_all += int((t == 1).sum())

            tp += int((remove & gt_noise).sum())
            fp += int((remove & ~gt_noise).sum())
            fn += int(((~remove) & gt_noise & actionable).sum())
            m_tp += int((remove & (t == 1)).sum())

            cleaned = label.clone()
            cleaned[remove] = 0.0
            acc_b += int((label == t).sum())
            acc_a += int((cleaned == t).sum())
            n_grid += 48

    np_, nr, nf = prf(tp, fp, fn)
    mp_ = m_tp / (tp + fp) if (tp + fp) else 0.0     # 与被移除总数之比 (=1-噪声P)
    mr = m_tp / crack_all if crack_all else 0.0      # 分母 = 全部真实裂纹格
    mf = 2 * mp_ * mr / (mp_ + mr) if (mp_ + mr) else 0.0

    print(f"===== split = {target} =====")
    print(f"  图像 {n_img} 张" + (f" (跳过 {skipped} 张无标注)" if skipped else ""))
    print(f"  网格总数            {n_grid}")
    print(f"  全部真实裂纹格      {crack_all}")
    print(f"  可操作集            {act_all}   (Stage1 判为裂纹)")
    print(f"  门控集              {gate_all}   (prob<{PROB_LIMIT}, PNR 可决策)")
    print(f"  可操作集内噪声格    {noise_all}")
    print(f"  被移除网格总数      {tp + fp}")
    print()
    print(f"  {'TABLE I':<20}{'Precision':>11}{'Recall':>11}{'F1-measure':>13}")
    print(f"  {'Noise grids':<20}{np_:>11.4f}{nr:>11.4f}{nf:>13.4f}")
    print(f"  {'Crack misdetection':<20}{mp_:>11.4f}{mr:>11.4f}{mf:>13.4f}")
    print(f"  校验: 两列 Precision 之和 = {np_ + mp_:.4f}  (应为 1.0000)")
    print()
    print(f"  真实裂纹格误删率 = {m_tp}/{crack_all} = {100*mr:.4f}%")
    print(f"  网格精度: {100*acc_b/n_grid:.2f}% -> {100*acc_a/n_grid:.2f}%")

    out = {"split": target, "n_images": n_img, "n_grids": n_grid,
           "n_true_crack": crack_all, "n_actionable": act_all, "n_gate": gate_all,
           "n_noise_in_actionable": noise_all, "n_removed": tp + fp,
           "noise_precision": round(np_, 4), "noise_recall": round(nr, 4),
           "noise_f1": round(nf, 4),
           "crack_mis_precision": round(mp_, 4), "crack_mis_recall": round(mr, 4),
           "crack_mis_f1": round(mf, 4),
           "true_crack_removed_%": round(100 * mr, 4),
           "grid_acc_before_%": round(100 * acc_b / n_grid, 4),
           "grid_acc_after_%": round(100 * acc_a / n_grid, 4)}
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"table1_{target}.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
