
import os
import sys
import json
import glob
import datetime

import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CARETR_Project")
OUT_DIR = os.path.join(ROOT, "results", "verification")
GRIDS = os.path.join(ROOT, "data", "grids_448")
CACHE = os.path.join(ROOT, "data", "cache", "stage1_g_cache_448_bal.pt")
PNR_DATASET = os.path.join(ROOT, "data", "cache", "pnr_dataset.pt")
PNR_P = os.path.join(ROOT, "checkpoints", "best_pnr.pth")
MODEL = os.path.join(ROOT, "checkpoints", "caret_s0_448_bal_best.pth")
EVAL_SCRIPT = os.path.join(ROOT, "stage1_care_transformer", "eval_stage1.py")
os.makedirs(OUT_DIR, exist_ok=True)
PROB_LIMIT = 0.85
TYPE_MAP = {"grid": 0, "horizontal": 1, "vertical": 2}
NAMES = ["网状", "横向", "纵向"]

report = {"date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ==================================================================
print("=" * 78)
print("PART 1  数据划分核验 (从磁盘计数)")
print("=" * 78)
p1 = {}
for s in ["train", "valid", "test"]:
    f = len(glob.glob(f"{GRIDS}/{s}/F/*.png"))
    t = len(glob.glob(f"{GRIDS}/{s}/T/*.png"))
    p1[s] = {"F_on_disk": f, "T_on_disk": t, "total_on_disk": f + t,
             "T_used": t, "F_used": min(f, t), "total_used": 2 * min(f, t)}
    print(f"  {s:>6}: 磁盘 F={f:6d} T={t:5d} 合计={f+t:6d} | "
          f"1:1 实际使用 F={min(f,t):6d} T={t:5d} 合计={2*min(f,t):6d}")
report["part1_data"] = p1

# ==================================================================
print()
print("=" * 78)
print("PART 2  Stage1 网格检测 (III.C)")
print("=" * 78)
p2 = {}
for mode, tag in ((1, "1:1 平衡"),):
    print(f"\n  --- 口径: {tag} ---")
    print(f"  {'set':>6}{'网格':>8}{'准确率%':>10}{'非裂纹F1':>10}{'裂纹F1':>9}{'宏F1':>9}   混淆矩阵")
    meta_p = os.path.join(OUT_DIR, "verify_s1_bal.json")
    os.system(f"{sys.executable} {EVAL_SCRIPT} "
              f"--model {MODEL} "
              f"--data-root {GRIDS} --img-size 448 "
              f"--splits train,valid,test --balance-f-to-t {mode} "
              f"--out {meta_p} > /dev/null 2>&1")
    m = json.load(open(meta_p, encoding="utf-8"))
    p2[tag] = {}
    for r in m["results"]:
        rep = r["classification_report"]
        p2[tag][r["set"]] = {"n": r["total"], "acc": r["acc"],
                             "F1_noncrack": round(rep["F"]["f1-score"], 4),
                             "F1_crack": round(rep["T"]["f1-score"], 4),
                             "macro_f1": r["macro_f1"], "cm": r["confusion_matrix"]}
        print(f"  {r['set']:>6}{r['total']:>8}{r['acc']:>10.2f}{rep['F']['f1-score']:>10.4f}"
              f"{rep['T']['f1-score']:>9.4f}{r['macro_f1']:>9.4f}   {r['confusion_matrix']}")
report["part2_stage1"] = p2

# ==================================================================
print()
print("=" * 78)
print("PART 3  PNR 性能 (Table I)")
print("=" * 78)
print("""  定义:
    可操作集 = Stage1 判为裂纹的网格 (label == 1)
    门控集   = 可操作集 ∩ (Stage1 裂纹概率 < 0.85)   ← PNR 实际可决策的范围
    噪声格   = 真值为非裂纹 且 在可操作集内
    有效裂格 = 真值为裂纹   且 在可操作集内

  噪声检测 (正类 = 被移除的噪声格):
    TP = 门控集内被移除 且 真=噪声
    FP = 门控集内被移除 且 真=裂纹
    FN = 可操作集内未被移除 且 真=噪声   (含门控外的噪声格, 保守口径)
    P = TP/(TP+FP)   R = TP/(TP+FN)

  裂纹误检 (正类 = 被误删的真实裂纹格):
    TP = 门控集内被移除 且 真=裂纹
    FP = 门控集内被移除 且 真=噪声
    FN = 全部真实裂纹格中未被移除的
    P = TP/(TP+FP)   R = TP/(全部真实裂纹格)
    误删率 = TP / 全部真实裂纹格   (与 R 同分母)""")

X, y, truth, splits = None, None, None, None
d = torch.load(PNR_DATASET, weights_only=False)
X, y, truth, splits = d["X"], d["y"], d["truth"], d["splits"]

sys.path.insert(0, ROOT)
from models.pnr_model import PositionAwareNoiseRemover
pnr = PositionAwareNoiseRemover()
pnr.load_state_dict(torch.load(PNR_P, map_location="cpu"))
pnr.eval()

tp = fp = fn = 0
m_tp = m_fp = 0
crack_all = 0
noise_all = 0
act_all = 0
gate_all = 0
acc_b = acc_a = n_grid = 0

with torch.no_grad():
    for i in range(len(X)):
        G = X[i]
        t = truth[i].view(6, 8)
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
        m_fp += int((remove & (t == 0)).sum())

        cleaned = label.clone()
        cleaned[remove] = 0.0
        acc_b += int((label == t).sum())
        acc_a += int((cleaned == t).sum())
        n_grid += 48

np_, nr, nf = prf(tp, fp, fn)
mp_, mr, mf = prf(m_tp, m_fp, 0)   # 裂纹误检的 FN 分母改用全部真实裂纹格
mr = m_tp / crack_all if crack_all else 0.0
mf = 2 * mp_ * mr / (mp_ + mr) if (mp_ + mr) else 0.0

print(f"\n  样本规模:")
print(f"    全部网格            {n_grid}")
print(f"    全部真实裂纹格      {crack_all}")
print(f"    可操作集            {act_all}")
print(f"    门控集(PNR 可决策)  {gate_all}   ({100*gate_all/act_all:.1f}% 的可操作集)")
print(f"    可操作集内噪声格    {noise_all}   ({100*noise_all/act_all:.1f}%)")
print(f"    被移除网格总数      {tp+fp}")
print(f"\n  Table I:")
print(f"    {'':>16}{'Precision':>12}{'Recall':>12}{'F1':>12}")
print(f"    {'噪声网格检测':>14}{np_:>12.4f}{nr:>12.4f}{nf:>12.4f}")
print(f"    {'裂纹网格误检':>14}{mp_:>12.4f}{mr:>12.4f}{mf:>12.4f}")
print(f"\n    真实裂纹格误删率 = {m_tp}/{crack_all} = {100*mr:.4f}%")
print(f"    网格精度: {100*acc_b/n_grid:.2f}% -> {100*acc_a/n_grid:.2f}%")

report["part3_pnr"] = {
    "definitions": "见脚本 docstring; FN 用可操作集口径, 误删率分母=全部真实裂纹格",
    "n_grids": n_grid, "n_true_crack": crack_all, "n_actionable": act_all,
    "n_gate": gate_all, "n_noise_in_actionable": noise_all,
    "n_removed_total": tp + fp,
    "noise_precision": round(np_, 4), "noise_recall": round(nr, 4), "noise_f1": round(nf, 4),
    "crack_misdetection_precision": round(mp_, 4),
    "crack_misdetection_recall": round(mr, 4),
    "crack_misdetection_f1": round(mf, 4),
    "true_crack_removed_%": round(100 * mr, 4),
    "grid_acc_before_%": round(100 * acc_b / n_grid, 4),
    "grid_acc_after_%": round(100 * acc_a / n_grid, 4),
}

report_path = os.path.join(OUT_DIR, "verify_report.json")
json.dump(report, open(report_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\nSaved: {report_path}")
