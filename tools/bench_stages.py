"""三阶段耗时精确测量 (CUDA Event 计时, 多 batch size 扫描)

动机: 之前 batch=1 测出的"0.x ms/网格"几乎全是 GPU 内核启动开销,
      无法反映真实计算量。本脚本扫多个 batch, 分离"固定开销"与"稳态吞吐"。

用法: python bench_stages.py
产物: bench_stages_meta.json + bench_stages_curve.png
"""
import os
import json
import datetime

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CARETR_Project")
sys.path.insert(0, ROOT)
import models.care_model  # noqa
from models.pnr_model import PositionAwareNoiseRemover
from models.cdcc_model import CDCC

DEV = "cuda"
CKPT = os.path.join(ROOT, "checkpoints")
OUT_DIR = os.path.join(ROOT, "results", "timing")

SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#2a2a28", "#52514e",
                                          "#898781", "#e1e0d9", "#c3c2b7")
C_S1, C_S2, C_S3 = "#2a78d6", "#eb6834", "#12855c"


def bench_call(fn, reps=50, warmup=10):
    """返回单次调用(整批)的平均毫秒数, CUDA Event 计时"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def load_s1(p):
    m = care_model.CARENets(
        token_mixers=care_model.TokenMixer,
        depths=(2, 4, 8, 4), feat_dims=(24, 48, 96, 192),
        mem_dims=(8, 16, 32, 64), cls_dim=1024,
        ratios=((2, 2), (2, 4), (4, 4), (4, 4)), num_classes=2)
    if hasattr(m, 'head') and hasattr(m.head, 'fc'):
        m.head.fc = nn.Sequential(nn.Dropout(0.45), m.head.fc)
    m.load_state_dict(torch.load(p, map_location=DEV))
    return m.to(DEV).eval()


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    out = {"gpu": torch.cuda.get_device_name(0),
           "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "timing": "torch.cuda.Event, 含预热, 取多次平均", "stages": {}}

    # ---------------- Stage1 ----------------
    print("\n[Stage1] 448x448x3 网格图块")
    s1 = load_s1(os.path.join(CKPT, "caret_s0_448_bal_best.pth"))
    rows = []
    print(f"  {'batch':>6}{'整批ms':>10}{'每网格ms':>11}{'网格/秒':>10}")
    for b in (1, 8, 48, 128, 256):
        x = torch.randn(b, 3, 448, 448, device=DEV)
        with torch.no_grad():
            ms = bench_call(lambda: s1(x), reps=max(10, 200 // b), warmup=5)
        per = ms / b
        rows.append({"batch": b, "batch_ms": round(ms, 4), "per_grid_ms": round(per, 4),
                     "grids_per_s": round(1000 / per, 1)})
        print(f"  {b:>6}{ms:>10.3f}{per:>11.4f}{1000/per:>10.1f}")
    out["stages"]["stage1"] = {"input": "3x448x448 (单个网格图块)", "unit": "网格",
                               "curve": rows}
    del s1
    torch.cuda.empty_cache()

    # ---------------- Stage2 PNR ----------------
    print("\n[Stage2 PNR] 6x8x2 检测矩阵 (每张图一个)")
    pnr = PositionAwareNoiseRemover().to(DEV)
    pnr.load_state_dict(torch.load(os.path.join(CKPT, "best_pnr.pth"), map_location=DEV))
    pnr.eval()
    rows = []
    print(f"  {'batch(图)':>10}{'整批ms':>10}{'每图ms':>10}{'每网格ms':>11}")
    for b in (1, 8, 32, 128):
        x = torch.rand(b, 2, 6, 8, device=DEV)
        with torch.no_grad():
            ms = bench_call(lambda: pnr(x), reps=max(20, 400 // b), warmup=10)
        rows.append({"batch": b, "batch_ms": round(ms, 4), "per_image_ms": round(ms / b, 4),
                     "per_grid_ms": round(ms / b / 48, 5)})
        print(f"  {b:>10}{ms:>10.3f}{ms/b:>10.4f}{ms/b/48:>11.5f}")
    out["stages"]["stage2_pnr"] = {"input": "2x6x8 (一张图的检测矩阵)", "unit": "图",
                                   "curve": rows}
    del pnr
    torch.cuda.empty_cache()

    # ---------------- Stage3 CDCC ----------------
    print("\n[Stage3 CDCC] 48 维向量 (每张图一个)")
    cd = CDCC().to(DEV)
    cd.load_state_dict(torch.load(os.path.join(CKPT, "best_cdcc_pnr.pth"), map_location=DEV))
    cd.eval()
    rows = []
    print(f"  {'batch(图)':>10}{'整批ms':>10}{'每图ms':>10}{'每网格ms':>11}")
    for b in (1, 8, 32, 128):
        x = torch.rand(b, 48, device=DEV)
        with torch.no_grad():
            ms = bench_call(lambda: cd(x), reps=max(20, 400 // b), warmup=10)
        rows.append({"batch": b, "batch_ms": round(ms, 4), "per_image_ms": round(ms / b, 4),
                     "per_grid_ms": round(ms / b / 48, 5)})
        print(f"  {b:>10}{ms:>10.3f}{ms/b:>10.4f}{ms/b/48:>11.5f}")
    out["stages"]["stage3_cdcc"] = {"input": "48 (一张图的展平网格图)", "unit": "图",
                                    "curve": rows}
    del cd
    torch.cuda.empty_cache()

    # ---------------- 端到端 ----------------
    print("\n[端到端] 单张图 = 48 个网格 -> Stage1 -> PNR -> CDCC")
    s1 = load_s1(os.path.join(CKPT, "caret_s0_448_bal_best.pth"))
    pnr = PositionAwareNoiseRemover().to(DEV)
    pnr.load_state_dict(torch.load(os.path.join(CKPT, "best_pnr.pth"), map_location=DEV)); pnr.eval()
    cd = CDCC().to(DEV)
    cd.load_state_dict(torch.load(os.path.join(CKPT, "best_cdcc_pnr.pth"), map_location=DEV)); cd.eval()
    x = torch.randn(1, 3, 448, 448, device=DEV)
    G = torch.zeros(1, 2, 6, 8, device=DEV)
    G[0, 0] = (torch.rand(1, 6, 8, device=DEV) > 0.5).float()
    G[0, 1] = torch.rand(1, 6, 8, device=DEV)

    def one_image():
        with torch.no_grad():
            s1(x.repeat(48, 1, 1, 1))
            pnr(G)
            cd(G.flatten(1))

    ms = bench_call(one_image, reps=20, warmup=5)
    print(f"  单张图 48 网格全流程: {ms:.3f} ms/图")
    out["end2end"] = {"per_image_ms": round(ms, 4), "per_grid_ms": round(ms / 48, 4),
                      "note": "含 48 次 Stage1 前向(未批量化) + PNR + CDCC"}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "bench_stages_meta.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved: {os.path.join(OUT_DIR, 'bench_stages_meta.json')}")

    # ---------------- 曲线 ----------------
    plt.rcParams.update({"font.size": 10, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    specs = [("stage1", "Stage1 (per grid, ms)", C_S1, "per_grid_ms"),
             ("stage2_pnr", "PNR (per image, ms)", C_S2, "per_image_ms"),
             ("stage3_cdcc", "CDCC (per image, ms)", C_S3, "per_image_ms")]
    for ax, (key, title, col, field) in zip(axes, specs):
        c = out["stages"][key]["curve"]
        xs = [r["batch"] for r in c]
        ys = [r[field] for r in c]
        ax.plot(xs, ys, "o-", color=col, lw=2, ms=6)
        for xi, yi in zip(xs, ys):
            ax.annotate(f"{yi:.3f}", (xi, yi), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8, color=MUTED)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Batch size")
        ax.set_ylabel(title)
        ax.set_title(title.split(" (")[0], loc="left", fontsize=11, color=INK)
        ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, lw=0.8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.margins(x=0.15, y=0.22)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bench_stages_curve.png"), dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved: {os.path.join(OUT_DIR, 'bench_stages_curve.png')}")


if __name__ == "__main__":
    main()
