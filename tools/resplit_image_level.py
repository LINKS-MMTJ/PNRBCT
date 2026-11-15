
import os
import sys
import json
import glob
import random
import argparse
from collections import defaultdict

from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CARETR_Project")

DATA_ROOT = os.path.join(ROOT, "data", "images")
CLASSES = ["grid", "horizontal", "vertical"]
GRID_ROWS, GRID_COLS = 6, 8
GRID_SIZE = 448                      # override with --grid-size
IMEXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")

DEFAULT_OUT_ROOT = os.path.join(ROOT, "data", "grids_448")
DEFAULT_MANIFEST = os.path.join(ROOT, "data", "split_v2_manifest.json")


def discover(data_root=DATA_ROOT):
    """收集有网格标注的训练图: (cls, image_id, image_path, json_path)"""
    items, missing = [], []
    for ct in CLASSES:
        for p in glob.glob(os.path.join(data_root, ct, "annotated", "*")):
            if not p.endswith(IMEXTS):
                continue
            iid = os.path.splitext(os.path.basename(p))[0]
            # 与打标工具一致: 三个类的 json 目录里都找一遍(个别 json 曾被误存到其他类目录)
            jp = None
            for jct in CLASSES:
                cand = os.path.join(data_root, jct, jct, f"{iid}.json")
                if os.path.exists(cand):
                    jp = cand
                    break
            if jp is None:
                missing.append(f"{ct}/{iid}")
                continue
            items.append((ct, iid, p, jp))
    return items, missing


def read_labels(json_path):
    """读取 6x8 网格标注 -> list[list[int]]"""
    with open(json_path, encoding="utf-8") as f:
        cells = json.load(f)["grid"]
    lab = [[0] * GRID_COLS for _ in range(GRID_ROWS)]
    for c in cells:
        lab[c["r"]][c["c"]] = c["label"]
    return lab


def stratified_split(items, ratios, seed=42):
    """按类别分层切分为 train/valid/test"""
    by_cls = defaultdict(list)
    for it in items:
        by_cls[it[0]].append(it)

    rng = random.Random(seed)
    out = {"train": [], "valid": [], "test": []}
    for ct in CLASSES:
        pool = sorted(by_cls[ct], key=lambda x: x[1])
        rng.shuffle(pool)
        n = len(pool)
        n_tr = int(round(n * ratios["train"]))
        n_va = int(round(n * ratios["valid"]))
        out["train"] += pool[:n_tr]
        out["valid"] += pool[n_tr:n_tr + n_va]
        out["test"] += pool[n_tr + n_va:]
    return out


def crop_cells(img, grid_size=GRID_SIZE):
    """与打标工具一致: 左上对齐, 丢弃不足一格的余数, 返回 48 个 PIL 图块"""
    W, H = img.size
    new_W, new_H = (W // GRID_COLS) * GRID_COLS, (H // GRID_ROWS) * GRID_ROWS
    img = img.crop((0, 0, new_W, new_H))
    pw, ph = new_W // GRID_COLS, new_H // GRID_ROWS
    return [img.crop((c * pw, r * ph, (c + 1) * pw, (r + 1) * ph))
            for r in range(GRID_ROWS) for c in range(GRID_COLS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT, help="源图根目录")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--valid", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid-size", type=int, default=GRID_SIZE,
                    help="output grid size in pixels (default: 448)")
    ap.add_argument("--materialize", action="store_true",
                    help="write the grid PNGs to out_root")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    args = ap.parse_args()

    ratios = {"train": args.train, "valid": args.valid, "test": args.test}
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        sys.exit(f"[ERR] 比例之和必须为 1, 当前={sum(ratios.values())}")

    items, missing = discover(args.data_root)
    print(f"有网格标注的训练图: {len(items)} 张")
    if missing:
        print(f"  无标注跳过: {len(missing)} 张 {missing}")

    # 预读标签, 供统计与落盘复用
    labels = {iid: read_labels(jp) for _, iid, _, jp in items}

    split = stratified_split(items, ratios, seed=args.seed)

    # ---- 校验: 三套图片互不重叠 ----
    ids = {s: set(iid for _, iid, _, _ in v) for s, v in split.items()}
    assert len(ids["train"] & ids["valid"]) == 0, "train/valid 重叠"
    assert len(ids["train"] & ids["test"]) == 0, "train/test 重叠"
    assert len(ids["valid"] & ids["test"]) == 0, "valid/test 重叠"
    print("\n[OK] 图片级互斥校验通过: train/valid/test 无任何重叠")

    # ---- 统计 ----
    print("\n" + "=" * 74)
    print(f"{'split':>7} | {'images':>7} | {'T(crack)':>9} | {'F(non)':>8} | "
          f"{'T:F':>7} | {'1:1 平衡后':>11}")
    print("-" * 74)
    stats = {}
    for s in ["train", "valid", "test"]:
        T = F = 0
        for _, iid, _, _ in split[s]:
            for row in labels[iid]:
                for v in row:
                    if v == 1:
                        T += 1
                    else:
                        F += 1
        ratio = F / T if T else 0
        stats[s] = {"images": len(split[s]), "T": T, "F": F,
                    "T_F_ratio": round(ratio, 3), "balanced_total": 2 * min(T, F)}
        print(f"{s:>7} | {len(split[s]):>7} | {T:>9} | {F:>8} | "
              f"1:{ratio:>4.2f} | {2*min(T,F):>11}")
    print("=" * 74)
    print("注: '1:1 平衡后' = 把多数类下采样到与少数类相同后的网格总数")

    print("\n各类别的图片数分布:")
    for ct in CLASSES:
        row = [sum(1 for c, _, _, _ in split[s] if c == ct)
               for s in ["train", "valid", "test"]]
        print(f"  {ct:11s} train={row[0]:4d}  valid={row[1]:4d}  test={row[2]:4d}")

    # ---- save manifest (store a repository-relative source path when possible) ----
    try:
        src_rel = os.path.relpath(args.data_root, ROOT)
        if src_rel.startswith(".."):
            src_rel = args.data_root
    except ValueError:
        src_rel = args.data_root
    manifest = {
        "note": "Image-level split: train/valid/test are disjoint; the 150 e2e images are not part of the split.",
        "source_root": src_rel,
        "seed": args.seed,
        "ratios": ratios,
        "grid": {"rows": GRID_ROWS, "cols": GRID_COLS, "cell_size": args.grid_size},
        "crop": "Top-left aligned; discard the remainder that does not fill a full cell.",
        "stats": stats,
        "missing_json": missing,
        "split": {s: sorted([{"cls": c, "id": i} for c, i, _, _ in v],
                            key=lambda x: (x["cls"], x["id"]))
                  for s, v in split.items()},
    }
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nmanifest 已保存: {args.manifest}")

    if not args.materialize:
        print("\n[提示] 仅完成切分与统计, 未写出图片。确认比例后加 --materialize 落盘。")
        return

    # ---- 落盘 ----
    print(f"\n落盘到: {args.out_root}", flush=True)
    n = 0
    for s in ["train", "valid", "test"]:
        print(f"  [{s}] 开始, {len(split[s])} 图 ...", flush=True)
        for k_img, (ct, iid, ipath, _) in enumerate(split[s], 1):
            if k_img % 50 == 0 or k_img == len(split[s]):
                print(f"    [{s}] {k_img}/{len(split[s])} 图, 累计 {n} 网格", flush=True)
            try:
                img = Image.open(ipath).convert("RGB")
            except Exception as e:
                print(f"  [WARN] 读取失败 {ipath}: {e}", flush=True)
                continue
            cells = crop_cells(img, args.grid_size)
            lab = labels[iid]
            for k, (r, c) in enumerate([(r, c) for r in range(GRID_ROWS)
                                        for c in range(GRID_COLS)]):
                tag = "T" if lab[r][c] == 1 else "F"
                d = os.path.join(args.out_root, s, tag)
                os.makedirs(d, exist_ok=True)
                cell = cells[k]
                if cell.size != (args.grid_size, args.grid_size):
                    cell = cell.resize((args.grid_size, args.grid_size))
                cell.save(os.path.join(d, f"{iid}_r{r}_c{c}_{tag}.png"))
                n += 1
        print(f"  {s}: {len(split[s])} 图 -> 累计 {n} 网格")
    print(f"\n[OK] 共写出 {n} 个网格图 -> {args.out_root}")


if __name__ == "__main__":
    main()
