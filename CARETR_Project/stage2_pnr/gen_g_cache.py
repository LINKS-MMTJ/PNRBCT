
import os
import re
import sys
import json
import argparse
from collections import defaultdict, Counter
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from timm import create_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import models.care_model  # noqa: F401  # pyright: ignore[reportUnusedImport]

IMG_SIZE, GRID_ROWS, GRID_COLS = 448, 6, 8   # override with --img-size
CLASSES = ["grid", "horizontal", "vertical"]
PAT_ANNO = re.compile(r"^(.*)_r(\d+)_c(\d+)_[FT]\.png$")
PAT_E2E = re.compile(r"^(.*)_r(\d+)_c(\d+)\.png$")

def make_transform(size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.450, 0.400], [0.229, 0.225, 0.225]),
    ])


transform = make_transform()


def load_model(path, device):
    m: Any = create_model('CARETrans_S0', pretrained=False, num_classes=2)
    if hasattr(m, 'head') and hasattr(m.head, 'fc'):
        m.head.fc = nn.Sequential(nn.Dropout(0.45), m.head.fc)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()


def collect_anno(grids_root):
    """{split}/{F,T}/*.png -> {(split, id): {48 cells}}"""
    groups: defaultdict[tuple[str, str], list[Any]] = defaultdict(lambda: [None] * 48)
    for split in ("train", "valid", "test"):
        for tag in ("F", "T"):
            d = os.path.join(grids_root, split, tag)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                m = PAT_ANNO.match(f)
                if not m:
                    continue
                iid, r, c = m.group(1), int(m.group(2)), int(m.group(3))
                groups[(split, iid)][r * GRID_COLS + c] = os.path.join(d, f)
    return groups


def collect_e2e(e2e_root):
    """{cls}/*.png -> {(cls, id): {48 cells}}"""
    groups: defaultdict[tuple[str, str], list[Any]] = defaultdict(lambda: [None] * 48)
    for ct in CLASSES:
        d = os.path.join(e2e_root, ct)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            m = PAT_E2E.match(f)
            if not m:
                continue
            iid, r, c = m.group(1), int(m.group(2)), int(m.group(3))
            groups[(ct, iid)][r * GRID_COLS + c] = os.path.join(d, f)
    return groups


@torch.no_grad()
def run_groups(groups, model, device, batch_imgs, desc, key_info):
    """对每张图的 48 个网格做推理, 组装 G

    key_info(key) -> (split, cls)
    """
    keys, entries = list(groups.keys()), []
    pbar = tqdm(range(0, len(keys), batch_imgs), desc=f"{desc} 推理", ncols=90)
    for i in pbar:
        chunk = keys[i:i + batch_imgs]
        paths, ok = [], []
        for key in chunk:
            cells = groups[key]
            if any(p is None for p in cells):
                continue                       # 网格不全则跳过
            paths.extend(cells)
            ok.append(key)
        if not paths:
            continue
        x = torch.stack([
            cast(torch.Tensor, transform(Image.open(p).convert("RGB")))
            for p in paths
        ]).to(device)
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = F.softmax(logits, dim=1)[:, 1].cpu()
        labels = logits.argmax(1).cpu().float()
        for j, key in enumerate(ok):
            G = torch.stack([labels[j * 48:(j + 1) * 48],
                             probs[j * 48:(j + 1) * 48]], dim=0).view(2, GRID_ROWS, GRID_COLS)
            sp, cls = key_info(key)
            entries.append({"id": key[1], "cls": cls, "split": sp, "G": G})
        pbar.set_postfix(done=len(entries))
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "checkpoints", "caret_s0_448_bal_best.pth"))
    ap.add_argument("--grids-root", default=os.path.join(ROOT, "data", "grids_448"))
    ap.add_argument("--e2e-root", default=os.path.join(ROOT, "data", "e2e_grids_448"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "split_v2_manifest.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "cache", "stage1_g_cache_448_bal.pt"))
    ap.add_argument("--batch-imgs", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE,
                    help="与训练时一致; 448 数据必须设为 448")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    global transform
    transform = make_transform(args.img_size)
    print(f"模型: {args.model}\n网格源: {args.grids_root}\n"
          f"输入尺寸: {args.img_size}\n设备: {device}", flush=True)

    # 类别查表: id -> cls (来自 manifest)
    man = json.load(open(args.manifest, encoding="utf-8"))
    cls_of_anno = {}
    for s in ("train", "valid", "test"):
        for item in man["split"][s]:
            cls_of_anno[item["id"]] = item["cls"]

    model = load_model(args.model, device)

    g1 = collect_anno(args.grids_root)
    print(f"标注图: {len(g1)} 张 (split,id) 组", flush=True)
    entries = run_groups({k: v for k, v in g1.items() if k[1] in cls_of_anno},
                         model, device, args.batch_imgs, "标注图",
                         lambda k: (k[0], cls_of_anno.get(k[1], "unknown")))

    if os.path.isdir(args.e2e_root):
        g2 = collect_e2e(args.e2e_root)
        print(f"端到端测试图: {len(g2)} 张", flush=True)
        entries += run_groups(g2, model, device, args.batch_imgs, "e2e",
                              lambda k: ("e2e_test", k[0]))
    else:
        print(f"[WARN] 端到端网格目录不存在: {args.e2e_root}", flush=True)

    if not entries:
        print("[ERR] 没有任何条目, 检查网格目录与文件名格式", flush=True)
        sys.exit(1)

    torch.save({"entries": entries}, args.out)
    print(f"\n[OK] {len(entries)} 条 -> {args.out}", flush=True)
    print("split 分布:", dict(Counter(e["split"] for e in entries)), flush=True)
    print("类别分布:", dict(Counter(e["cls"] for e in entries)), flush=True)


if __name__ == "__main__":
    main()
