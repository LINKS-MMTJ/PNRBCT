
import os
import argparse

from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CARETR_Project")
DATA_ROOT = os.path.join(ROOT, "data", "images")
OUT_ROOT = os.path.join(ROOT, "data", "e2e_grids_448")
CLASSES = ["grid", "horizontal", "vertical"]
GRID_ROWS, GRID_COLS = 6, 8
GRID_SIZE = 448          # override with --grid-size
IMEXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--grid-size", type=int, default=GRID_SIZE)
    args = ap.parse_args()

    total_img, total_cell = 0, 0
    for ct in CLASSES:
        src = os.path.join(args.data_root, ct, "e2e_test")
        if not os.path.isdir(src):
            print(f"[WARN] 不存在: {src}")
            continue
        files = sorted(f for f in os.listdir(src) if f.endswith(IMEXTS))
        dst = os.path.join(args.out_root, ct)
        os.makedirs(dst, exist_ok=True)
        for f in files:
            iid = os.path.splitext(f)[0]
            img = Image.open(os.path.join(src, f)).convert("RGB")
            W, H = img.size
            new_W, new_H = (W // GRID_COLS) * GRID_COLS, (H // GRID_ROWS) * GRID_ROWS
            img = img.crop((0, 0, new_W, new_H))
            pw, ph = new_W // GRID_COLS, new_H // GRID_ROWS
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    cell = img.crop((c * pw, r * ph, (c + 1) * pw, (r + 1) * ph))
                    if cell.size != (args.grid_size, args.grid_size):
                        cell = cell.resize((args.grid_size, args.grid_size))
                    cell.save(os.path.join(dst, f"{iid}_r{r}_c{c}.png"))
                    total_cell += 1
            total_img += 1
        print(f"  {ct}: {len(files)} 图 -> {len(files)*48} 网格")

    print(f"\n[OK] 共 {total_img} 图, {total_cell} 网格 -> {args.out_root}")


if __name__ == "__main__":
    main()
