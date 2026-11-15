#!/usr/bin/env python
"""PNRBCT pipeline entry point.

Run from the workspace root (e.g., D:/PNRBCT):
  python main.py stage1-train
  python main.py stage1-eval
  python main.py stage2
  python main.py stage3
  python main.py e2e
  python main.py all

This is a thin dispatcher: it calls the stage scripts in CARETR_Project/ in the
correct order via subprocess. It does not re-implement any training or
evaluation logic.

Use --dry-run to print the commands without executing them, and
--skip-existing to skip stages whose output file already exists.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "CARETR_Project")
PY = sys.executable

# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------
def build_steps(args):
    """Return an ordered list of (name, command, expected_output)."""
    img = str(args.img_size)
    data_root = args.data_root or os.path.join(ROOT, "data", "grids_448")
    e2e_root = os.path.join(ROOT, "data", "e2e_grids_448")
    manifest = os.path.join(ROOT, "data", "split_v2_manifest.json")
    cache = os.path.join(ROOT, "data", "cache", "stage1_g_cache_448_bal.pt")
    pnr_ds = os.path.join(ROOT, "data", "cache", "pnr_dataset.pt")
    ckpt = os.path.join(ROOT, "checkpoints")
    res = os.path.join(ROOT, "results")
    s1, s2, s3 = (os.path.join(ROOT, d) for d in
                  ("stage1_care_transformer", "stage2_pnr", "stage3_cdcc"))

    stage1_eval = [PY, os.path.join(s1, "eval_stage1.py"),
                   "--data-root", data_root, "--img-size", img]
    if args.batch_size:
        stage1_eval += ["--batch-size", str(args.batch_size)]

    gcache = [PY, os.path.join(s2, "gen_g_cache.py"),
              "--grids-root", data_root, "--e2e-root", e2e_root,
              "--manifest", manifest, "--img-size", img, "--out", cache]
    if args.batch_imgs:
        gcache += ["--batch-imgs", str(args.batch_imgs)]

    e2e_common = [PY, os.path.join(s3, "test_e2e.py"), "--arch", "cdcc",
                  "--cache", cache]

    return [
        ("stage1-train",
         [PY, os.path.join(s1, "train_stage1.py"),
          "--data-root", data_root, "--img-size", img],
         os.path.join(ckpt, "caret_s0_448_bal_best.pth")),
        ("stage1-eval", stage1_eval,
         os.path.join(res, "stage1", "stage1_448_bal_eval_meta.json")),
        ("stage2-gcache", gcache, cache),
        ("stage2-pnr-data", [PY, os.path.join(s2, "build_pnr_dataset.py")], pnr_ds),
        ("stage2-pnr-train", [PY, os.path.join(s2, "train_pnr.py")],
         os.path.join(ckpt, "best_pnr.pth")),
        ("stage3-data-pnr",
         [PY, os.path.join(s3, "gen_cdcc_data.py"), "--arm", "pnr"],
         os.path.join(ROOT, "data", "cdcc_data", "pnr", "train")),
        ("stage3-data-nopnr",
         [PY, os.path.join(s3, "gen_cdcc_data.py"), "--arm", "nopnr"],
         os.path.join(ROOT, "data", "cdcc_data", "nopnr", "train")),
        ("stage3-train-cdcc-pnr",
         [PY, os.path.join(s3, "train_cdcc.py"), "--arm", "pnr", "--model", "cdcc"],
         os.path.join(ckpt, "best_cdcc_pnr.pth")),
        ("stage3-train-cdcc-nopnr",
         [PY, os.path.join(s3, "train_cdcc.py"), "--arm", "nopnr", "--model", "cdcc"],
         os.path.join(ckpt, "best_cdcc_nopnr.pth")),
        ("stage3-train-agnostic",
         [PY, os.path.join(s3, "train_cdcc.py"), "--arm", "pnr", "--model", "agnostic"],
         os.path.join(ckpt, "best_agnostic_pnr_agnostic.pth")),
        ("e2e-pnr",
         e2e_common + ["--arm", "pnr",
                       "--pnr-model", os.path.join(ckpt, "best_pnr.pth"),
                       "--cdcc-model", os.path.join(ckpt, "best_cdcc_pnr.pth")],
         os.path.join(res, "stage3", "e2e_cdcc_pnr_eval_meta.json")),
        ("e2e-nopnr",
         e2e_common + ["--arm", "nopnr",
                       "--cdcc-model", os.path.join(ckpt, "best_cdcc_nopnr.pth")],
         os.path.join(res, "stage3", "e2e_cdcc_nopnr_eval_meta.json")),
    ]


STAGE_GROUPS = {
    "stage1-train": ["stage1-train"],
    "stage1-eval": ["stage1-eval"],
    "stage2": ["stage2-gcache", "stage2-pnr-data", "stage2-pnr-train"],
    "stage3": ["stage3-data-pnr", "stage3-data-nopnr",
               "stage3-train-cdcc-pnr", "stage3-train-cdcc-nopnr",
               "stage3-train-agnostic"],
    "e2e": ["e2e-pnr", "e2e-nopnr"],
    "all": None,   # every step, in order
}


def run_step(name, cmd, expected, args):
    if args.skip_existing and os.path.exists(expected):
        print(f"[SKIP] {name}: {os.path.relpath(expected, ROOT)} exists")
        return True
    print(f"[RUN ] {name}")
    print("       " + " ".join(cmd))
    if args.dry_run:
        return True

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"{name}.log")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", env=env)
        assert proc.stdout is not None, "stdout=PIPE should always provide a stream"
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        rc = proc.wait()
    if rc != 0:
        print(f"[FAIL] {name}: exit code {rc} (see {log_path})")
        return False
    print(f"[OK  ] {name}")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="PNRBCT pipeline dispatcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("command", choices=list(STAGE_GROUPS),
                    help="pipeline stage to run")
    ap.add_argument("--data-root", default=None,
                    help="448 grid dataset root (default: data/grids_448)")
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override Stage-1 batch size")
    ap.add_argument("--batch-imgs", type=int, default=None,
                    help="override gen_g_cache.py --batch-imgs")
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "results", "logs"))
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a step if its expected output already exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands without executing them")
    args = ap.parse_args()

    steps = build_steps(args)
    wanted = STAGE_GROUPS[args.command]
    selected = steps if wanted is None else [s for s in steps if s[0] in wanted]

    for name, cmd, expected in selected:
        if not run_step(name, cmd, expected, args):
            sys.exit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
