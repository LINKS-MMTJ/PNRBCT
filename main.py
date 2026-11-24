#!/usr/bin/env python
"""PNRBCT pipeline entry point.

Run from the workspace root (e.g., D:/PNRBCT):
  python main.py stage1-train
  python main.py stage1-eval
  python main.py cache
  python main.py stage2
  python main.py stage3
  python main.py e2e
  python main.py all

This is a thin dispatcher: it calls the stage scripts in CARETR_Project/ in the
correct order via subprocess. It does not re-implement any training or
evaluation logic.

Use --dry-run to print the commands without executing them. Each invocation
writes to a new run directory, separate from the published artifacts.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional
from uuid import uuid4

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "CARETR_Project")
PY = sys.executable

class Step(NamedTuple):
    name: str
    command: list
    output: Path
    env: Optional[dict] = None

# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------
def build_steps(args):
    img = str(args.img_size)
    data = Path(ROOT) / "data"
    published = Path(ROOT) / "checkpoints"
    data_root = args.data_root or data / "grids_448"
    e2e_root = args.e2e_root or data / "e2e_grids_448"
    manifest = args.manifest or data / "split_v2_manifest.json"
    ckpt = args.run_dir / "checkpoints"
    res = args.run_dir / "results"
    cache = args.run_dir / "cache" / "stage1_g_cache_448_bal.pt"
    pnr_ds = cache.with_name("pnr_dataset.pt")
    cdcc_data = args.run_dir / "cdcc_data"
    stage1_name = "caret_s0_448_bal_best.pth"
    pnr_name = "best_pnr.pth"
    pnr_cdcc_name = "best_cdcc_pnr.pth"
    nopnr_cdcc_name = "best_cdcc_nopnr.pth"
    input_ckpt = ckpt if args.command == "all" else published
    stage1_model = args.stage1_model or input_ckpt / stage1_name
    pnr_model = args.pnr_model or input_ckpt / pnr_name
    pnr_cdcc = args.cdcc_pnr_model or input_ckpt / pnr_cdcc_name
    nopnr_cdcc = args.cdcc_nopnr_model or input_ckpt / nopnr_cdcc_name
    input_cache = cache if args.command == "all" else args.cache or data / "cache" / cache.name
    stage1_report = res / "stage1" / "stage1_448_bal_eval_meta.json"
    s1, s2, s3 = (Path(ROOT) / d for d in
                  ("stage1_care_transformer", "stage2_pnr", "stage3_cdcc"))

    stage1_eval = [PY, s1 / "eval_stage1.py", "--model", stage1_model,
                   "--data-root", data_root, "--img-size", img, "--out", stage1_report]
    if args.batch_size:
        stage1_eval += ["--batch-size", str(args.batch_size)]

    gcache = [PY, s2 / "gen_g_cache.py", "--model", stage1_model,
              "--grids-root", data_root, "--e2e-root", e2e_root,
              "--manifest", manifest, "--img-size", img, "--out", cache]
    if args.batch_imgs:
        gcache += ["--batch-imgs", str(args.batch_imgs)]

    cdcc_prepare = [PY, s3 / "gen_cdcc_data.py", "--cache", input_cache, "--out-root", cdcc_data]
    cdcc_train = [PY, s3 / "train_cdcc.py", "--data-root", cdcc_data,
                  "--ckpt-dir", ckpt, "--out-dir", res / "stage3"]
    e2e_common = [PY, s3 / "test_e2e.py", "--arch", "cdcc",
                  "--cache", input_cache, "--out-dir", res / "stage3"]

    return [
        Step("stage1-train", [PY, s1 / "train_stage1.py", "--data-root", data_root,
                              "--img-size", img, "--save-dir", ckpt,
                              "--results-dir", res / "stage1"], ckpt / stage1_name),
        Step("stage1-eval", stage1_eval, stage1_report),
        Step("stage2-gcache", gcache, cache),
        Step("stage2-pnr-data", [PY, s2 / "build_pnr_dataset.py", "--cache", cache,
                                 "--out", pnr_ds, "--out-stats",
                                 res / "stage2" / "pnr_dataset_stats.json"], pnr_ds),
        Step("stage2-pnr-train", [PY, s2 / "train_pnr.py"], ckpt / pnr_name,
             {"PNR_DATASET": pnr_ds, "PNR_OUT_DIR": res / "stage2",
              "PNR_CHECKPOINT": ckpt / pnr_name, "PNR_BALANCE": os.getenv("PNR_BALANCE", "1")}),
        Step("stage3-data-pnr", cdcc_prepare + ["--arm", "pnr", "--pnr-model", pnr_model],
             cdcc_data / "pnr" / "train"),
        Step("stage3-data-nopnr", cdcc_prepare + ["--arm", "nopnr"],
             cdcc_data / "nopnr" / "train"),
        Step("stage3-train-cdcc-pnr", cdcc_train + ["--arm", "pnr", "--model", "cdcc"],
             ckpt / pnr_cdcc_name),
        Step("stage3-train-cdcc-nopnr", cdcc_train + ["--arm", "nopnr", "--model", "cdcc"],
             ckpt / nopnr_cdcc_name),
        Step("stage3-train-agnostic", cdcc_train + ["--arm", "pnr", "--model", "agnostic"],
             ckpt / "best_agnostic_pnr_agnostic.pth"),
        Step("e2e-pnr", e2e_common + ["--arm", "pnr", "--pnr-model", pnr_model,
                                      "--cdcc-model", pnr_cdcc],
             res / "stage3" / "e2e_cdcc_pnr_eval_meta.json"),
        Step("e2e-nopnr", e2e_common + ["--arm", "nopnr", "--cdcc-model", nopnr_cdcc],
             res / "stage3" / "e2e_cdcc_nopnr_eval_meta.json"),
    ]


STAGE_GROUPS = {
    "stage1-train": ["stage1-train"],
    "stage1-eval": ["stage1-eval"],
    "cache": ["stage2-gcache"],
    "stage2": ["stage2-gcache", "stage2-pnr-data", "stage2-pnr-train"],
    "stage3": ["stage3-data-pnr", "stage3-data-nopnr",
               "stage3-train-cdcc-pnr", "stage3-train-cdcc-nopnr",
               "stage3-train-agnostic"],
    "e2e": ["e2e-pnr", "e2e-nopnr"],
    "all": None,   # every step, in order
}


def run_step(step, args):
    name, cmd, expected, overrides = step
    cmd = list(map(str, cmd))
    print(f"[RUN ] {name}")
    print("       " + subprocess.list2cmdline(cmd))
    if overrides:
        print("       env: " + ", ".join(f"{key}={value}" for key, value in overrides.items()))
    if args.dry_run:
        return True

    log_path = args.run_dir / "logs" / f"{name}.log"
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    env.update({key: str(value) for key, value in (overrides or {}).items()})
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
    if not (any(expected.rglob("*.pt")) if expected.is_dir() else expected.is_file()):
        print(f"[FAIL] {name}: missing output {expected} (see {log_path})")
        return False
    print(f"[OK  ] {name}")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="PNRBCT pipeline dispatcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("command", choices=list(STAGE_GROUPS),
                    help="pipeline stage to run")
    ap.add_argument("--data-root", type=Path, help="annotated grid dataset root")
    ap.add_argument("--e2e-root", type=Path, help="end-to-end grid dataset root")
    ap.add_argument("--manifest", type=Path, help="image split manifest for cache generation")
    ap.add_argument("--stage1-model", type=Path, help="existing weight for stage1-eval, cache or stage2")
    ap.add_argument("--cache", type=Path, help="existing prediction cache for stage3 or e2e")
    ap.add_argument("--pnr-model", type=Path, help="existing PNR weight for stage3 or e2e")
    ap.add_argument("--cdcc-pnr-model", type=Path, help="existing CDCC weight for e2e with PNR")
    ap.add_argument("--cdcc-nopnr-model", type=Path, help="existing CDCC weight for e2e without PNR")
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override Stage-1 evaluation batch size")
    ap.add_argument("--batch-imgs", type=int, default=None,
                    help="override gen_g_cache.py --batch-imgs")
    ap.add_argument("--run-dir", type=Path,
                    help="new output directory (default: runs/<timestamp>-<id>)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands without executing them")
    args = ap.parse_args()
    input_commands = {
        "stage1_model": ("stage1-eval", "cache", "stage2"),
        "cache": ("stage3", "e2e"),
        "pnr_model": ("stage3", "e2e"),
        "cdcc_pnr_model": ("e2e",),
        "cdcc_nopnr_model": ("e2e",),
    }
    for name, commands in input_commands.items():
        if getattr(args, name) is not None and args.command not in commands:
            ap.error(f"--{name.replace('_', '-')} is only used by {', '.join(commands)}")
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.run_dir = Path(HERE) / "runs" / f"{stamp}-{uuid4().hex[:8]}"

    steps = build_steps(args)
    wanted = STAGE_GROUPS[args.command]
    selected = steps if wanted is None else [step for step in steps if step.name in wanted]
    if args.run_dir.exists():
        ap.error(f"run directory already exists: {args.run_dir}")
    print(f"Run directory: {args.run_dir}")
    if not args.dry_run:
        try:
            args.run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            ap.error(f"run directory already exists: {args.run_dir}")
        for directory in ("logs", "cache", "checkpoints", "results/stage1", "results/stage2", "results/stage3"):
            (args.run_dir / directory).mkdir(parents=True)
        plan = [{"name": step.name, "argv": step.command, "env": step.env or {},
                 "expected_output": step.output} for step in selected]
        with (args.run_dir / "run.json").open("w", encoding="utf-8") as report:
            json.dump({"command": args.command, "steps": plan}, report, indent=2, default=str)

    for step in selected:
        if not run_step(step, args):
            sys.exit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
