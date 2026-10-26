"""
DashLight - local training script (Windows / Linux, NVIDIA GPU).

Usage (from repo root, venv active):
    export ROBOFLOW_API_KEY=xxxxx          # Git Bash  (PowerShell: $env:ROBOFLOW_API_KEY="xxxxx")
    python scripts/train_local.py                       # full run, defaults below
    python scripts/train_local.py --epochs 5 --name smoke_test
    python scripts/train_local.py --resume --name dashlight_v1_e100

Big files (dataset, runs, weights) go to ~/dashlight_work, OUTSIDE the repo/OneDrive.
Only small result files (charts + metrics.md) are copied into results/<name>/.
"""
import argparse
import datetime
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="Train DashLight YOLO model locally")
    p.add_argument("--workspace", default="snapmyauto")
    p.add_argument("--project", default="dashlight")
    p.add_argument("--version", type=int, default=1, help="Roboflow dataset version")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8, help="8 fits a 4GB GPU at 640px")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--name", default=None, help="run name (default dashlight_v<version>_e<epochs>)")
    p.add_argument("--workdir", default="~/dashlight_work")
    p.add_argument("--resume", action="store_true", help="resume run <name> from last.pt")
    return p.parse_args()


def get_dataset(args, workdir: Path) -> Path:
    ds_dir = workdir / "datasets" / f"{args.project}-v{args.version}"
    data_yaml = ds_dir / "data.yaml"
    if data_yaml.exists():
        print(f"Dataset already downloaded: {ds_dir}")
        return data_yaml

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("ROBOFLOW_API_KEY is not set. Run:  export ROBOFLOW_API_KEY=your_key")

    from roboflow import Roboflow

    rf = Roboflow(api_key=key)
    version = rf.workspace(args.workspace).project(args.project).version(args.version)
    version.download("yolov8", location=str(ds_dir))
    return data_yaml


def copy_results(run_dir: Path, name: str, metrics, args, data_yaml: Path):
    out = REPO_ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    for f in ["results.png", "results.csv", "confusion_matrix.png",
              "confusion_matrix_normalized.png", "BoxPR_curve.png", "PR_curve.png",
              "BoxF1_curve.png", "F1_curve.png", "args.yaml"]:
        src = run_dir / f
        if src.exists():
            shutil.copy(src, out / f)

    import yaml
    names = yaml.safe_load(open(data_yaml))["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]

    n_train = len(list((data_yaml.parent / "train" / "images").glob("*")))
    lines = [
        f"# DashLight {name}",
        "",
        f"- Date: {datetime.date.today()}",
        f"- Base model: {args.model}",
        f"- Epochs (max): {args.epochs}, patience: {args.patience}, imgsz: {args.imgsz}, batch: {args.batch}",
        f"- Roboflow: {args.workspace}/{args.project} v{args.version}, train images: {n_train}",
        "- Hardware: local GPU",
        "",
        "## Test set",
        "",
        "| mAP@50 | mAP@50-95 | Precision | Recall |",
        "|---|---|---|---|",
        f"| {metrics.box.map50:.3f} | {metrics.box.map:.3f} | {metrics.box.mp:.3f} | {metrics.box.mr:.3f} |",
        "",
        "## Per-class mAP@50-95",
        "",
        "| Class | mAP@50-95 |",
        "|---|---|",
    ]
    for c in metrics.box.ap_class_index:
        lines.append(f"| {names[c]} | {metrics.box.maps[c]:.3f} |")
    (out / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out / "labels.txt").write_text("\n".join(names), encoding="utf-8")
    print(f"\nResults copied to {out}")


def main():
    args = parse_args()
    workdir = Path(args.workdir).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)
    runs_dir = workdir / "runs"
    name = args.name or f"{args.project}_v{args.version}_e{args.epochs}"

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU not found. Check the PyTorch install (needs +cu build).")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_yaml = get_dataset(args, workdir)
    run_dir = runs_dir / name

    if args.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.exists():
            sys.exit(f"Nothing to resume: {last} not found")
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(args.model).train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            workers=args.workers,
            fliplr=0.0,          # icons are not symmetric
            project=str(runs_dir),
            name=name,
            exist_ok=True,
        )

    best = run_dir / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    metrics = YOLO(str(best)).val(data=str(data_yaml), split="test", imgsz=args.imgsz,
                                  batch=args.batch, project=str(runs_dir), name=f"{name}_test",
                                  exist_ok=True)
    print(f"mAP@50={metrics.box.map50:.3f}  mAP@50-95={metrics.box.map:.3f}  "
          f"P={metrics.box.mp:.3f}  R={metrics.box.mr:.3f}")
    copy_results(run_dir, name, metrics, args, data_yaml)


if __name__ == "__main__":   # required on Windows (dataloader workers)
    main()
