"""Treina uma variante do YOLO11 (n, s ou m) no dataset 2,5D do LUNA16
Exemplos:
    python train_luna16_yolo.py --model n --seed 42
    python train_luna16_yolo.py --model s --seed 123 --evaluate-test   # só ao final, uma vez
"""
import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent
DATA_YAML = PROJECT_DIR / "dataset_25d" / "data.yaml"
RUNS_DIR = PROJECT_DIR / "runs" / "luna16"
PRETRAINED_WEIGHTS_DIR = PROJECT_DIR / "weights" / "pretrained"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Treino do YOLO11 no LUNA16 (2,5D).")
    ap.add_argument("--model", choices=["n", "s", "m"], required=True, help="variante do YOLO11")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--data", type=Path, default=DATA_YAML)
    ap.add_argument("--evaluate-test", action="store_true",
                    help="avalia no conjunto de teste ao final (usar uma única vez por modelo)")
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    if not a.data.is_file():
        raise FileNotFoundError(f"{a.data} não encontrado. Rode prepare_luna16.py primeiro.")
    device = 0 if torch.cuda.is_available() else "cpu"
    name = f"yolo11{a.model}_seed{a.seed}"
    print(f"Treinando {name} em {'GPU 0' if device == 0 else 'CPU'}")

    PRETRAINED_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    pretrained_weights = PRETRAINED_WEIGHTS_DIR / f"yolo11{a.model}.pt"
    legacy_weights = PROJECT_DIR / pretrained_weights.name
    if not pretrained_weights.exists() and legacy_weights.is_file():
        legacy_weights.replace(pretrained_weights)
    model = YOLO(str(pretrained_weights))  # pesos pré-treinados no COCO (baixados se necessário)
    results = model.train(
        data=str(a.data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=a.patience,
        device=device, project=str(RUNS_DIR), name=name, seed=a.seed, deterministic=True,
        # aumento de dados do artigo (Seção 3.3): imagens em tons de cinza, sem matiz/saturação
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, fliplr=0.5, flipud=0.0, mosaic=1.0, scale=0.5, translate=0.1,
    )
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    summary = {"model": f"yolo11{a.model}", "seed": a.seed, "epochs": a.epochs, "imgsz": a.imgsz,
               "batch": a.batch, "best_weights": str(best)}
    if a.evaluate_test:
        m = YOLO(str(best)).val(data=str(a.data), split="test", imgsz=a.imgsz, device=device, plots=True,
                                project=str(RUNS_DIR), name=f"{name}_test")
        summary["test"] = {"precision": float(m.box.mp), "recall": float(m.box.mr),
                           "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map)}
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
