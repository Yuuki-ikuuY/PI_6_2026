"""Inferência em todos os cortes dos exames e geração de candidatos 3D (arquivo para o FROC).

Para cada exame: monta as imagens 2,5D de todos os cortes, roda o YOLO, converte as caixas para
coordenadas do mundo, agrupa em candidatos 3D e grava candidates.csv. Também registra o tempo de
inferência por corte e por exame (timing.csv).

Exemplo:
    python predict_luna16_yolo.py --weights runs/luna16/yolo11n_seed42/weights/best.pt --split test
"""
import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from froc import box_to_detection, cluster_detections, write_candidates  # noqa: E402
from luna16_to_yolo import SPLIT_BY_SUBSET, find_series, load_volume, make_25d  # noqa: E402


def detect_exam(model, volume, batch: int = 16, conf: float = 0.001, iou: float = 0.6,
                imgsz: int = 512, device=None):
    """Roda o modelo em todos os cortes axiais do exame e devolve a lista de Detection."""
    nz = volume.shape_zyx[0]
    dets = []
    for start in range(0, nz, batch):
        ks = list(range(start, min(start + batch, nz)))
        images = [make_25d(volume.array, k) for k in ks]
        results = model.predict(source=images, imgsz=imgsz, conf=conf, iou=iou, device=device,
                                batch=len(images), verbose=False)
        for k, r in zip(ks, results):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy, cf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
            dets.extend(box_to_detection(volume, k, b, c) for b, c in zip(xyxy, cf))
    return dets


def main() -> None:
    ap = argparse.ArgumentParser(description="Candidatos 3D do YOLO para avaliação FROC.")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--luna-dir", type=Path, default=PROJECT_DIR / "LUNA16")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--scan", type=Path, default=None, help="processa só este .mhd")
    ap.add_argument("--conf", type=float, default=0.001, help="confiança mínima (baixa, para a curva FROC)")
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--merge-min-mm", type=float, default=3.0)
    ap.add_argument("--merge-factor", type=float, default=0.5)
    ap.add_argument("--min-slices", type=int, default=1)
    ap.add_argument("--limit-series", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="candidates.csv (padrão: ao lado dos pesos)")
    a = ap.parse_args()
    if not a.weights.is_file():
        raise FileNotFoundError(f"Pesos não encontrados: {a.weights}")

    from ultralytics import YOLO
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(a.weights))

    if a.scan:
        scans = [(a.scan.stem, a.scan)]
    else:
        scans = [(uid, p) for uid, p, sub in find_series(a.luna_dir) if SPLIT_BY_SUBSET[sub] == a.split]
    if a.limit_series:
        scans = scans[:a.limit_series]
    out = a.out or a.weights.parent.parent / f"candidates_{a.split}.csv"

    rows, timing = [], []
    for uid, path in scans:
        vol = load_volume(path)
        t0 = time.perf_counter()
        dets = detect_exam(model, vol, a.batch, a.conf, a.iou, a.imgsz, device)
        dt = time.perf_counter() - t0
        cands = cluster_detections(dets, a.merge_min_mm, a.merge_factor, a.min_slices)
        rows.extend((uid, c) for c in cands)
        nz = vol.shape_zyx[0]
        timing.append((uid, nz, dt, dt / nz * 1000.0, len(dets), len(cands)))
        print(f"{uid}: {nz} cortes, {len(dets)} detecções, {len(cands)} candidatos, {dt:.1f} s")

    write_candidates(out, rows)
    with open(out.with_name(out.stem + "_timing.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seriesuid", "n_slices", "seconds", "ms_per_slice", "n_detections", "n_candidates"])
        w.writerows(timing)
    print(f"Candidatos salvos em {out}")


if __name__ == "__main__":
    main()
