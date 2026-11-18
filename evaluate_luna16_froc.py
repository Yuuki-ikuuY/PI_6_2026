"""Avalia candidatos 3D pelo protocolo FROC do LUNA16 (sensibilidade x falsos positivos por exame).

Exemplo:
    python evaluate_luna16_froc.py --candidates runs/luna16/candidates_test.csv --split test
"""
import argparse
import csv
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from froc import evaluate, read_candidates  # noqa: E402
from luna16_to_yolo import SPLIT_BY_SUBSET, find_series, read_annotations  # noqa: E402


def read_excluded(path: Path) -> dict[str, list[tuple]]:
    out: dict[str, list[tuple]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["seriesuid"], []).append(
                (float(r["coordX"]), float(r["coordY"]), float(r["coordZ"]), float(r["diameter_mm"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="FROC/CPM no protocolo do LUNA16.")
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--luna-dir", type=Path, default=PROJECT_DIR / "LUNA16")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--excluded", type=Path, default=None,
                    help="annotations_excluded.csv (achados ignorados, opcional)")
    ap.add_argument("--out", type=Path, default=None, help="prefixo dos arquivos de saída")
    a = ap.parse_args()

    scans = {uid for uid, _, sub in find_series(a.luna_dir) if SPLIT_BY_SUBSET[sub] == a.split}
    ann = read_annotations(a.luna_dir / "annotations.csv")
    nodules = {uid: [(n.x, n.y, n.z, n.diameter_mm) for n in ann.get(uid, [])] for uid in scans}
    cands = {uid: arr for uid, arr in read_candidates(a.candidates).items() if uid in scans}
    excluded = read_excluded(a.excluded) if a.excluded else None

    res = evaluate(cands, nodules, n_scans=len(scans), excluded=excluded)
    print(f"Exames: {len(scans)} | nódulos: {res['n_nodules']} | detectados por algum candidato: "
          f"{res['n_detected']} ({res['max_sensitivity']:.1%})")
    for r, s in zip(res["rates"], res["sensitivity"]):
        print(f"  sensibilidade com até {r:g} FP/exame: {s:.3f}")
    print(f"CPM (média das 7 taxas): {res['cpm']:.4f}")

    prefix = a.out or a.candidates.with_suffix("")
    fps, sens, thr = res["curve"]
    with open(f"{prefix}_froc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["fp_per_scan", "sensitivity", "threshold"])
        w.writerows(zip(fps, sens, thr))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 4)); plt.semilogx(fps[1:], sens[1:], drawstyle="steps-post")
    plt.xlim(0.125, 8); plt.ylim(0, 1); plt.grid(alpha=0.3)
    plt.xlabel("Falsos positivos por exame"); plt.ylabel("Sensibilidade"); plt.title(f"FROC (CPM = {res['cpm']:.3f})")
    plt.tight_layout(); plt.savefig(f"{prefix}_froc.png", dpi=200)
    print(f"Curva salva em {prefix}_froc.csv e {prefix}_froc.png")


if __name__ == "__main__":
    main()
