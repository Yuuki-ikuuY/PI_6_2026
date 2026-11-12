"""Prepara o LUNA16 para o YOLO: cortes axiais 2,5D, divisão por subconjunto oficial.

A lógica fica em src/luna16_to_yolo.py, que tem testes automáticos (pytest).
"""
import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from luna16_to_yolo import build_dataset  # noqa: E402

LUNA_ROOT = PROJECT_DIR / "LUNA16"           # contém subset0..subset9 e annotations.csv
OUTPUT_DIR = PROJECT_DIR / "dataset_25d"     # pasta nova: não misturar com o dataset antigo
NEGATIVE_RATIO = 0.10                         # fração de cortes sem nódulo (treino e validação)
EMPTY_SCAN_SLICES = 10                        # cortes por exame sem nódulos
RANDOM_SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser(description="Gera o dataset 2,5D no formato YOLO a partir do LUNA16.")
    ap.add_argument("--luna-dir", type=Path, default=LUNA_ROOT)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--full-test", action="store_true", help="exporta todos os cortes dos exames de teste")
    ap.add_argument("--min-section-mm", type=float, default=0.0, help="descarta caixas com secção menor (mm)")
    ap.add_argument("--limit-series", type=int, default=None, help="processa só N exames (depuração)")
    a = ap.parse_args()

    if not (a.luna_dir / "annotations.csv").is_file():
        raise FileNotFoundError(f"annotations.csv não encontrado em {a.luna_dir}")
    if a.out.exists() and any(a.out.iterdir()):
        raise FileExistsError(f"{a.out} não está vazia. Use outra pasta para não misturar datasets.")

    stats = build_dataset(a.luna_dir, a.out, neg_ratio=NEGATIVE_RATIO, seed=RANDOM_SEED,
                          full_test=a.full_test, empty_series_slices=EMPTY_SCAN_SLICES,
                          min_section_diameter_mm=a.min_section_mm, limit_series=a.limit_series)
    for split, s in stats.items():
        print(f"{split:5s}: {s['series']:4d} exames | {s['images']:6d} imagens | "
              f"{s['positives']:6d} com nódulo | {s['boxes']:6d} caixas")
    print(f"Dataset pronto em {a.out.resolve()}")


if __name__ == "__main__":
    main()
