"""Conversão do LUNA16 para o formato do YOLO (detecção de nódulos pulmonares).

Fluxo:
  1. lê cada exame (.mhd) em unidades Hounsfield (HU);
  2. converte as coordenadas dos nódulos (mm) para índices de voxel;
  3. para cada corte axial que intercepta o nódulo, gera uma caixa quadrada cujo lado é
     o diâmetro da secção da esfera naquele corte: 2 * sqrt(r^2 - dz^2);
  4. aplica o janelamento (-1000 a 400 HU) e monta imagens 2,5D (cortes z-1, z, z+1);
  5. divide por exame, pelos subconjuntos oficiais: 0-6 treino, 7 validação, 8-9 teste;
  6. grava imagens, rótulos no formato YOLO e o arquivo data.yaml.

Uso:
    python src/luna16_to_yolo.py --luna-dir /dados/LUNA16 --out /dados/luna16_yolo
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import yaml

# Janela de pulmão e divisão oficial dos 10 subconjuntos do LUNA16
HU_MIN, HU_MAX = -1000.0, 400.0
SPLIT_BY_SUBSET = {**{i: "train" for i in range(7)}, 7: "val", 8: "test", 9: "test"}

# Leitura de dados
@dataclass
class Nodule:
    """Nódulo de referência: centro em mm no sistema do mundo (x, y, z) e diâmetro em mm."""
    x: float
    y: float
    z: float
    diameter_mm: float

class Volume:
    """Exame de TC: array (z, y, x) em HU e conversão de coordenadas do mundo para voxel."""
    def __init__(self, image: sitk.Image):
        self._image = image
        self.array = sitk.GetArrayFromImage(image)  # ordem (z, y, x)
        sx, sy, sz = image.GetSpacing()
        self.spacing = (sx, sy, sz)  # em mm, ordem (x, y, z)

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return self.array.shape

    def index_to_world(self, ix: float, iy: float, iz: float) -> tuple[float, float, float]:
        """Índice contínuo (ix, iy, iz) -> coordenada em mm (x, y, z)."""
        x, y, z = self._image.TransformContinuousIndexToPhysicalPoint((ix, iy, iz))
        return float(x), float(y), float(z)

    def world_to_index(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Coordenada em mm -> índice contínuo (ix, iy, iz). Considera origem, espaçamento e direção."""
        ix, iy, iz = self._image.TransformPhysicalPointToContinuousIndex((x, y, z))
        return float(ix), float(iy), float(iz)


def load_volume(mhd_path: str | Path) -> Volume:
    return Volume(sitk.ReadImage(str(mhd_path)))

def read_annotations(csv_path: str | Path) -> dict[str, list[Nodule]]:
    """Lê annotations.csv (seriesuid, coordX, coordY, coordZ, diameter_mm), agrupado por exame."""
    out: dict[str, list[Nodule]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Colunas esperadas em {csv_path}: {sorted(required)}")
        for row in reader:
            out.setdefault(row["seriesuid"], []).append(
                Nodule(float(row["coordX"]), float(row["coordY"]), float(row["coordZ"]),
                       float(row["diameter_mm"])))
    return out

def find_series(luna_dir: str | Path) -> list[tuple[str, Path, int]]:
    """Lista (seriesuid, caminho do .mhd, número do subconjunto) em subset0..subset9."""
    found = []
    for subset in range(10):
        for mhd in sorted((Path(luna_dir) / f"subset{subset}").glob("*.mhd")):
            found.append((mhd.stem, mhd, subset))
    return found

# Geometria das caixas
def slice_box(ix: float, iy: float, iz: float, k: int, diameter_mm: float,
              spacing: tuple[float, float, float], shape_yx: tuple[int, int],
              min_section_diameter_mm: float = 0.0):
    """Caixa (x1, y1, x2, y2), em pixels da imagem, do nódulo no corte k; ou None.

    A secção da esfera no corte k tem raio rs = sqrt(r^2 - dz^2), em que dz é a distância em
    mm entre o corte e o centro. Se o corte não intercepta a esfera, não há caixa.
    As coordenadas de saída usam as bordas da imagem (o pixel i ocupa [i, i + 1]).
    """
    sx, sy, sz = spacing
    r = diameter_mm / 2.0
    dz = (k - iz) * sz
    if abs(dz) >= r:
        return None
    rs = math.sqrt(r * r - dz * dz)
    if 2.0 * rs < min_section_diameter_mm:
        return None
    h, w = shape_yx
    cx, cy = ix + 0.5, iy + 0.5          # centro do voxel -> coordenada de borda
    x1, x2 = max(0.0, cx - rs / sx), min(float(w), cx + rs / sx)
    y1, y2 = max(0.0, cy - rs / sy), min(float(h), cy + rs / sy)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def boxes_per_slice(volume: Volume, nodules: list[Nodule],
                    min_section_diameter_mm: float = 0.0) -> dict[int, list[tuple]]:
    """Dicionário {índice do corte: [caixas]} para todos os nódulos do exame."""
    nz, h, w = volume.shape_zyx
    sx, sy, sz = volume.spacing
    result: dict[int, list[tuple]] = {}
    for n in nodules:
        ix, iy, iz = volume.world_to_index(n.x, n.y, n.z)
        half_slices = (n.diameter_mm / 2.0) / sz
        for k in range(max(0, math.floor(iz - half_slices)), min(nz - 1, math.ceil(iz + half_slices)) + 1):
            box = slice_box(ix, iy, iz, k, n.diameter_mm, volume.spacing, (h, w),
                            min_section_diameter_mm)
            if box is not None:
                result.setdefault(k, []).append(box)
    return result

def yolo_line(box: tuple, width: int, height: int, class_id: int = 0) -> str:
    """Linha de rótulo do YOLO: 'classe cx cy w h', normalizados entre 0 e 1."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height
    bw, bh = (x2 - x1) / width, (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# Imagens
def window_hu(slice_hu: np.ndarray, lo: float = HU_MIN, hi: float = HU_MAX) -> np.ndarray:
    """Janelamento: limita a [lo, hi] HU e normaliza para uint8 (0 a 255)."""
    clipped = np.clip(slice_hu.astype(np.float32), lo, hi)
    return np.round((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)


def make_25d(volume_array: np.ndarray, k: int, lo: float = HU_MIN, hi: float = HU_MAX) -> np.ndarray:
    """Imagem 2,5D (H, W, 3) com os cortes k-1, k, k+1 nos canais; as bordas repetem o corte vizinho."""
    nz = volume_array.shape[0]
    idx = (max(k - 1, 0), k, min(k + 1, nz - 1))
    return np.stack([window_hu(volume_array[i], lo, hi) for i in idx], axis=-1)



# Montagem do conjunto de dados
def choose_slices(pos: list[int], nz: int, split: str, rng: np.random.Generator,
                  neg_ratio: float, full_test: bool, empty_series_slices: int) -> list[int]:
    """Cortes negativos a exportar. Positivos são sempre exportados (fora desta função).

    - treino/validação: negativos ~ neg_ratio do total do exame (n_neg = n_pos * r / (1 - r));
    - exame sem nódulos: `empty_series_slices` cortes aleatórios;
    - teste com full_test=True: todos os cortes sem nódulo.
    """
    pos_set = set(pos)
    negatives = [k for k in range(nz) if k not in pos_set]
    if split == "test" and full_test:
        return negatives
    if pos:
        n_neg = round(len(pos) * neg_ratio / (1.0 - neg_ratio))
    else:
        n_neg = empty_series_slices
    n_neg = min(n_neg, len(negatives))
    return sorted(rng.choice(negatives, size=n_neg, replace=False).tolist()) if n_neg else []


def build_dataset(luna_dir: str | Path, out_dir: str | Path, neg_ratio: float = 0.10,
                  seed: int = 42, full_test: bool = False, empty_series_slices: int = 10,
                  min_section_diameter_mm: float = 0.0, limit_series: int | None = None) -> dict:
    """Gera imagens, rótulos, data.yaml e slices.csv. Retorna contagens por conjunto."""
    luna_dir, out_dir = Path(luna_dir), Path(out_dir)
    annotations = read_annotations(luna_dir / "annotations.csv")
    rng = np.random.default_rng(seed)
    series = find_series(luna_dir)
    if limit_series:
        series = series[:limit_series]

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {s: {"series": 0, "images": 0, "positives": 0, "boxes": 0} for s in ("train", "val", "test")}
    index_rows = []
    for uid, mhd_path, subset in series:
        split = SPLIT_BY_SUBSET[subset]
        vol = load_volume(mhd_path)
        nz, h, w = vol.shape_zyx
        per_slice = boxes_per_slice(vol, annotations.get(uid, []), min_section_diameter_mm)
        pos = sorted(per_slice)
        keep = sorted(set(pos) | set(choose_slices(pos, nz, split, rng, neg_ratio, full_test,
                                                   empty_series_slices)))
        for k in keep:
            name = f"{uid}_{k:04d}"
            cv2.imwrite(str(out_dir / "images" / split / f"{name}.png"), make_25d(vol.array, k))
            lines = [yolo_line(b, w, h) for b in per_slice.get(k, [])]
            (out_dir / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            index_rows.append((uid, subset, split, k, len(lines)))
            stats[split]["images"] += 1
            stats[split]["positives"] += int(bool(lines))
            stats[split]["boxes"] += len(lines)
        stats[split]["series"] += 1

    with open(out_dir / "slices.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["seriesuid", "subset", "split", "slice", "n_boxes"])
        wr.writerows(index_rows)
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(out_dir.resolve()), "train": "images/train", "val": "images/val",
                        "test": "images/test", "names": {0: "nodulo"}}, f, allow_unicode=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Converte o LUNA16 para o formato YOLO (cortes 2,5D).")
    ap.add_argument("--luna-dir", required=True, help="pasta com subset0..subset9 e annotations.csv")
    ap.add_argument("--out", required=True, help="pasta de saída do conjunto no formato YOLO")
    ap.add_argument("--neg-ratio", type=float, default=0.10, help="fração de cortes sem nódulo (treino/val)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full-test", action="store_true", help="exporta todos os cortes dos exames de teste")
    ap.add_argument("--empty-series-slices", type=int, default=10, help="cortes por exame sem nódulos")
    ap.add_argument("--min-section-mm", type=float, default=0.0, help="descarta caixas com secção menor (mm)")
    ap.add_argument("--limit-series", type=int, default=None, help="processa só N exames (depuração)")
    a = ap.parse_args()
    stats = build_dataset(a.luna_dir, a.out, a.neg_ratio, a.seed, a.full_test, a.empty_series_slices,
                          a.min_section_mm, a.limit_series)
    for split, s in stats.items():
        print(f"{split:5s}: {s['series']:4d} exames | {s['images']:6d} imagens | "
              f"{s['positives']:6d} com nódulo | {s['boxes']:6d} caixas")


if __name__ == "__main__":
    main()