import csv
import math
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
LUNA_ROOT = PROJECT_DIR / "LUNA16"
OUTPUT_DIR = PROJECT_DIR / "luna16_yolo"
VALIDATION_FRACTION = 0.2
RANDOM_SEED = 42
NEGATIVE_SLICES_PER_EMPTY_SCAN = 8
HU_MIN = -1000
HU_MAX = 400

PLANES = {
    "axial": {"slice_axis": 2, "x_axis": 0, "y_axis": 1},
    "coronal": {"slice_axis": 1, "x_axis": 0, "y_axis": 2},
    "sagittal": {"slice_axis": 0, "x_axis": 1, "y_axis": 2},
}


def read_annotations(csv_path: Path) -> dict[str, list[dict[str, float]]]:
    annotations: dict[str, list[dict[str, float]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path} must contain columns {sorted(required_columns)}.")

        for row in reader:
            annotations.setdefault(row["seriesuid"], []).append(
                {
                    "x": float(row["coordX"]),
                    "y": float(row["coordY"]),
                    "z": float(row["coordZ"]),
                    "diameter": float(row["diameter_mm"]),
                }
            )
    return annotations


def yolo_box(
    center_xyz: tuple[float, float, float],
    radius_mm: float,
    slice_index: int,
    slice_axis: int,
    x_axis: int,
    y_axis: int,
    spacing_xyz: tuple[float, float, float],
    width: int,
    height: int,
) -> str | None:
    distance_mm = abs(slice_index - center_xyz[slice_axis]) * spacing_xyz[slice_axis]
    if distance_mm > radius_mm:
        return None

    chord_radius_mm = math.sqrt(max(0.0, radius_mm**2 - distance_mm**2))
    radius_x = chord_radius_mm / spacing_xyz[x_axis]
    radius_y = chord_radius_mm / spacing_xyz[y_axis]
    left = max(0.0, center_xyz[x_axis] - radius_x)
    right = min(float(width), center_xyz[x_axis] + radius_x)
    top = max(0.0, center_xyz[y_axis] - radius_y)
    bottom = min(float(height), center_xyz[y_axis] + radius_y)
    if right <= left or bottom <= top:
        return None

    center_x = (left + right) / (2 * width)
    center_y = (top + bottom) / (2 * height)
    box_width = (right - left) / width
    box_height = (bottom - top) / height
    return f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"


def save_slice(volume: np.ndarray, slice_axis: int, index: int, image_path: Path) -> None:
    slice_image = np.take(volume, index, axis=2 - slice_axis)
    windowed = np.clip(slice_image, HU_MIN, HU_MAX)
    gray = ((windowed - HU_MIN) * (255.0 / (HU_MAX - HU_MIN))).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    Image.fromarray(rgb).save(image_path, quality=95)


def prepare_scan(
    scan_path: Path,
    series_uid: str,
    split: str,
    annotations: list[dict[str, float]],
    rng: random.Random,
) -> None:
    image = sitk.ReadImage(str(scan_path))
    volume = sitk.GetArrayFromImage(image)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D scan in {scan_path}, got shape {volume.shape}.")

    spacing_xyz = tuple(float(value) for value in image.GetSpacing())
    size_xyz = image.GetSize()
    positive_labels: dict[str, dict[int, list[str]]] = {
        plane: {} for plane in PLANES
    }

    for annotation in annotations:
        center_xyz = image.TransformPhysicalPointToContinuousIndex(
            (annotation["x"], annotation["y"], annotation["z"])
        )
        radius_mm = annotation["diameter"] / 2
        if radius_mm <= 0:
            raise ValueError(f"Invalid nodule diameter in scan {series_uid}.")

        for plane, axes in PLANES.items():
            slice_axis = axes["slice_axis"]
            x_axis = axes["x_axis"]
            y_axis = axes["y_axis"]
            width = size_xyz[x_axis]
            height = size_xyz[y_axis]
            slice_count = size_xyz[slice_axis]
            radius_slices = radius_mm / spacing_xyz[slice_axis]
            first_slice = max(0, math.ceil(center_xyz[slice_axis] - radius_slices))
            last_slice = min(slice_count - 1, math.floor(center_xyz[slice_axis] + radius_slices))

            for slice_index in range(first_slice, last_slice + 1):
                label = yolo_box(
                    center_xyz,
                    radius_mm,
                    slice_index,
                    slice_axis,
                    x_axis,
                    y_axis,
                    spacing_xyz,
                    width,
                    height,
                )
                if label is not None:
                    positive_labels[plane].setdefault(slice_index, []).append(label)

    for plane, axes in PLANES.items():
        slice_count = size_xyz[axes["slice_axis"]]
        positive_indices = set(positive_labels[plane])
        negative_candidates = [index for index in range(slice_count) if index not in positive_indices]
        negative_count = len(positive_indices) or NEGATIVE_SLICES_PER_EMPTY_SCAN
        negative_indices = rng.sample(negative_candidates, min(negative_count, len(negative_candidates)))
        selected_indices = sorted(positive_indices.union(negative_indices))

        image_dir = OUTPUT_DIR / "images" / split
        label_dir = OUTPUT_DIR / "labels" / split
        for slice_index in selected_indices:
            stem = f"{series_uid}_{plane}_{slice_index:04d}"
            save_slice(volume, axes["slice_axis"], slice_index, image_dir / f"{stem}.jpg")
            labels = positive_labels[plane].get(slice_index, [])
            (label_dir / f"{stem}.txt").write_text(
                "\n".join(labels), encoding="utf-8"
            )


def main() -> None:
    annotations_path = LUNA_ROOT / "annotations.csv"
    scan_paths = sorted(LUNA_ROOT.glob("subset*/*.mhd"))
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Could not find LUNA16 annotations at {annotations_path}.")
    if len(scan_paths) < 2:
        raise FileNotFoundError(
            f"Expected LUNA16 .mhd scans under {LUNA_ROOT}\\subset0 through subset9."
        )
    if not 0 < VALIDATION_FRACTION < 1:
        raise ValueError("VALIDATION_FRACTION must be between 0 and 1.")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_DIR} is not empty. Choose a new OUTPUT_DIR to avoid mixing datasets."
        )

    annotations_by_scan = read_annotations(annotations_path)
    series_uids = [path.stem for path in scan_paths]
    shuffled_uids = series_uids.copy()
    random.Random(RANDOM_SEED).shuffle(shuffled_uids)
    validation_count = max(1, round(len(shuffled_uids) * VALIDATION_FRACTION))
    validation_uids = set(shuffled_uids[:validation_count])

    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    for scan_path in scan_paths:
        series_uid = scan_path.stem
        split = "val" if series_uid in validation_uids else "train"
        rng = random.Random(f"{RANDOM_SEED}:{series_uid}")
        prepare_scan(
            scan_path,
            series_uid,
            split,
            annotations_by_scan.get(series_uid, []),
            rng,
        )
        print(f"Prepared {series_uid} ({split})")

    (OUTPUT_DIR / "data.yaml").write_text(
        f'path: "{OUTPUT_DIR.resolve().as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: nodule\n",
        encoding="utf-8",
    )
    print(f"YOLO dataset ready at {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
