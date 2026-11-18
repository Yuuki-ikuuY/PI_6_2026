"""Candidatos 3D a partir das detecções por corte e avaliação FROC no protocolo do LUNA16.

Etapas:
  1. cada detecção 2D (caixa em um corte) é convertida para coordenada do mundo (mm);
  2. detecções de cortes vizinhos são agrupadas por proximidade em candidatos 3D;
  3. um candidato acerta um nódulo se a distância ao centro for menor ou igual ao raio;
  4. calcula-se a sensibilidade em função de falsos positivos por exame e a média nas sete
     taxas do LUNA16 (1/8, 1/4, 1/2, 1, 2, 4 e 8): o CPM (competition performance metric).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FP_RATES = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


# ----------------------------------------------------------------------------------------
# Detecções 2D -> candidatos 3D
# ----------------------------------------------------------------------------------------
@dataclass
class Detection:
    """Detecção em um corte: centro no mundo (mm), diâmetro estimado (mm), confiança e corte."""
    x: float
    y: float
    z: float
    diameter_mm: float
    conf: float
    slice_idx: int


@dataclass
class Candidate:
    x: float
    y: float
    z: float
    diameter_mm: float
    prob: float
    n_slices: int


def box_to_detection(volume, k: int, box_xyxy, conf: float) -> Detection:
    """Caixa em pixels (bordas da imagem) no corte k -> Detecção em coordenadas do mundo."""
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    ix, iy = (x1 + x2) / 2.0 - 0.5, (y1 + y2) / 2.0 - 0.5   # borda -> centro do voxel
    x, y, z = volume.index_to_world(ix, iy, float(k))
    sx, sy, _ = volume.spacing
    return Detection(x, y, z, ((x2 - x1) * sx + (y2 - y1) * sy) / 2.0, float(conf), int(k))


def cluster_detections(dets: list[Detection], merge_min_mm: float = 3.0,
                       merge_factor: float = 0.5, min_slices: int = 1) -> list[Candidate]:
    """Agrupa detecções de cortes vizinhos em candidatos 3D (guloso, da maior confiança à menor).

    Uma detecção entra no grupo mais próximo se a distância entre os centros for menor ou igual a
    max(merge_min_mm, merge_factor * maior diâmetro entre a detecção e o grupo). O centro do grupo
    é a média ponderada pela confiança; a probabilidade do candidato é a maior confiança do grupo.
    Grupos com menos de `min_slices` cortes distintos são descartados.
    """
    if not dets:
        return []
    order = sorted(dets, key=lambda d: -d.conf)
    n = len(order)
    centers = np.zeros((n, 3)); diam = np.zeros(n); wsum = np.zeros(n)
    members: list[list[Detection]] = []
    for d in order:
        pt = np.array([d.x, d.y, d.z])
        best = -1
        if members:
            c = len(members)
            dist = np.linalg.norm(centers[:c] - pt, axis=1)
            thr = np.maximum(merge_min_mm, merge_factor * np.maximum(diam[:c], d.diameter_mm))
            ok = np.nonzero(dist <= thr)[0]
            if ok.size:
                best = int(ok[np.argmin(dist[ok])])
        if best < 0:
            best = len(members)
            members.append([])
            centers[best], diam[best], wsum[best] = pt, d.diameter_mm, d.conf
        else:
            w = d.conf
            centers[best] = (centers[best] * wsum[best] + pt * w) / (wsum[best] + w)
            diam[best] = (diam[best] * wsum[best] + d.diameter_mm * w) / (wsum[best] + w)
            wsum[best] += w
        members[best].append(d)
    out = []
    for i, ms in enumerate(members):
        n_sl = len({m.slice_idx for m in ms})
        if n_sl >= min_slices:
            out.append(Candidate(*centers[i], diam[i], max(m.conf for m in ms), n_sl))
    return sorted(out, key=lambda c: -c.prob)


def write_candidates(path: str | Path, rows: list[tuple[str, Candidate]]) -> None:
    """Grava no formato de submissão do LUNA16 (+ diâmetro e nº de cortes, informativos)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seriesuid", "coordX", "coordY", "coordZ", "probability", "diameter_mm", "n_slices"])
        for uid, c in rows:
            w.writerow([uid, f"{c.x:.4f}", f"{c.y:.4f}", f"{c.z:.4f}", f"{c.prob:.6f}",
                        f"{c.diameter_mm:.3f}", c.n_slices])


def read_candidates(path: str | Path) -> dict[str, np.ndarray]:
    """Lê candidatos: {seriesuid: array (n, 4) com x, y, z, probabilidade}."""
    tmp: dict[str, list] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tmp.setdefault(r["seriesuid"], []).append(
                (float(r["coordX"]), float(r["coordY"]), float(r["coordZ"]), float(r["probability"])))
    return {k: np.array(v) for k, v in tmp.items()}


# ----------------------------------------------------------------------------------------
# FROC
# ----------------------------------------------------------------------------------------
def match_candidates(candidates: dict[str, np.ndarray], nodules: dict[str, list[tuple]],
                     excluded: dict[str, list[tuple]] | None = None):
    """Classifica os candidatos em acertos, falsos positivos ou ignorados.

    `nodules` e `excluded`: {seriesuid: [(x, y, z, diametro_mm), ...]}.
    - acerto: distância ao centro <= raio; por nódulo conta só o candidato de maior probabilidade
      (os demais que o atingem são ignorados, não viram falso positivo);
    - achados 'excluídos' (ex.: nódulos < 3 mm) atingidos também são ignorados;
    - o resto é falso positivo.
    Retorna (melhor_prob_por_nodulo, probs_dos_falsos_positivos); -1 = nódulo não detectado.
    """
    excluded = excluded or {}
    best, fp = [], []
    best_by_scan = {uid: np.full(len(ns), -1.0) for uid, ns in nodules.items()}
    for uid, arr in candidates.items():
        ns = np.array(nodules.get(uid, []), dtype=float).reshape(-1, 4)
        ex = np.array(excluded.get(uid, []), dtype=float).reshape(-1, 4)
        for x, y, z, p in arr:
            hit = -1
            if len(ns):
                d = np.linalg.norm(ns[:, :3] - np.array([x, y, z]), axis=1)
                ok = np.nonzero(d <= ns[:, 3] / 2.0)[0]
                if ok.size:
                    hit = int(ok[np.argmin(d[ok])])
            if hit >= 0:
                best_by_scan[uid][hit] = max(best_by_scan[uid][hit], p)
                continue
            if len(ex):
                d = np.linalg.norm(ex[:, :3] - np.array([x, y, z]), axis=1)
                if np.any(d <= ex[:, 3] / 2.0):
                    continue
            fp.append(p)
    for v in best_by_scan.values():
        best.extend(v.tolist())
    return np.array(best), np.array(fp)


def froc_curve(best_prob: np.ndarray, fp_probs: np.ndarray, n_scans: int):
    """Curva FROC: (falsos positivos por exame, sensibilidade, limiares), por limiar decrescente."""
    n_nod = len(best_prob)
    thr = np.unique(np.concatenate([best_prob[best_prob >= 0], fp_probs]))[::-1]
    tp_sorted, fp_sorted = np.sort(best_prob[best_prob >= 0]), np.sort(fp_probs)
    tp = len(tp_sorted) - np.searchsorted(tp_sorted, thr, side="left")
    fp = len(fp_sorted) - np.searchsorted(fp_sorted, thr, side="left")
    sens = tp / max(n_nod, 1)
    fps = fp / max(n_scans, 1)
    return np.concatenate([[0.0], fps]), np.concatenate([[0.0], sens]), np.concatenate([[1.0], thr])


def sensitivity_at(fps: np.ndarray, sens: np.ndarray, rates=FP_RATES) -> list[float]:
    """Maior sensibilidade com no máximo `rate` falsos positivos por exame (degraus, sem interpolar)."""
    return [float(sens[fps <= r].max()) for r in rates]


def evaluate(candidates, nodules, n_scans: int, excluded=None, rates=FP_RATES) -> dict:
    best, fp = match_candidates(candidates, nodules, excluded)
    fps, sens, thr = froc_curve(best, fp, n_scans)
    s = sensitivity_at(fps, sens, rates)
    return {"rates": list(rates), "sensitivity": s, "cpm": float(np.mean(s)),
            "n_nodules": int(len(best)), "n_detected": int((best >= 0).sum()),
            "max_sensitivity": float((best >= 0).mean()) if len(best) else 0.0,
            "curve": (fps, sens, thr)}
