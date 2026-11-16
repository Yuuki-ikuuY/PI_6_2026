"""Testes do agrupamento 3D, do FROC e da predição (com modelo simulado)."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

import froc as F
import luna16_to_yolo as L
from test_luna16_to_yolo import CENTER_IDX, DIAMETER, SHAPE, center_world, make_ct, nodule

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # para importar predict_luna16_yolo.py (script na raiz)


def det(x, y, z, d=10.0, conf=0.9, k=0):
    return F.Detection(x, y, z, d, conf, k)


# ------------------------------------------------------------------ agrupamento 3D
def test_deteccoes_de_cortes_vizinhos_viram_um_candidato():
    dets = [det(0, 0, 0, conf=0.9, k=10), det(0.5, 0, 2.5, conf=0.7, k=11), det(0, 0.4, -2.5, conf=0.5, k=9)]
    cands = F.cluster_detections(dets)
    assert len(cands) == 1
    assert cands[0].prob == 0.9 and cands[0].n_slices == 3


def test_deteccoes_distantes_geram_candidatos_separados():
    cands = F.cluster_detections([det(0, 0, 0), det(60, 0, 0, conf=0.5)])
    assert len(cands) == 2 and cands[0].prob > cands[1].prob


def test_min_slices_descarta_deteccoes_isoladas():
    dets = [det(0, 0, 0, k=3), det(0, 0, 2.5, k=4), det(80, 0, 0, k=7)]
    assert len(F.cluster_detections(dets, min_slices=2)) == 1


def test_lista_vazia():
    assert F.cluster_detections([]) == []


# ------------------------------------------------------------------ caixa -> mundo
def test_caixas_verdadeiras_voltam_ao_centro_do_nodulo_em_mm():
    """Ida e volta: anotação -> caixas por corte -> detecções -> candidato = centro do nódulo."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        vol = L.load_volume(make_ct(Path(tmp) / "e.mhd"))
        per_slice = L.boxes_per_slice(vol, [nodule()])
        dets = [F.box_to_detection(vol, k, b, 0.9) for k, bs in per_slice.items() for b in bs]
    cands = F.cluster_detections(dets)
    assert len(cands) == 1
    c = cands[0]
    cx, cy, cz = center_world()
    assert np.linalg.norm([c.x - cx, c.y - cy, c.z - cz]) < 1.0           # < 1 mm do centro real
    assert c.diameter_mm == pytest.approx(DIAMETER, rel=0.35)


# ------------------------------------------------------------------ FROC
NODS = {"a": [(0, 0, 0, 10.0)], "b": [(0, 0, 0, 6.0)], "c": []}       # 3 exames, 2 nódulos


def cands(**kw):
    return {k: np.array(v, dtype=float).reshape(-1, 4) for k, v in kw.items()}


def test_deteccao_perfeita_sem_falsos_positivos():
    r = F.evaluate(cands(a=[(1, 0, 0, 0.9)], b=[(0, 1, 0, 0.8)]), NODS, n_scans=3)
    assert r["sensitivity"] == [1.0] * 7 and r["cpm"] == 1.0


def test_distancia_maior_que_o_raio_nao_e_acerto():
    r = F.evaluate(cands(a=[(5.5, 0, 0, 0.9)]), NODS, n_scans=3)      # raio 5 mm, distância 5,5 mm
    assert r["n_detected"] == 0 and r["cpm"] == 0.0


def test_candidatos_extras_no_mesmo_nodulo_nao_viram_falso_positivo():
    c = cands(a=[(0, 0, 0, 0.9), (1, 1, 0, 0.8), (-1, 0, 1, 0.7)])
    best, fp = F.match_candidates(c, NODS)
    assert best.tolist() == [0.9, -1.0] and len(fp) == 0


def test_falsos_positivos_por_exame_e_sensibilidade_por_limiar():
    # exame a: acerto (0.9) + 2 FP (0.8 e 0.4); exame b: acerto com 0.6; exame c: 1 FP (0.7)
    c = cands(a=[(0, 0, 0, 0.9), (50, 0, 0, 0.8), (60, 0, 0, 0.4)], b=[(0, 0, 0, 0.6)], c=[(5, 5, 5, 0.7)])
    best, fp = F.match_candidates(c, NODS)
    fps, sens, thr = F.froc_curve(best, fp, n_scans=3)
    # limiar 0.9: 1 de 2 nódulos, 0 FP; limiar 0.6: 2 de 2, 2 FP em 3 exames
    assert sens[1] == 0.5 and fps[1] == 0.0
    assert sens[thr.tolist().index(0.6)] == 1.0 and fps[thr.tolist().index(0.6)] == pytest.approx(2 / 3)
    s = F.sensitivity_at(fps, sens)
    assert s[:3] == [0.5, 0.5, 0.5] and s[3] == 1.0                    # com até 1 FP/exame: sens 1.0


def test_achados_excluidos_sao_ignorados():
    ex = {"c": [(5, 5, 5, 8.0)]}
    best, fp = F.match_candidates(cands(c=[(5, 5, 5, 0.7)]), NODS, excluded=ex)
    assert len(fp) == 0
    best, fp = F.match_candidates(cands(c=[(5, 5, 5, 0.7)]), NODS)
    assert len(fp) == 1


def test_sensibilidade_nunca_decresce_com_mais_fp():
    rng = np.random.default_rng(0)
    c = cands(a=[(0, 0, 0, 0.5)] + [(rng.uniform(20, 90), 0, 0, rng.uniform()) for _ in range(30)],
              b=[(0, 0, 0, 0.3)])
    best, fp = F.match_candidates(c, NODS)
    fps, sens, _ = F.froc_curve(best, fp, 3)
    assert np.all(np.diff(fps) >= 0) and np.all(np.diff(sens) >= 0)


def test_leitura_e_escrita_de_candidatos(tmp_path):
    p = tmp_path / "c.csv"
    F.write_candidates(p, [("a", F.Candidate(1.0, 2.0, 3.0, 8.0, 0.75, 3))])
    arr = F.read_candidates(p)["a"]
    assert arr.shape == (1, 4) and arr[0].tolist() == [1.0, 2.0, 3.0, 0.75]


# ------------------------------------------------------------------ predição com modelo simulado
class _T:
    def __init__(self, a): self.a = np.asarray(a, dtype=float)
    def cpu(self): return self
    def numpy(self): return self.a
    def __len__(self): return len(self.a)


class _Result:
    def __init__(self, xyxy, conf):
        self.boxes = type("B", (), {"xyxy": _T(xyxy), "conf": _T(conf), "__len__": lambda s: len(xyxy)})()


class ModeloOraculo:
    """'Modelo' que devolve exatamente as caixas verdadeiras (identifica o corte pelo canal central)."""
    def __init__(self, volume, per_slice):
        self.keys = {tuple(L.make_25d(volume.array, k)[0, 0]): k for k in range(volume.shape_zyx[0])}
        self.vol, self.per_slice, self.chamadas = volume, per_slice, 0

    def predict(self, source, **kw):
        self.chamadas += 1
        out = []
        for img in source:
            # localiza o corte comparando com o volume (suficiente para o teste)
            k = next(i for i in range(self.vol.shape_zyx[0])
                     if np.array_equal(L.make_25d(self.vol.array, i), img))
            bs = self.per_slice.get(k, [])
            out.append(_Result(bs, [0.9] * len(bs)))
        return out


def test_predicao_completa_com_modelo_oraculo_da_cpm_perfeito(tmp_path):
    import predict_luna16_yolo as P
    vol = L.load_volume(make_ct(tmp_path / "exame.mhd"))
    modelo = ModeloOraculo(vol, L.boxes_per_slice(vol, [nodule()]))
    dets = P.detect_exam(modelo, vol, batch=7)
    assert modelo.chamadas == -(-SHAPE[0] // 7)                      # processou em lotes
    cands_ = {"exame": np.array([[c.x, c.y, c.z, c.prob] for c in F.cluster_detections(dets)])}
    cx, cy, cz = center_world()
    r = F.evaluate(cands_, {"exame": [(cx, cy, cz, DIAMETER)]}, n_scans=1)
    assert r["cpm"] == 1.0 and r["n_detected"] == 1
