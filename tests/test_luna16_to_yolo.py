"""Testes da conversão LUNA16 -> YOLO, com exames sintéticos (não exigem o LUNA16 real)."""
import csv
import math
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

import luna16_to_yolo as L

SPACING = (0.8, 0.7, 2.5)            # (x, y, z) em mm
ORIGIN = (-120.0, -80.0, -300.0)     # (x, y, z) em mm
SHAPE = (40, 64, 80)                 # (z, y, x)
CENTER_IDX = (30.0, 25.0, 20.0)      # (ix, iy, iz) do nódulo sintético
DIAMETER = 10.0                      # mm


def center_world():
    """Centro do nódulo em mm: origem + índice * espaçamento."""
    return tuple(ORIGIN[i] + CENTER_IDX[i] * SPACING[i] for i in range(3))


def make_ct(path: Path, with_nodule=True, seed=0) -> Path:
    """Cria um .mhd com fundo de -800 HU e uma esfera de +100 HU (o 'nódulo')."""
    rng = np.random.default_rng(seed)
    arr = np.full(SHAPE, -800, dtype=np.int16) + rng.integers(-5, 5, SHAPE).astype(np.int16)
    if with_nodule:
        z, y, x = np.indices(SHAPE)
        cx, cy, cz = CENTER_IDX
        dist2 = ((x - cx) * SPACING[0]) ** 2 + ((y - cy) * SPACING[1]) ** 2 + ((z - cz) * SPACING[2]) ** 2
        arr[dist2 <= (DIAMETER / 2) ** 2] = 100
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(SPACING)
    img.SetOrigin(ORIGIN)
    sitk.WriteImage(img, str(path))
    return path


def write_annotations(path: Path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"])
        w.writerows(rows)


@pytest.fixture
def volume(tmp_path):
    return L.load_volume(make_ct(tmp_path / "exame.mhd"))


def nodule():
    x, y, z = center_world()
    return L.Nodule(x, y, z, DIAMETER)


# --------------------------------------------------------------------- coordenadas
def test_world_to_index_recupera_o_indice_do_nodulo(volume):
    ix, iy, iz = volume.world_to_index(*center_world())
    assert (ix, iy, iz) == pytest.approx(CENTER_IDX, abs=1e-6)


def test_array_na_ordem_zyx_e_espacamento_xyz(volume):
    assert volume.shape_zyx == SHAPE
    assert volume.spacing == pytest.approx(SPACING)


# --------------------------------------------------------------------- caixas
def test_caixa_central_coincide_com_o_nodulo_visivel(volume):
    boxes = L.boxes_per_slice(volume, [nodule()])
    k = int(CENTER_IDX[2])
    assert k in boxes
    x1, y1, x2, y2 = boxes[k][0]
    mask = volume.array[k] > 0
    ys, xs = np.nonzero(mask)
    # centro da caixa (coordenada de borda) = centroide dos pixels + 0,5
    assert (x1 + x2) / 2 == pytest.approx(xs.mean() + 0.5, abs=0.6)
    assert (y1 + y2) / 2 == pytest.approx(ys.mean() + 0.5, abs=0.6)
    # a caixa envolve a mancha brilhante (tolerância de 1 pixel)
    assert x1 <= xs.min() + 1 and x2 >= xs.max() + 1 - 1
    assert y1 <= ys.min() + 1 and y2 >= ys.max() + 1 - 1


def test_caixa_diminui_quando_o_corte_se_afasta_do_centro(volume):
    boxes = L.boxes_per_slice(volume, [nodule()])
    k = int(CENTER_IDX[2])
    largura = lambda kk: boxes[kk][0][2] - boxes[kk][0][0]
    assert largura(k) > largura(k + 1) > 0


def test_sem_caixa_fora_do_nodulo(volume):
    boxes = L.boxes_per_slice(volume, [nodule()])
    assert 0 not in boxes and SHAPE[0] - 1 not in boxes
    raio_em_cortes = (DIAMETER / 2) / SPACING[2]
    assert max(boxes) <= math.ceil(CENTER_IDX[2] + raio_em_cortes)
    assert min(boxes) >= math.floor(CENTER_IDX[2] - raio_em_cortes)


def test_largura_da_caixa_central_segue_a_formula_do_artigo(volume):
    # lado = 2 * sqrt(r^2 - dz^2); no corte central dz = 0 -> lado = diâmetro
    ix, iy, iz = CENTER_IDX
    x1, y1, x2, y2 = L.slice_box(ix, iy, iz, int(iz), DIAMETER, SPACING, SHAPE[1:])
    assert (x2 - x1) * SPACING[0] == pytest.approx(DIAMETER, rel=1e-6)
    assert (y2 - y1) * SPACING[1] == pytest.approx(DIAMETER, rel=1e-6)


def test_filtro_de_secao_minima_descarta_caixas_pequenas(volume):
    todas = L.boxes_per_slice(volume, [nodule()])
    # cortes vizinhos (dz = 2,5 mm) têm secção de ~8,66 mm; só o central mede 10 mm
    filtradas = L.boxes_per_slice(volume, [nodule()], min_section_diameter_mm=9.5)
    assert len(todas) == 3
    assert set(filtradas) == {int(CENTER_IDX[2])}


def test_caixa_e_recortada_nas_bordas_da_imagem():
    box = L.slice_box(1.0, 1.0, 10.0, 10, 20.0, (1.0, 1.0, 1.0), (64, 64))
    assert box[0] == 0.0 and box[1] == 0.0


# --------------------------------------------------------------------- rótulos YOLO
def test_rotulo_yolo_normalizado_e_no_formato_correto():
    linha = L.yolo_line((10.0, 20.0, 30.0, 60.0), width=100, height=200)
    cls, cx, cy, w, h = linha.split()
    assert cls == "0"
    assert (float(cx), float(cy), float(w), float(h)) == pytest.approx((0.2, 0.2, 0.2, 0.2))
    assert all(0.0 <= float(v) <= 1.0 for v in (cx, cy, w, h))


# --------------------------------------------------------------------- imagens
def test_janelamento_limites_e_tipo():
    hu = np.array([[-2000, -1000, -300, 400, 3000]])
    out = L.window_hu(hu)
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 0, round(700 / 1400 * 255), 255, 255]]


def test_imagem_25d_tem_tres_canais_e_repete_a_borda(volume):
    img = L.make_25d(volume.array, 0)
    assert img.shape == (SHAPE[1], SHAPE[2], 3) and img.dtype == np.uint8
    assert np.array_equal(img[..., 0], img[..., 1])       # corte -1 repete o corte 0
    meio = L.make_25d(volume.array, 5)
    assert np.array_equal(meio[..., 1], L.window_hu(volume.array[5]))


# --------------------------------------------------------------------- divisão
def test_divisao_oficial_dos_subconjuntos():
    assert [L.SPLIT_BY_SUBSET[i] for i in range(10)] == ["train"] * 7 + ["val"] + ["test"] * 2


def luna_sintetico(tmp_path):
    raiz = tmp_path / "luna"
    uids = {0: "serie_a", 7: "serie_b", 8: "serie_c"}
    for subset, uid in uids.items():
        (raiz / f"subset{subset}").mkdir(parents=True)
        make_ct(raiz / f"subset{subset}" / f"{uid}.mhd", with_nodule=(uid != "serie_c"))
    x, y, z = center_world()
    write_annotations(raiz / "annotations.csv", [("serie_a", x, y, z, DIAMETER), ("serie_b", x, y, z, DIAMETER)])
    return raiz


def test_leitura_das_anotacoes_e_do_subconjunto(tmp_path):
    raiz = luna_sintetico(tmp_path)
    ann = L.read_annotations(raiz / "annotations.csv")
    assert set(ann) == {"serie_a", "serie_b"} and ann["serie_a"][0].diameter_mm == DIAMETER
    achados = {uid: sub for uid, _, sub in L.find_series(raiz)}
    assert achados == {"serie_a": 0, "serie_b": 7, "serie_c": 8}


def test_anotacoes_com_colunas_erradas_geram_erro(tmp_path):
    ruim = tmp_path / "ruim.csv"
    ruim.write_text("id,x,y\n1,2,3\n")
    with pytest.raises(ValueError):
        L.read_annotations(ruim)


def test_pipeline_completo_sem_vazamento_entre_conjuntos(tmp_path):
    raiz, saida = luna_sintetico(tmp_path), tmp_path / "saida"
    stats = L.build_dataset(raiz, saida, neg_ratio=0.10, seed=1, empty_series_slices=4)
    assert stats["train"]["series"] == stats["val"]["series"] == stats["test"]["series"] == 1

    por_conjunto = {}
    for split in ("train", "val", "test"):
        imgs = sorted((saida / "images" / split).glob("*.png"))
        lbls = sorted((saida / "labels" / split).glob("*.txt"))
        assert [i.stem for i in imgs] == [l.stem for l in lbls] and imgs
        por_conjunto[split] = {i.stem.rsplit("_", 1)[0] for i in imgs}
    # nenhum exame em mais de um conjunto (requisito de não haver vazamento)
    assert por_conjunto["train"] == {"serie_a"} and por_conjunto["val"] == {"serie_b"}
    assert por_conjunto["test"] == {"serie_c"}

    # exame sem nódulo: só rótulos vazios; exame com nódulo: caixas válidas e proporção de negativos
    for l in (saida / "labels" / "test").glob("*.txt"):
        assert l.read_text() == ""
    for l in (saida / "labels" / "train").glob("*.txt"):
        for linha in l.read_text().splitlines():
            vals = [float(v) for v in linha.split()[1:]]
            assert len(vals) == 4 and all(0.0 <= v <= 1.0 for v in vals)
    n_pos = stats["train"]["positives"]
    assert stats["train"]["images"] - n_pos == round(n_pos * 0.10 / 0.90)
    assert (saida / "data.yaml").exists() and (saida / "slices.csv").exists()


def test_full_test_exporta_todos_os_cortes_de_teste(tmp_path):
    raiz, saida = luna_sintetico(tmp_path), tmp_path / "saida"
    stats = L.build_dataset(raiz, saida, full_test=True)
    assert stats["test"]["images"] == SHAPE[0]


def test_resultado_e_reproduzivel_com_a_mesma_semente(tmp_path):
    raiz = luna_sintetico(tmp_path)
    L.build_dataset(raiz, tmp_path / "s1", seed=7, empty_series_slices=5)
    L.build_dataset(raiz, tmp_path / "s2", seed=7, empty_series_slices=5)
    n1 = sorted(p.name for p in (tmp_path / "s1" / "images" / "test").glob("*.png"))
    n2 = sorted(p.name for p in (tmp_path / "s2" / "images" / "test").glob("*.png"))
    assert n1 == n2
