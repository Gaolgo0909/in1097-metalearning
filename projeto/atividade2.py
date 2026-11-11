"""Meta-características de domínio e de complexidade da Atividade 2, copiadas de ../atividade2/metacaracteristicas_dominio.ipynb."""
import time

import numpy as np
import problexity as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import atividade1 as a1


def _hellinger(p, q):
    return float(np.sqrt(0.5 * ((np.sqrt(p) - np.sqrt(q)) ** 2).sum()))


def imbalance_degree(contagem):
    K = len(contagem)
    zeta, uniforme = contagem / contagem.sum(), np.full(K, 1 / K)
    m = int((zeta < 1 / K).sum())
    if m == 0:
        return 0.0
    extremo = np.r_[np.zeros(m), np.full(K - m - 1, 1 / K), (m + 1) / K]
    return (m - 1) + _hellinger(zeta, uniforme) / _hellinger(extremo, uniforme)


def lrid(contagem):
    N, K = contagem.sum(), len(contagem)
    return float(2 * (contagem * np.log(K * contagem / N)).sum())


def mf_dominio(Z, y):
    contagem = np.bincount(y).astype(float)
    minoritaria = contagem / contagem.sum() < 1 / len(contagem)
    if not minoritaria.any():
        minoritaria[:] = True
    eh_min = minoritaria[y]

    indices = NearestNeighbors(n_neighbors=6, n_jobs=1).fit(Z).kneighbors(Z, return_distance=False)
    proprio = indices == np.arange(len(y))[:, None]
    proprio[~proprio.any(axis=1), -1] = True  # pontos duplicados
    vizinhos = indices[~proprio].reshape(len(y), 5)
    iguais = (y[vizinhos] == y[:, None]).sum(axis=1)
    erro_1nn = y[vizinhos[:, 0]] != y

    s = iguais[eh_min]
    return {
        "ImbalanceDegree": imbalance_degree(contagem),
        "LRID": lrid(contagem),
        "EffectiveMinorityClassSize": float(np.log2(contagem.min())),
        "MinoritySafeRatio": float(np.mean(s >= 4)),
        "MinorityBorderlineRatio": float(np.mean((s == 2) | (s == 3))),
        "MinorityRareRatio": float(np.mean(s == 1)),
        "MinorityOutlierRatio": float(np.mean(s == 0)),
        "MinorityN3": float(erro_1nn[eh_min].mean()),
        "MajorityN3": float(erro_1nn[~eh_min].mean()) if (~eh_min).any() else np.nan,
        "MeanKDN": float(np.mean((5 - iguais) / 5)),
    }


MEDIDAS = {"F1": px.f1, "F1v": px.f1v, "F2": px.f2, "F3": px.f3, "F4": px.f4,
           "L1": px.l1, "L2": px.l2, "L3": px.l3,
           "N1": px.n1, "N2": px.n2, "N3": px.n3, "N4": px.n4, "T1": px.t1, "LSC": px.lsc,
           "Density": px.density, "ClsCoef": px.clsCoef, "Hubs": px.hubs,
           "T2": px.t2, "T3": px.t3, "T4": px.t4, "C1": px.c1, "C2": px.c2}
N_COMPLEXIDADE = 1000


def mf_complexidade(Z, y):
    if len(y) > N_COMPLEXIDADE:
        _, Z, _, y = train_test_split(Z, y, test_size=N_COMPLEXIDADE, stratify=y,
                                      random_state=a1.SEMENTE)
    valores, tempos = {}, {}
    for nome, medida in MEDIDAS.items():
        inicio = time.perf_counter()
        try:
            valores[nome] = float(medida(Z, y))
        except Exception:
            valores[nome] = np.nan
        tempos[nome] = time.perf_counter() - inicio
    return valores, tempos
