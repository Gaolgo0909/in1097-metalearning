"""Fluxos, aprendizado ativo, meta-características e meta-modelos do projeto."""
import warnings

warnings.filterwarnings("ignore")  # o google-auth, que vem com o tsfel, avisa do Python 3.8 a cada import

import pickle
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import tsfel
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from skmultiflow.data import (ConceptDriftStream, HyperplaneGenerator, LEDGenerator, LEDGeneratorDrift, MIXEDGenerator,
                              RandomRBFGeneratorDrift, RandomTreeGenerator, SEAGenerator, SineGenerator, STAGGERGenerator)
from skmultiflow.drift_detection import ADWIN
from skmultiflow.trees import HoeffdingTreeClassifier

import atividade1 as a1
import atividade2 as a2

RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ.parent / "dados_fluxos"
CACHE = RAIZ / "cache"

FIXOS = [0.05, 0.1, 0.2, 0.5, 0.7]
CANDIDATOS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MARGEM, PRE, DELTA = 0.02, 200, 1e-4
JANELA_MIN, JANELA_MAX, JANELAS_POR_FLUXO, MIN_ROTULOS = 64, 30_000, 100, 30
N_JOBS = 16  # deixa núcleos livres para a máquina continuar usável
ACC = [f"acc_{z:g}" for z in CANDIDATOS]
ROT = [f"rot_{z:g}" for z in CANDIDATOS]

# fluxos de avaliação: os 34 do artigo menos os 13 do CTU-13 e o INSECTS sem drift
INSECTS = {
    "insects_abrupt_bal": ("abrupt_balanced", "1WQoIuuVgiuXfzv4kvao6XuLQG37V923O"),
    "insects_incremental_bal": ("incremental_balanced", "1tKQ2KL4m-ACHCVKUDLFPrM4cyhioiOpu"),
    "insects_gradual_bal": ("gradual_balanced", "1fepYkDxwMbuoRUaG_fsymSzkuapS4vJp"),
    "insects_reoccurring_bal": ("incremental-reoccurring_balanced", "1mSKTSsxzYMjdV005AJqrcMGajuu7dUfW"),
    "insects_incr_abrupt_bal": ("incremental-abrupt_balanced", "1-J5WIBN8_F_tomdcrOaiLCxk9nzxtFsf"),
    "insects_abrupt_imb": ("abrupt_imbalanced", "1Z9W2_mwawobXeTihZDTDnFOCtfdgmgDa"),
    "insects_incremental_imb": ("incremental_imbalanced", "1K3vp0EjA4FPDeSgffiBe4CMosbB4CKbL"),
    "insects_gradual_imb": ("gradual_imbalanced", "1_WJ7lxK2sx1i6kq9Nqv4KrjT0ZXB0QbC"),
    "insects_reoccurring_imb": ("incremental-reoccurring_imbalanced", "1aSqdxvZvug-SwQw5NLY_9nTPhhEjB2ig"),
    "insects_incr_abrupt_imb": ("incremental-abrupt_imbalanced", "1M6QfsernUlM0qvqXdbo9bPYAsHCcfuhb"),
}
SINTETICOS = ["rbf_250k", "rbf_500k", "rbf_1m", "led_0", "led_10", "led_20", "hyperplane", "sea"]
FLUXOS = ["electricity"] + SINTETICOS[:6] + ["hyperplane", "poker", "sea"] + list(INSECTS)

# acurácia (5 limiares fixos, Z_MtL, todos os rótulos) e % de rótulos (5 fixos, Z_MtL) publicados por
# Martins et al. (2023), Tabelas 4 e 5, com duas casas como na dissertação de Martins (2022), Tabelas 11 e 12
ARTIGO = {
    "electricity": ([79.86, 79.41, 78.8, 77.71, 77.71, 78.01, 78.73], [75.79, 49.04, 34.6, 1.69, 0.45, 21.32]),
    "rbf_250k": ([97.43, 95.3, 97.74, 95.66, 74.28, 97.1, 98.44], [7.02, 4.92, 5.79, 3.87, 3.33, 4.48]),
    "rbf_500k": ([87.09, 86.57, 86.87, 86.31, 84.3, 85.67, 89.89], [56.99, 52.19, 41.18, 24.45, 16.83, 20.44]),
    "rbf_1m": ([89.07, 88.58, 88.79, 87.9, 85.74, 87.37, 91.13], [50.88, 47.5, 35.11, 18.16, 12.92, 15.27]),
    "led_0": ([99.82, 99.82, 99.82, 99.82, 99.82, 99.82, 99.99], [99.95, 99.95, 99.95, 99.95, 99.95, 99.95]),
    "led_10": ([72.87, 73.12, 72.89, 72.5, 72.84, 72.64, 73.73], [91.53, 94.38, 89.84, 60.95, 59.14, 58.17]),
    "led_20": ([51.06, 51.06, 51.07, 51.06, 51.03, 51.04, 51.12], [99.94, 99.93, 99.91, 98.7, 98.36, 99.37]),
    "hyperplane": ([87.73, 87.46, 86.48, 86.9, 87.21, 86.77, 89.16], [91.57, 88.49, 77.75, 56.48, 46.08, 54.0]),
    "poker": ([41.98, 43.18, 50.55, 45.85, 33.67, 42.78, 74.09], [6.52, 5.81, 14.04, 3.74, 5.54, 2.21]),
    "sea": ([84.83, 84.82, 84.82, 84.64, 83.2, 84.61, 84.72], [95.41, 93.24, 84.79, 38.38, 28.16, 36.01]),
    "insects_abrupt_imb": ([61.31, 60.72, 60.88, 57.48, 57.68, 60.59, 66.15], [70.88, 66.05, 67.84, 41.36, 28.76, 38.44]),
    "insects_incremental_imb": ([62.98, 63.26, 62.33, 63.22, 63.35, 62.07, 67.08], [80.14, 78.81, 79.29, 65.38, 53.82, 45.91]),
    "insects_gradual_imb": ([64.92, 65.86, 64.87, 62.81, 65.23, 64.74, 57.67], [56.09, 52.11, 47.69, 30.34, 31.16, 19.94]),
    "insects_reoccurring_imb": ([62.83, 61.8, 63.98, 60.19, 58.82, 61.62, 62.96], [81.99, 70.51, 76.14, 65.38, 47.68, 56.56]),
    "insects_incr_abrupt_imb": ([60.14, 62.34, 61.43, 58.9, 58.05, 60.19, 64.59], [80.29, 77.46, 79.44, 51.16, 49.85, 48.87]),
    "insects_abrupt_bal": ([57.45, 57.95, 57.81, 57.53, 57.67, 58.05, 53.78], [44.22, 42.51, 35.41, 27.36, 20.77, 25.29]),
    "insects_incremental_bal": ([48.39, 49.3, 49.21, 49.04, 48.19, 49.04, 52.22], [56.42, 51.43, 49.08, 38.06, 30.87, 36.58]),
    "insects_gradual_bal": ([69.62, 69.52, 71.45, 69.63, 72.06, 75.13, 60.89], [29.88, 25.04, 21.7, 15.52, 11.3, 11.88]),
    "insects_reoccurring_bal": ([55.01, 53.93, 55.02, 53.38, 53.15, 56.26, 53.31], [48.7, 44.12, 34.12, 24.06, 17.66, 20.32]),
    "insects_incr_abrupt_bal": ([57.85, 58.75, 57.29, 58.07, 50.58, 59.61, 57.66], [51.67, 45.22, 36.52, 27.02, 16.99, 16.56]),
}


# dados

def _codifica(y):
    return np.unique(np.asarray(y).astype(str), return_inverse=True)[1].astype(np.int64)


def _sintetico(nome):
    if nome.startswith("rbf"):
        n, f = {"rbf_250k": (250_000, 50), "rbf_500k": (500_000, 10), "rbf_1m": (1_000_000, 10)}[nome]
        g = RandomRBFGeneratorDrift(model_random_state=11, sample_random_state=12, n_classes=2, n_features=f,
                                    n_centroids=50, change_speed=0.0001, num_drift_centroids=50)
    elif nome.startswith("led"):
        n = 1_000_000
        g = LEDGenerator(random_state=13, noise_percentage=int(nome[4:]) / 100, has_noise=True)
    elif nome == "hyperplane":
        n = 250_000
        g = HyperplaneGenerator(random_state=14, n_features=10, n_drift_features=2, mag_change=0.0,
                                noise_percentage=0.05, sigma_percentage=0.1)
    else:
        n = 60_000
        g = SEAGenerator(classification_function=3, noise_percentage=0.1, random_state=18)
        for f in (2, 1, 0):
            g = ConceptDriftStream(stream=SEAGenerator(classification_function=f, noise_percentage=0.1, random_state=15 + f),
                                   drift_stream=g, position=15_000, width=1, random_state=20 + f)
    X, y = g.next_sample(n)
    return X, y


def _baixa_insects(arquivo, id_drive):
    destino = DADOS / "insects" / f"INSECTS_{arquivo}.csv"
    if not destino.exists():
        destino.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.usercontent.google.com/download?id={id_drive}&export=download&confirm=t"
        urllib.request.urlretrieve(url, destino.with_suffix(".parcial"))
        destino.with_suffix(".parcial").rename(destino)
    c = pd.read_csv(destino, header=None)
    X = c.iloc[:, :-1].to_numpy(float)
    X = (X - X.min(0)) / np.where(X.max(0) > X.min(0), X.max(0) - X.min(0), 1.0)
    return X, c.iloc[:, -1]


def _openml(did):
    import openml
    ds = openml.datasets.get_dataset(did, download_data=True, download_qualities=False, download_features_meta_data=False)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].cat.codes if hasattr(X[c], "cat") else pd.factorize(X[c])[0]
    return X, y


def prepara(nome):
    """Grava X e y do fluxo em dados_fluxos/ (uma vez só) e devolve o tamanho."""
    x_arq, y_arq = DADOS / f"{nome}_X.npy", DADOS / f"{nome}_y.npy"
    if not x_arq.exists():
        DADOS.mkdir(exist_ok=True)
        if nome == "electricity":
            arq = RAIZ.parent / "dados_openml" / "151_electricity.parquet"
            d = pd.read_parquet(arq) if arq.exists() else pd.concat(_openml(151), axis=1).rename(columns={"class": "target"})
            X, y = d.drop(columns=["target", "date"]).astype(float).to_numpy(), d["target"]
        elif nome == "poker":
            X, y = _openml(155)
            X = X.to_numpy(float)
        elif nome in INSECTS:
            X, y = _baixa_insects(*INSECTS[nome])
        else:
            X, y = _sintetico(nome)
        np.save(y_arq, _codifica(y))
        np.save(x_arq, np.asarray(X, dtype=np.float64))
    return len(np.load(y_arq, mmap_mode="r"))


def carrega(nome):
    return np.load(DADOS / f"{nome}_X.npy", mmap_mode="r"), np.load(DADOS / f"{nome}_y.npy")


# aprendizado ativo por incerteza (VFDT do scikit-multiflow, entropia em base 2)

def entropia(p):
    p = np.asarray(p, float).ravel()
    s = p.sum()
    if s <= 0:
        return np.inf  # sem previsão: o modelo pede o rótulo
    p = p[p > 0] / s
    return float(-(p * np.log2(p)).sum())


def ativo(X, y, z, classes, pre=PRE, mascara=False):
    m = HoeffdingTreeClassifier()
    m.partial_fit(X[:pre], y[:pre], classes=classes)
    acertos, q = 0, pre
    marca = np.zeros(len(y), bool)
    marca[:pre] = True
    for i in range(pre, len(y)):
        xi = X[i:i + 1]
        p = m.predict_proba(xi)[0]
        acertos += int(np.argmax(p)) == y[i]
        if z is None or entropia(p) >= z:
            m.partial_fit(xi, y[i:i + 1])
            q += 1
            marca[i] = True
    r = (acertos / (len(y) - pre), q / len(y))
    return r + (marca,) if mascara else r


def melhor_z(acc, rot):
    """Algoritmo 1 do artigo: o limiar com menos rótulos entre os que ficam a até MARGEM da melhor acurácia."""
    acc, rot = np.asarray(acc), np.asarray(rot)
    ok = np.where(acc >= acc.max() - MARGEM)[0]
    return int(ok[np.argmin(rot[ok])])


def posicoes(acc, rot):
    """Posição de cada candidato na ordem do Algoritmo 1 (1 = o escolhido pela regra)."""
    dentro = acc >= acc.max() - MARGEM
    ordem = sorted(range(len(acc)), key=lambda k: (not dentro[k], rot[k] if dentro[k] else -acc[k], -acc[k]))
    pos = np.empty(len(acc), int)
    pos[ordem] = np.arange(1, len(acc) + 1)
    return pos


# meta-características

FUNCS = [tsfel.feature_extraction.features.__dict__[n] for n in
         ["abs_energy", "total_energy", "autocorr", "calc_centroid", "neighbourhood_peaks", "pk_pk_distance",
          "negative_turning", "positive_turning", "slope", "zero_cross", "mean_abs_diff", "mean_diff",
          "median_abs_diff", "median_diff", "distance", "sum_abs_diff", "auc", "entropy"]]
USA_FS = [n in ("total_energy", "calc_centroid", "auc") for n in
          ["abs_energy", "total_energy", "autocorr", "calc_centroid", "neighbourhood_peaks", "pk_pk_distance",
           "negative_turning", "positive_turning", "slope", "zero_cross", "mean_abs_diff", "mean_diff",
           "median_abs_diff", "median_diff", "distance", "sum_abs_diff", "auc", "entropy"]]
T_COLS = [f"t_{f.__name__}_{a}" for a in ("media", "min", "max", "dp") for f in FUNCS]
A1_COLS = ["a1_" + c for c in [
    "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses", "NumberOfNumericFeatures", "NumberOfSymbolicFeatures",
    "NumberOfBinaryFeatures", "NumberOfMissingValues", "NumberOfInstancesWithMissingValues", "Dimensionality",
    "MajorityClassSize", "MinorityClassSize", "MajorityClassPercentage", "MinorityClassPercentage",
    "MeanMeansOfNumericAtts", "MeanStdDevOfNumericAtts", "MeanSkewnessOfNumericAtts", "MeanKurtosisOfNumericAtts",
    "ClassEntropy", "MeanAttributeEntropy", "MeanMutualInformation", "EquivalentNumberOfAtts", "NoiseToSignalRatio",
    "InstancesPerAttribute", "PercentageOfMissingValues", "PercentageOfInstancesWithMissingValues",
    "PercentageOfNumericFeatures", "PercentageOfSymbolicFeatures", "PercentageOfBinaryFeatures", "ClassImbalanceRatio",
    "NormalizedClassEntropy", "LogInstances", "LogFeatures"]]
A2_COLS = ["a2_" + c for c in [
    "ImbalanceDegree", "LRID", "EffectiveMinorityClassSize", "MinoritySafeRatio", "MinorityBorderlineRatio",
    "MinorityRareRatio", "MinorityOutlierRatio", "MinorityN3", "MajorityN3", "MeanKDN"] + list(a2.MEDIDAS)]
CONJUNTOS = {"temporais": T_COLS, "Atividade 1": A1_COLS, "Atividade 2": A2_COLS,
             "Atividades 1 e 2": A1_COLS + A2_COLS, "todas": T_COLS + A1_COLS + A2_COLS}


def mf_temporal(X):
    """As 18 meta-características do TSFEL por atributo, resumidas por média, mínimo, máximo e desvio (Tabela 1 do artigo)."""
    X = np.asarray(X, float)
    fs = 0.2 * len(X)
    M = np.zeros((X.shape[1], len(FUNCS)))
    for j in range(X.shape[1]):
        for k, (f, usa_fs) in enumerate(zip(FUNCS, USA_FS)):
            try:
                M[j, k] = float(np.ravel(f(X[:, j], fs) if usa_fs else f(X[:, j]))[0])
            except Exception:
                M[j, k] = 0.0
    M = np.nan_to_num(M, posinf=0.0, neginf=0.0)
    return dict(zip(T_COLS, np.r_[M.mean(0), M.min(0), M.max(0), M.std(0)]))


def mf_atividades(X, y):
    """Meta-características das Atividades 1 e 2, só com os exemplos rotulados; NaN quando faltam rótulos."""
    saida = dict.fromkeys(A1_COLS + A2_COLS, np.nan)
    if len(y) < MIN_ROTULOS or len(np.unique(y)) < 2:
        return saida
    Xd, yd = pd.DataFrame(np.asarray(X, float)).add_prefix("x"), pd.Series(np.asarray(y).astype(str))
    try:
        saida.update({"a1_" + k: v for k, v in a1.metafeatures(Xd, yd).items()})
    except Exception:
        pass
    try:
        Xp, yp = a1._preparar(Xd, yd)
        Z = a1._preprocessador(Xp).fit_transform(Xp).astype(float)
        cod = np.unique(yp.astype(str), return_inverse=True)[1]
        saida.update({"a2_" + k: v for k, v in a2.mf_dominio(Z, cod).items()})
        saida.update({"a2_" + k: v for k, v in a2.mf_complexidade(Z, cod)[0].items()})
    except Exception:
        pass
    return saida


def avalia_janela(X, y, classes):
    """Uma linha da meta-base: acurácia e rótulos de cada limiar candidato e as meta-características da janela."""
    pre = min(PRE, len(y) // 2)
    linha, marca = {"n": len(y)}, None
    for z, a, r in zip(CANDIDATOS, ACC, ROT):
        res = ativo(X, y, z, classes, pre=pre, mascara=z == 0.5)
        linha[a], linha[r] = res[0], res[1]
        if z == 0.5:
            marca = res[2]
    linha.update(mf_temporal(X))
    linha.update(mf_atividades(X[marca], y[marca]))
    linha["rotulados_0.5"] = int(marca.sum())
    return linha


def segmenta(X, y, classes):
    """Janelas entre detecções do ADWIN sobre os acertos de uma VFDT com todos os rótulos (Algoritmo 2)."""
    m, ad = HoeffdingTreeClassifier(), ADWIN(delta=DELTA)
    m.partial_fit(X[:PRE], y[:PRE], classes=classes)
    cortes, ini = [], 0
    for i in range(PRE, len(y)):
        xi = X[i:i + 1]
        ad.add_element(int(m.predict(xi)[0] == y[i]))
        m.partial_fit(xi, y[i:i + 1])
        if ad.detected_change():
            cortes.append((ini, i + 1))
            ini = i + 1
    return cortes


# meta-base

TIPOS = ["hyper", "led", "mixed", "rbf", "tree", "sine", "stagger"]


def _fabrica(tipo, r):
    if tipo == "hyper":
        n = r.choice([5, 10, 20]); d = r.choice([0, 2, n // 2]); m = r.choice([0.0, 0.001, 0.01]); ru = r.choice([0.0, 0.05, 0.1])
        return lambda s, i: HyperplaneGenerator(random_state=s, n_features=n, n_drift_features=d, mag_change=m,
                                                noise_percentage=ru, sigma_percentage=0.1)
    if tipo == "led":
        ru = r.choice([0.0, 0.1, 0.2]); d = r.randint(0, 7)
        return lambda s, i: LEDGeneratorDrift(random_state=s, noise_percentage=ru, has_noise=True, n_drift_features=(d + 2 * i) % 8)
    if tipo == "mixed":
        f = r.randint(0, 1)
        return lambda s, i: MIXEDGenerator(classification_function=(f + i) % 2, random_state=s)
    if tipo == "rbf":
        c = r.choice([2, 3, 5]); n = r.choice([5, 10, 20]); k = r.choice([10, 50]); v = r.choice([0.0, 0.0001, 0.001]); d = r.choice([5, 10])
        return lambda s, i: RandomRBFGeneratorDrift(model_random_state=s, sample_random_state=s + 1, n_classes=c, n_features=n,
                                                    n_centroids=k, change_speed=v, num_drift_centroids=d)
    if tipo == "tree":
        c = r.choice([2, 3, 5]); nc = r.choice([0, 2, 5]); nn = r.choice([3, 5, 10])
        return lambda s, i: RandomTreeGenerator(tree_random_state=s, sample_random_state=s + 1, n_classes=c, n_cat_features=nc,
                                                n_num_features=nn, n_categories_per_cat_feature=3, max_tree_depth=5,
                                                min_leaf_depth=3, fraction_leaves_per_level=0.15)
    if tipo == "sine":
        f = r.randint(0, 3); ru = r.random() < 0.5
        return lambda s, i: SineGenerator(classification_function=(f + i) % 4, random_state=s, has_noise=ru)
    f = r.randint(0, 2)
    return lambda s, i: STAGGERGenerator(classification_function=(f + i) % 3, random_state=s)


def gera_fluxo_meta(i, n=30_000):
    """Fluxo sintético com 1 a 3 drifts abruptos, graduais ou recorrentes, com gerador e parâmetros sorteados."""
    tipo, r = TIPOS[i % 7], random.Random(1000 + i)
    criar = _fabrica(tipo, r)
    modo = r.choice(["abrupto", "gradual", "recorrente"])
    k = r.randint(1, 3)
    gaps = [r.randint(n // 8, n // (k + 1)) for _ in range(k)]
    ids = [j % 2 for j in range(k + 1)] if modo == "recorrente" else list(range(k + 1))
    atual = criar(r.randrange(10 ** 6), ids[-1])
    for j in range(k - 1, -1, -1):
        w = 1 if modo != "gradual" else r.choice([1000, 3000, 5000])
        atual = ConceptDriftStream(stream=criar(r.randrange(10 ** 6), ids[j]), drift_stream=atual, position=gaps[j],
                                   width=w, random_state=r.randrange(10 ** 6))
    X, y = atual.next_sample(n)
    return tipo, modo, X.astype(float), y.astype(int).ravel()


def janelas_meta(i):
    tipo, modo, X, y = gera_fluxo_meta(i)
    classes = list(range(int(y.max()) + 1))
    linhas = []
    for a, b in segmenta(X, y, classes):
        if b - a >= JANELA_MIN:
            linhas.append(dict(fluxo=i, gerador=tipo, drift=modo, ini=a, **avalia_janela(X[a:b], y[a:b], classes)))
    return linhas


# janelas dos fluxos de avaliação

def cortes_avaliacao(nome):
    X, y = carrega(nome)
    cortes = [(max(a, b - JANELA_MAX), b) for a, b in segmenta(X, y, list(range(int(y.max()) + 1))) if b - a >= JANELA_MIN]
    if len(cortes) > JANELAS_POR_FLUXO:
        cortes = [cortes[k] for k in np.linspace(0, len(cortes) - 1, JANELAS_POR_FLUXO).round().astype(int)]
    return cortes


def janela_avaliacao(nome, a, b):
    X, y = carrega(nome)
    Xw, yw = np.asarray(X[a:b], float), y[a:b]
    return dict(fluxo=nome, ini=a, **avalia_janela(Xw, yw, list(range(int(y.max()) + 1))))


# meta-modelos

def alvo(base):
    return np.array([CANDIDATOS[melhor_z(a, r)] for a, r in zip(base[ACC].to_numpy(), base[ROT].to_numpy())])


def treina(base, metodo, conjunto):
    X = base[CONJUNTOS[conjunto]].to_numpy(float)
    y = alvo(base) if metodo == "regressão" else base[ACC + ROT].to_numpy(float)
    modelo = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(random_state=0, n_jobs=1))
    return dict(metodo=metodo, conjunto=conjunto, modelo=modelo.fit(X, y))


def recomenda(meta, mf):
    """Limiar recomendado para cada linha de meta-características (DataFrame)."""
    X = mf[CONJUNTOS[meta["conjunto"]]].to_numpy(float)
    previsto = meta["modelo"].predict(X)
    if meta["metodo"] == "regressão":
        return previsto
    k = len(CANDIDATOS)
    return np.array([CANDIDATOS[melhor_z(p[:k], p[k:])] for p in previsto])


def indice_candidato(z):
    return np.abs(np.array(CANDIDATOS)[None, :] - np.asarray(z, float)[:, None]).argmin(1)


def metricas_janelas(base, indices):
    """Distância da escolha até a regra do artigo, janela a janela."""
    A, R = base[ACC].to_numpy(), base[ROT].to_numpy()
    w = np.arange(len(base))
    pos = np.array([posicoes(a, r)[k] for a, r, k in zip(A, R, indices)])
    return pd.DataFrame({"fluxo": base["fluxo"].to_numpy(), "posição": pos,
                         "perda de acurácia (p.p.)": 100 * (A.max(1) - A[w, indices]),
                         "rótulos (%)": 100 * R[w, indices],
                         "dentro da margem": A[w, indices] >= A.max(1) - MARGEM})


# implantação no fluxo (Algoritmo 3)

def implanta(nome, config, meta=None):
    """Aprendizado ativo no fluxo inteiro: limiar fixo, todos os rótulos ou limiar recomendado a cada detecção do ADWIN."""
    X, y = carrega(nome)
    classes = list(range(int(y.max()) + 1))
    m, ad = HoeffdingTreeClassifier(), ADWIN(delta=DELTA)
    m.partial_fit(np.asarray(X[:PRE], float), y[:PRE], classes=classes)
    todos = config["tipo"] == "todos"
    Z = config.get("z", 0.5)
    acertos, q, ini, zs, mantidos, t_mf = 0, PRE, 0, [], 0, 0.0
    marca = np.zeros(len(y), bool)
    marca[:PRE] = True
    inicio = time.perf_counter()
    for i in range(PRE, len(y)):
        xi = np.asarray(X[i:i + 1], float)
        p = m.predict_proba(xi)[0]
        certo = int(np.argmax(p)) == y[i]
        acertos += certo
        if todos or entropia(p) >= Z:
            m.partial_fit(xi, y[i:i + 1])
            q += 1
            marca[i] = True
        if meta is not None:
            ad.add_element(int(certo))
            if ad.detected_change():
                a = max(ini, i + 1 - JANELA_MAX)
                if i + 1 - ini >= 30:
                    t = time.perf_counter()
                    Xw = np.asarray(X[a:i + 1], float)
                    if meta["conjunto"] == "temporais":
                        mf = mf_temporal(Xw)
                    else:
                        mf = mf_atividades(Xw[marca[a:i + 1]], y[a:i + 1][marca[a:i + 1]])
                        if meta["conjunto"] == "todas":
                            mf.update(mf_temporal(Xw))
                    if meta["conjunto"] == "temporais" or not np.isnan(mf[A1_COLS[0]]):
                        Z = float(recomenda(meta, pd.DataFrame([mf]))[0])
                        zs.append(Z)
                    else:
                        mantidos += 1
                    t_mf += time.perf_counter() - t
                ini = i + 1
    return dict(fluxo=nome, config=config["nome"], acc=100 * acertos / (len(y) - PRE), rot=100 * q / len(y),
                trocas=len(zs), mantidos=mantidos, z_medio=np.mean(zs) if zs else Z, z_min=min(zs) if zs else Z,
                z_max=max(zs) if zs else Z, segundos=time.perf_counter() - inicio, segundos_mf=t_mf)


# execução em paralelo com cache

def _executa(funcao, pasta, chave, args):
    warnings.filterwarnings("ignore")
    resultado = funcao(*args)
    tmp = pasta / f"{chave}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(resultado, f)
    tmp.rename(pasta / f"{chave}.pkl")


def roda(secao, tarefas, funcao, n_jobs=N_JOBS):
    """Roda funcao(*args) para cada (chave, args) que ainda não tem resultado em cache/secao/ e devolve todos."""
    pasta = CACHE / secao
    pasta.mkdir(parents=True, exist_ok=True)
    pendentes = [(k, a) for k, a in tarefas if not (pasta / f"{k}.pkl").exists()]
    if pendentes:
        Parallel(n_jobs=n_jobs)(delayed(_executa)(funcao, pasta, k, a) for k, a in pendentes)
    return [pickle.load(open(pasta / f"{k}.pkl", "rb")) for k, _ in tarefas]


# Nemenyi para até 14 grupos (Demšar, 2006)
Q_NEMENYI = {**a1.Q_NEMENYI, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354}


def diferenca_critica(k, n):
    return Q_NEMENYI[k] * np.sqrt(k * (k + 1) / (6 * n))
