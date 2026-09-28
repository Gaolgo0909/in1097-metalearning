"""Codigo da Atividade 1, extraido literalmente de ../coleta_openml.ipynb.

Nada aqui foi reescrito: cada definicao e copiada do notebook da Atividade 1 via ast,
para que a Atividade 2 rode exatamente o mesmo protocolo. Gerado por .build/extrair_atividade1.py.
"""
from collections import namedtuple
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ---- configuracao dos experimentos ----

N_FOLDS = 10

N_MAX = 10_000

SEMENTE = 42

ALGORITMOS = {
    "logreg": LogisticRegression(max_iter=1000),
    "nb": GaussianNB(),
    "knn": KNeighborsClassifier(n_jobs=-1),
    "tree": DecisionTreeClassifier(random_state=SEMENTE),
    "rf": RandomForestClassifier(n_estimators=100, random_state=SEMENTE, n_jobs=-1),
    "hgb": HistGradientBoostingClassifier(random_state=SEMENTE),
}


# ---- preparacao dos dados e AUC ----

def _preparar(X, y):
    if len(y) > N_MAX:
        X, _, y, _ = train_test_split(X, y, train_size=N_MAX, stratify=y, random_state=SEMENTE)
    frequentes = y.value_counts()  # depois da subamostragem: uma classe rara pode encolher abaixo de N_FOLDS
    y = y[y.isin(frequentes[frequentes >= N_FOLDS].index)]
    X = X.loc[y.index]
    return X.reset_index(drop=True), y.reset_index(drop=True)

def _preprocessador(X):
    numericos = X.select_dtypes(include="number").columns
    simbolicos = X.columns.difference(numericos)
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("esc", StandardScaler())]), numericos),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), simbolicos),
    ])

def _auc(y_teste, probabilidades, classes):
    if len(classes) == 2:
        return roc_auc_score(y_teste, probabilidades[:, 1])
    return roc_auc_score(y_teste, probabilidades, multi_class="ovr",
                         average="macro", labels=classes)


# ---- meta-caracteristicas do X_base ----

def _entropia(s):
    p = s.value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum())

def _entropia_conjunta(a, b):
    p = pd.DataFrame({"a": a, "b": b}).value_counts(normalize=True)
    return float(-(p * np.log2(p)).sum())

def _discretizar(coluna, faixas=10):
    if pd.api.types.is_numeric_dtype(coluna) and coluna.nunique() > faixas:
        codigos = pd.qcut(coluna, faixas, labels=False, duplicates="drop")
        return codigos.fillna(-1).astype("int16")
    return coluna.astype("category").cat.codes.astype("int16")


# ---- meta-caracteristicas do X_base (familias) ----

def mf_gerais(X, y):
    n, p = X.shape
    numericos = X.select_dtypes(include="number").shape[1]
    binarios = int((X.nunique(dropna=True) == 2).sum()) + int(y.nunique() == 2)
    return {
        "NumberOfInstances": n,
        "NumberOfFeatures": p + 1,
        "NumberOfClasses": int(y.nunique()),
        "NumberOfNumericFeatures": numericos,
        "NumberOfSymbolicFeatures": p - numericos + 1,
        "NumberOfBinaryFeatures": binarios,
        "NumberOfMissingValues": int(X.isna().sum().sum() + y.isna().sum()),
        "NumberOfInstancesWithMissingValues": int(X.isna().any(axis=1).sum()),
        "Dimensionality": (p + 1) / n,
    }

def mf_balanceamento(y):
    contagem = y.value_counts()
    contagem = contagem[contagem > 0]  # alvo categorico pode declarar classes sem nenhuma instancia
    return {
        "MajorityClassSize": int(contagem.max()),
        "MinorityClassSize": int(contagem.min()),
        "MajorityClassPercentage": 100 * contagem.max() / len(y),
        "MinorityClassPercentage": 100 * contagem.min() / len(y),
    }

def mf_estatisticas(X):
    num = X.select_dtypes(include="number")
    if num.empty:
        return dict.fromkeys(["MeanMeansOfNumericAtts", "MeanStdDevOfNumericAtts",
                              "MeanSkewnessOfNumericAtts", "MeanKurtosisOfNumericAtts"], np.nan)
    return {
        "MeanMeansOfNumericAtts": float(num.mean().mean()),
        "MeanStdDevOfNumericAtts": float(num.std().mean()),
        "MeanSkewnessOfNumericAtts": float(num.skew().mean()),
        "MeanKurtosisOfNumericAtts": float(num.kurtosis().mean()),
    }

def mf_informacao(X, y, faixas=10):
    alvo = y.astype("category").cat.codes
    h_classe = _entropia(alvo)

    entropias, informacoes = [], []
    for coluna in X.columns:
        atributo = _discretizar(X[coluna], faixas)
        h_atributo = _entropia(atributo)
        entropias.append(h_atributo)
        informacoes.append(h_atributo + h_classe - _entropia_conjunta(atributo, alvo))

    h_media, mi_media = float(np.mean(entropias)), float(np.mean(informacoes))
    return {
        "ClassEntropy": h_classe,
        "MeanAttributeEntropy": h_media,
        "MeanMutualInformation": mi_media,
        "EquivalentNumberOfAtts": h_classe / mi_media if mi_media > 0 else np.nan,
        "NoiseToSignalRatio": (h_media - mi_media) / mi_media if mi_media > 0 else np.nan,
    }

def mf_derivadas(mf):
    n, p = mf["NumberOfInstances"], mf["NumberOfFeatures"]
    return {
        "InstancesPerAttribute": n / p,
        "PercentageOfMissingValues": 100 * mf["NumberOfMissingValues"] / (n * p),
        "PercentageOfInstancesWithMissingValues": 100 * mf["NumberOfInstancesWithMissingValues"] / n,
        "PercentageOfNumericFeatures": 100 * mf["NumberOfNumericFeatures"] / p,
        "PercentageOfSymbolicFeatures": 100 * mf["NumberOfSymbolicFeatures"] / p,
        "PercentageOfBinaryFeatures": 100 * mf["NumberOfBinaryFeatures"] / p,
        "ClassImbalanceRatio": mf["MajorityClassSize"] / mf["MinorityClassSize"],
        "NormalizedClassEntropy": mf["ClassEntropy"] / np.log2(mf["NumberOfClasses"]),
        "LogInstances": np.log2(n),
        "LogFeatures": np.log2(p),
    }

def metafeatures(X, y):
    mf = {}
    mf.update(mf_gerais(X, y))
    mf.update(mf_balanceamento(y))
    mf.update(mf_estatisticas(X))
    mf.update(mf_informacao(X, y))
    mf.update(mf_derivadas(mf))
    return mf


# ---- utilidades do meta-nivel ----

Rodada = namedtuple("Rodada", "Xtr Ptr Rtr Vtr xte")

def _ranquear(v):
    ordem = np.argsort(v, kind="stable")
    ranks = np.empty(len(v))
    ranks[ordem] = np.arange(1.0, len(v) + 1)
    _, inversa, contagem = np.unique(v, return_inverse=True, return_counts=True)
    if len(contagem) < len(v):  # empates recebem o rank medio
        ranks = (np.bincount(inversa, weights=ranks) / contagem)[inversa]
    return ranks


# ---- agregacao de rankings ----

def ar(d):
    return _ranquear(d.Rtr.mean(axis=0))

def mr(d):
    return _ranquear(np.median(d.Rtr, axis=0))

def vs(d):
    return _ranquear(-d.Vtr.sum(axis=0))


# ---- meta-modelos treinados ----

def _floresta(d, alvo):
    modelo = RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                   random_state=SEMENTE, n_jobs=-1)
    modelo.fit(d.Xtr, alvo)
    return modelo.predict(d.xte.reshape(1, -1))[0]

def reg_p(d):
    return _ranquear(-_floresta(d, d.Ptr))

def reg_r(d):
    return _ranquear(_floresta(d, d.Rtr))


# ---- HARRIS ----

def _perda_regressao(Pm):
    return float(((Pm - Pm.mean(axis=0)) ** 2).mean())

def _perda_ranking(Rm):
    consenso = _ranquear(Rm.mean(axis=0))
    a = Rm - Rm.mean(axis=1, keepdims=True)
    b = consenso - consenso.mean()
    den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum())
    rho = np.divide(a @ b, den, out=np.zeros(len(Rm)), where=den > 0)
    return float(np.mean(1 - rho))

class ArvoreHarris:
    def __init__(self, lam, min_folha=3, max_prof=3, mtry=5, n_cortes=10, semente=0):
        self.lam, self.min_folha, self.max_prof = lam, min_folha, max_prof
        self.mtry, self.n_cortes = mtry, n_cortes
        self.rng = np.random.default_rng(semente)

    def treinar(self, X, Pnorm, R, idx):
        self.X, self.Pnorm, self.R = X, Pnorm, R
        self.escala = (max(_perda_regressao(Pnorm[idx]), 1e-12),
                       max(_perda_ranking(R[idx]), 1e-12))
        self.raiz = self._crescer(idx, self.max_prof)
        return self

    def _custo(self, idx):
        return (self.lam * _perda_ranking(self.R[idx]) / self.escala[1]
                + (1 - self.lam) * _perda_regressao(self.Pnorm[idx]) / self.escala[0])

    def _crescer(self, idx, prof):
        if prof == 0 or len(idx) < 2 * self.min_folha:
            return self.R[idx].mean(axis=0)

        melhor, corte = np.inf, None
        for f in self.rng.choice(self.X.shape[1], self.mtry, replace=False):
            valores = np.unique(self.X[idx, f])
            if len(valores) < 2:
                continue
            for limiar in np.unique(np.quantile(valores, np.linspace(0, 1, self.n_cortes + 2)[1:-1])):
                esquerda = self.X[idx, f] <= limiar
                if esquerda.sum() < self.min_folha or (~esquerda).sum() < self.min_folha:
                    continue
                custo = (esquerda.sum() * self._custo(idx[esquerda])
                         + (~esquerda).sum() * self._custo(idx[~esquerda])) / len(idx)
                if custo < melhor:
                    melhor, corte = custo, (f, limiar, esquerda)

        if corte is None:
            return self.R[idx].mean(axis=0)
        f, limiar, esquerda = corte
        return (f, limiar,
                self._crescer(idx[esquerda], prof - 1),
                self._crescer(idx[~esquerda], prof - 1))

    def prever(self, x):
        no = self.raiz
        while isinstance(no, tuple):
            no = no[2] if x[no[0]] <= no[1] else no[3]
        return no

class FlorestaHarris:
    def __init__(self, lam, n_arvores=100, semente=SEMENTE, **kw):
        self.lam, self.n_arvores, self.semente, self.kw = lam, n_arvores, semente, kw

    def treinar(self, X, Pbruto, R):
        piso = Pbruto.min(axis=1, keepdims=True)
        amplitude = Pbruto.max(axis=1, keepdims=True) - piso
        Pnorm = (Pbruto - piso) / np.where(amplitude > 0, amplitude, 1.0)

        rng = np.random.default_rng(self.semente)
        self.arvores = []
        for _ in range(self.n_arvores):
            bolsa = rng.choice(len(X), len(X), replace=True)
            arvore = ArvoreHarris(self.lam, semente=int(rng.integers(1 << 31)), **self.kw)
            self.arvores.append(arvore.treinar(X, Pnorm, R, bolsa))
        return self

    def prever(self, x):
        return _ranquear(np.mean([a.prever(x) for a in self.arvores], axis=0))

def harris(d, lam):
    return FlorestaHarris(lam).treinar(d.Xtr, d.Ptr, d.Rtr).prever(d.xte)


# ---- lambdas ----

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]

LAMBDA_DC = 0.5


# ---- cores dos graficos ----
TINTA, TINTA_2, GRADE, SUPERFICIE = "#0b0b0b", "#52514e", "#e5e4df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


# ---- diagrama de diferenca critica ----

Q_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
             7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

def diagrama_dc(ranks_medios, cd, titulo):
    nomes = np.array(ranks_medios.index)[np.argsort(ranks_medios.to_numpy())]
    valores = np.sort(ranks_medios.to_numpy())
    k = len(nomes)
    inicio, fim = np.floor(valores.min() * 2) / 2, np.ceil(valores.max() * 2) / 2
    metade, passo = (k + 1) // 2, 0.34

    figura, eixo = plt.subplots(figsize=(9, 1.9 + passo * metade))
    eixo.set_xlim(inicio - 1.9, fim + 1.9)
    eixo.set_ylim(-passo * (metade + 1.4), 0.85)
    eixo.axis("off")

    eixo.plot([inicio, fim], [0, 0], color=TINTA, lw=1.2)
    for marca in np.arange(inicio, fim + 1e-9, 0.5):
        eixo.plot([marca, marca], [0, 0.07], color=TINTA, lw=1.0)
        eixo.text(marca, 0.13, f"{marca:g}", ha="center", va="bottom", fontsize=8.5, color=TINTA_2)

    for j, (nome, valor) in enumerate(zip(nomes, valores)):
        esquerda = j < metade
        y = -passo * ((j + 1) if esquerda else (k - j))
        ponta = inicio - 0.15 if esquerda else fim + 0.15
        eixo.plot([valor, valor, ponta], [0, y, y], color=TINTA_2, lw=1.0)
        eixo.text(ponta + (-0.12 if esquerda else 0.12), y, f"{nome} ({valor:.2f})",
                  ha="right" if esquerda else "left", va="center", fontsize=9.5, color=TINTA)

    grupos = []
    for i in range(k):
        j = i
        while j + 1 < k and valores[j + 1] - valores[i] <= cd:
            j += 1
        if j > i:
            grupos.append((i, j))
    grupos = [g for g in grupos
              if not any(h != g and h[0] <= g[0] and g[1] <= h[1] for h in grupos)]

    for nivel, (i, j) in enumerate(grupos):
        y = -passo * (0.42 + 0.30 * nivel)
        eixo.plot([valores[i] - 0.03, valores[j] + 0.03], [y, y], color=TINTA, lw=3.5,
                  solid_capstyle="round")

    eixo.plot([inicio, inicio + cd], [0.52, 0.52], color=TINTA_2, lw=1.6)
    for ponta in (inicio, inicio + cd):
        eixo.plot([ponta, ponta], [0.47, 0.57], color=TINTA_2, lw=1.6)
    eixo.text(inicio + cd / 2, 0.62, f"CD = {cd:.2f}", ha="center", fontsize=9, color=TINTA_2)
    eixo.set_title(titulo, loc="left", fontsize=12, pad=6)
    figura.tight_layout()
    return figura
