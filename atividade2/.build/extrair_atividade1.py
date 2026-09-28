"""Gera atividade1.py copiando literalmente as definicoes do notebook da Atividade 1."""
import ast, json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
nb = json.load(open(RAIZ.parent / "coleta_openml.ipynb"))
fontes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def celula(prefixo):
    achadas = [s for s in fontes if s.lstrip().startswith(prefixo)]
    assert len(achadas) == 1, (prefixo, len(achadas))
    return achadas[0]


BLOCOS = [
    ("configuracao dos experimentos", "from sklearn.base import clone", ["N_FOLDS", "N_MAX", "SEMENTE", "ALGORITMOS"]),
    ("preparacao dos dados e AUC", "def _preparar", ["_preparar", "_preprocessador", "_auc"]),
    ("meta-caracteristicas do X_base", "import numpy as np", ["_entropia", "_entropia_conjunta", "_discretizar"]),
    ("meta-caracteristicas do X_base (familias)", "def mf_gerais",
     ["mf_gerais", "mf_balanceamento", "mf_estatisticas", "mf_informacao", "mf_derivadas", "metafeatures"]),
    ("utilidades do meta-nivel", "from collections import namedtuple", ["Rodada", "_ranquear"]),
    ("agregacao de rankings", "def ar(d)", ["ar", "mr", "vs"]),
    ("meta-modelos treinados", "def _floresta", ["_floresta", "reg_p", "reg_r"]),
    ("HARRIS", "def _perda_regressao", ["_perda_regressao", "_perda_ranking", "ArvoreHarris", "FlorestaHarris", "harris"]),
    ("lambdas", "LAMBDAS = ", ["LAMBDAS", "LAMBDA_DC"]),
    ("diagrama de diferenca critica", "Q_NEMENYI = ", ["Q_NEMENYI", "diagrama_dc"]),
]

CABECALHO = '''"""Codigo da Atividade 1, extraido literalmente de ../coleta_openml.ipynb.

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
'''

partes = [CABECALHO]
for titulo, prefixo, nomes in BLOCOS:
    fonte, pendentes = celula(prefixo), list(nomes)
    partes.append(f"\n\n# ---- {titulo} ----\n")
    for no in ast.parse(fonte).body:
        if isinstance(no, (ast.FunctionDef, ast.ClassDef)):
            nome = no.name
        elif isinstance(no, ast.Assign) and isinstance(no.targets[0], ast.Name):
            nome = no.targets[0].id
        else:
            continue
        if nome in pendentes:
            partes.append("\n" + ast.get_source_segment(fonte, no) + "\n")
            pendentes.remove(nome)
    assert not pendentes, (titulo, pendentes)

cores = celula("import matplotlib.pyplot as plt").splitlines()
partes.append("\n\n# ---- cores dos graficos ----\n")
partes += [l + "\n" for l in cores if l.startswith(("TINTA, TINTA_2", "SERIES = "))]

# o diagrama DC usa as cores, entao elas precisam vir antes dele
texto = "".join(partes)
bloco_cores = texto[texto.index("\n\n# ---- cores dos graficos ----"):]
texto = texto[:texto.index("\n\n# ---- cores dos graficos ----")]
texto = texto.replace("\n\n# ---- diagrama de diferenca critica ----", bloco_cores + "\n\n# ---- diagrama de diferenca critica ----")

destino = RAIZ / "atividade1.py"
destino.write_text(texto)
compile(texto, str(destino), "exec")
print(f"{destino.name}: {len(texto.splitlines())} linhas")
