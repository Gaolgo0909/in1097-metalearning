"""Uma execução do Auto-sklearn 0.15 (ambiente .venv, Python 3.9) num dataset e num orçamento."""
import argparse
import json
import resource
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import autosklearn
import autosklearn.classification
import autosklearn.metrics

parser = argparse.ArgumentParser()
parser.add_argument("--particao", required=True)
parser.add_argument("--saida", required=True)
parser.add_argument("--orcamento", type=int, required=True)
parser.add_argument("--variante", choices=["sem_nossos", "original", "sem_meta"], required=True)
parser.add_argument("--metadados", default=None)
parser.add_argument("--temporaria", required=True)
parser.add_argument("--semente", type=int, default=1)
args = parser.parse_args()

particao, saida, temporaria = Path(args.particao), Path(args.saida), Path(args.temporaria)
X_treino = pd.read_parquet(particao / "X_treino.parquet")
X_teste = pd.read_parquet(particao / "X_teste.parquet")
y_treino = np.load(particao / "y_treino.npy")

for coluna in X_treino.columns:
    if pd.api.types.is_numeric_dtype(X_treino[coluna]) and not pd.api.types.is_bool_dtype(X_treino[coluna]):
        X_treino[coluna] = X_treino[coluna].astype(float)
        X_teste[coluna] = X_teste[coluna].astype(float)
    else:
        treino = X_treino[coluna].astype(str).where(X_treino[coluna].notna())
        categorias = sorted(treino.dropna().unique())
        X_treino[coluna] = pd.Categorical(treino, categories=categorias)
        X_teste[coluna] = pd.Categorical(X_teste[coluna].astype(str).where(X_teste[coluna].notna()),
                                         categories=categorias)

binario = len(np.unique(y_treino)) == 2
metrica = autosklearn.metrics.roc_auc if binario else autosklearn.metrics.log_loss

shutil.rmtree(temporaria, ignore_errors=True)
saida.mkdir(parents=True, exist_ok=True)
resultado = {"variante": args.variante, "orcamento": args.orcamento, "semente": args.semente,
             "metrica": metrica.name, "versao": autosklearn.__version__}
try:
    modelo = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=args.orcamento,
        n_jobs=1,
        memory_limit=3072,
        seed=args.semente,
        metric=metrica,
        initial_configurations_via_metalearning=0 if args.variante == "sem_meta" else 25,
        metadata_directory=args.metadados if args.variante == "sem_nossos" else None,
        tmp_folder=str(temporaria),
        delete_tmp_folder_after_terminate=True,
    )
    inicio, relogio = time.perf_counter(), time.time()
    modelo.fit(X_treino, y_treino)
    resultado["tempo_treino"] = time.perf_counter() - inicio
    # o orçamento do Auto-sklearn é medido no relógio de parede; se o computador suspende, o relógio pula
    resultado["relogio_pulou"] = abs(time.time() - relogio - resultado["tempo_treino"]) > 5

    inicio = time.perf_counter()
    proba = modelo.predict_proba(X_teste)
    resultado["tempo_predicao"] = time.perf_counter() - inicio
    np.save(saida / "proba_teste.npy", proba)

    try:
        todos = modelo.leaderboard(detailed=True, ensemble_only=False)
    except KeyError:
        # bug do 0.15: se o ensemble usa um modelo que não está no runhistory, faltam as colunas de pré-processamento
        todos = modelo.leaderboard(ensemble_only=False, include=[
            "model_id", "rank", "ensemble_weight", "type", "cost", "duration", "config_id", "train_loss", "seed",
            "start_time", "end_time", "budget", "status", "config_origin"])
    todos.to_csv(saida / "modelos.csv")
    resultado["modelos_avaliados"] = len(todos)
    resultado["modelos_no_ensemble"] = int((todos["ensemble_weight"] > 0).sum())
    resultado["estatisticas"] = modelo.sprint_statistics()
except Exception:
    resultado["erro"] = traceback.format_exc()[-2000:]

uso = [resource.getrusage(r) for r in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
resultado["cpu"] = sum(u.ru_utime + u.ru_stime for u in uso)
(saida / "resultado.json").write_text(json.dumps(resultado, indent=1, ensure_ascii=False))
