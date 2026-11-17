# IN1097 · Meta-aprendizado

Gabriel Oliveira Gonçalves · Tópicos Avançados em Agentes Inteligentes 2

Meta-dataset de seleção de algoritmos com 33 datasets de classificação do OpenML, 6 algoritmos do
scikit-learn e AUC em validação cruzada de 10 folds. Os notebooks já estão com as saídas.

| Atividade | Notebook |
|---|---|
| 1. Meta-dataset e ranking de algoritmos | [`coleta_openml.ipynb`](coleta_openml.ipynb) |
| 2. Meta-características específicas do domínio | [`atividade2/metacaracteristicas_dominio.ipynb`](atividade2/metacaracteristicas_dominio.ipynb) |
| 3. Warm-starting da otimização de hiperparâmetros | [`atividade3/warmstart_hpo.ipynb`](atividade3/warmstart_hpo.ipynb) |
| 4. AutoML versus meta-aprendizagem | [`atividade4/automl_vs_meta.ipynb`](atividade4/automl_vs_meta.ipynb) |
| Projeto final. Meta-aprendizado para o aprendizado ativo em fluxos | [`projeto/meta_aprendizado_fluxos.ipynb`](projeto/meta_aprendizado_fluxos.ipynb) |

A Atividade 3 fixa o hgb e compara formas de começar a busca de hiperparâmetros em 29 desses datasets,
sobre uma matriz de perdas de 129 configurações × 29 datasets × 5 folds
([`atividade3/matriz_perdas.csv`](atividade3/matriz_perdas.csv)).

A Atividade 4 compara o Auto-sklearn 0.15, com 60, 300 e 900 s por dataset, com as soluções das
Atividades 1 a 3 nos mesmos 29 datasets, numa partição de teste de 20%. Os resultados brutos das 261
execuções do Auto-sklearn estão em [`atividade4/automl/`](atividade4/automl/), e o resumo por dataset, em
[`atividade4/automl_resultados.csv`](atividade4/automl_resultados.csv).

O projeto final reproduz Martins, Cano e Barbon Jr. (2023, *Pattern Recognition*), em que um meta-modelo
escolhe o limiar de incerteza do aprendizado ativo a cada drift, em 20 fluxos com concept drift. Ele compara
esse meta-modelo com referências simples, com um meta-modelo que prevê o desempenho de cada limiar e com as
meta-características das Atividades 1 e 2.

## Como rodar

```bash
pip install -r requirements.txt
```

1. `coleta_openml.ipynb` baixa os datasets para `dados_openml/` (277 MB, fora do repositório). Os
   experimentos já feitos ficam nos CSVs e não são refeitos.
2. `atividade2/metacaracteristicas_dominio.ipynb` precisa de `dados_openml/`. Dá para rodar pelo
   Jupyter ou com `python3 atividade2/.build/executar.py`, que salva as saídas no notebook. Só os
   testes do hgb rodam de novo (uns 25 minutos); o resto vem dos CSVs.
3. `atividade3/warmstart_hpo.ipynb` só precisa de `dados_openml/` para refazer a matriz de perdas
   (33 horas de CPU, 98 minutos com 22 processos). Com a matriz e as buscas já gravadas
   (`perdas_folds.csv`, `buscas.csv`), `python3 atividade3/.build/executar.py` roda tudo em segundos.
4. `projeto/` precisa de Python 3.8, porque o scikit-multiflow 0.5.3, a biblioteca do artigo, não instala
   em versões mais novas: `pip install -r projeto/requirements_py38.txt` num ambiente à parte. O notebook
   baixa ou gera os fluxos em `dados_fluxos/` (2 GB, fora do repositório). Os resultados pesados ficam em
   `projeto/cache/` e são reaproveitados, então `python projeto/.build/executar.py` roda em poucos minutos;
   sem o cache, são cerca de 2,5 horas com 16 processos.
5. `atividade4/` usa o Auto-sklearn 0.15, que precisa de Python 3.9 e de versões antigas do scikit-learn.
   Ele fica num ambiente à parte em `atividade4/.venv`, criado com o [uv](https://docs.astral.sh/uv/):

   ```bash
   cd atividade4
   uv venv --python 3.9 .venv
   uv pip install "setuptools<70" swig wheel
   uv pip install --no-build-isolation-package pyrfr -r requirements_autosklearn.txt
   ```

   O notebook usa esse ambiente nas seções 1.2, 1.3 e 3, e o resto roda no ambiente principal. Com as
   execuções já gravadas em `automl/`, `python3 atividade4/.build/executar.py` roda tudo em segundos.
   Refazer as 261 execuções leva cerca de 4 horas com 8 processos, e o computador não pode suspender no
   meio, porque o orçamento do Auto-sklearn é contado no relógio.

A Atividade 2 usa P, R e X_base da Atividade 1 (cópia em `atividade2/base/`) e as funções dela em
`atividade2/atividade1.py`, e reproduz os resultados da Atividade 1 exatamente.

Os resultados dependem das versões do `requirements.txt`. Com várias threads o hgb não é
determinístico no shuttle e no page-blocks, então refazer a validação cruzada da Atividade 1 muda
essas duas células de P (Quadro 17 da Atividade 2).
