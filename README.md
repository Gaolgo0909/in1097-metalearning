# IN1097 · Meta-aprendizado

Gabriel Oliveira Gonçalves · Tópicos Avançados em Agentes Inteligentes 2

Meta-dataset de seleção de algoritmos com 33 datasets de classificação do OpenML, 6 algoritmos do
scikit-learn e AUC em validação cruzada de 10 folds. Os notebooks já estão com as saídas.

| Atividade | Notebook |
|---|---|
| 1. Meta-dataset e ranking de algoritmos | [`coleta_openml.ipynb`](coleta_openml.ipynb) |
| 2. Meta-características específicas do domínio | [`atividade2/metacaracteristicas_dominio.ipynb`](atividade2/metacaracteristicas_dominio.ipynb) |

## Como rodar

```bash
pip install -r requirements.txt
```

1. `coleta_openml.ipynb` baixa os datasets para `dados_openml/` (277 MB, fora do repositório). Os
   experimentos já feitos ficam nos CSVs e não são refeitos.
2. `atividade2/metacaracteristicas_dominio.ipynb` precisa de `dados_openml/`. Dá para rodar pelo
   Jupyter ou com `python3 atividade2/.build/executar.py`, que salva as saídas no notebook. Só os
   testes do hgb rodam de novo (uns 25 minutos); o resto vem dos CSVs.

A Atividade 2 usa P, R e X_base da Atividade 1 (cópia em `atividade2/base/`) e as funções dela em
`atividade2/atividade1.py`, e reproduz os resultados da Atividade 1 exatamente.

Os resultados dependem das versões do `requirements.txt`. Com várias threads o hgb não é
determinístico no shuttle e no page-blocks, então refazer a validação cruzada da Atividade 1 muda
essas duas células de P (Quadro 17 da Atividade 2).
