# IN1097 · Meta-aprendizado

Gabriel Oliveira Gonçalves · Tópicos Avançados em Agentes Inteligentes 2

Meta-dataset de seleção de algoritmos: 33 datasets de classificação do OpenML, 6 algoritmos do
scikit-learn, AUC em validação cruzada estratificada de 10 folds. Os notebooks estão com as saídas
executadas.

| Atividade | Notebook | Conteúdo |
|---|---|---|
| 1 — Meta-dataset e ranking de algoritmos | [`coleta_openml.ipynb`](coleta_openml.ipynb) | coleta dos datasets, meta-características X, performances P por fold, ranks R por vitórias significativas (Wilcoxon), as seis abordagens (AR, MR, vitórias, regressores sobre P e sobre R, HARRIS) e a avaliação leave-one-dataset-out |
| 2 — Meta-características específicas do domínio | [`atividade2/metacaracteristicas_dominio.ipynb`](atividade2/metacaracteristicas_dominio.ipynb) | levantamento, meta-características de desbalanceamento e dificuldade de classe, medidas de complexidade, custo de extração, relação com o meta-alvo, redundância e efeito no sistema de recomendação |

## Como reproduzir

```bash
pip install -r requirements.txt
```

1. `coleta_openml.ipynb`, com Run All. Baixa os 33 datasets para `dados_openml/` (277 MB, fora do
   repositório). Os experimentos já feitos estão nos CSVs e não são refeitos.
2. `atividade2/metacaracteristicas_dominio.ipynb`, que depende de `dados_openml/`. Para executar
   gravando as saídas: `python3 atividade2/.build/executar.py`. A extração e o leave-one-dataset-out
   vêm dos CSVs em cache; só os diagnósticos do hgb rodam de novo, cerca de 25 minutos.

A Atividade 2 usa P, R e X_base congelados da Atividade 1 em `atividade2/base/`, e o código da
Atividade 1 em `atividade2/atividade1.py`, extraído literalmente do notebook por
`atividade2/.build/extrair_atividade1.py`. Com isso, a Atividade 2 reproduz os resultados da
Atividade 1 bit a bit.

A reprodução exata depende das versões em `requirements.txt`. Uma exceção conhecida: com várias
threads, o hgb não é determinístico no shuttle e no page-blocks, de modo que refazer a validação
cruzada da Atividade 1 muda essas duas células de P — ver o Quadro 17 da Atividade 2.
