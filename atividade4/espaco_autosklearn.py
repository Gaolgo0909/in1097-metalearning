"""Resume o espaço de busca do Auto-sklearn 0.15 para classificação densa (ambiente .venv)."""
import json

from autosklearn.pipeline.classification import SimpleClassificationPipeline

espaco = SimpleClassificationPipeline(dataset_properties={"multiclass": True, "sparse": False}).get_hyperparameter_search_space()
escolhas = {nome: espaco.get_hyperparameter(nome).choices
            for nome in ("classifier:__choice__", "feature_preprocessor:__choice__", "balancing:strategy",
                         "data_preprocessor:feature_type:numerical_transformer:rescaling:__choice__",
                         "data_preprocessor:feature_type:categorical_transformer:categorical_encoding:__choice__")}
por_classificador = {c: sum(n.startswith(f"classifier:{c}:") for n in espaco.get_hyperparameter_names())
                     for c in escolhas["classifier:__choice__"]}
print(json.dumps({"hiperparametros": len(espaco.get_hyperparameters()), "escolhas": escolhas,
                  "por_classificador": por_classificador}))
