# Planejamento — Port do daltoolbox (R) para pydaltoolbox (Python)

> Documento vivo. Atualizado a cada checkpoint. Objetivo: você saber **todos os
> passos**, o que já foi feito, o que vem, e por quê — sem surpresas.

Última atualização: 2026-11-17

---

## 1. O todo: o ecossistema DAL

`pydaltoolbox` é a **fundação** de uma família:

```
          pydaltoolbox  (fundação: Linha de Experimento)
                 │
        ┌────────┴────────┐
   pytspredit         pyharbinger
 (previsão de TS)   (detecção de eventos em TS)
```

No R: `harbinger` depende de `daltoolbox` **e** `tspredit`; `tspredit` depende de
`daltoolbox`. Estratégia: **publicar o `pydaltoolbox` primeiro**, já rico o
suficiente para viabilizar `pytspredit` e `pyharbinger` depois.

---

## 2. Critério de escopo da v1 (o princípio)

> **Entra na v1 tudo que for um *adapter fino* sobre o que o `scikit-learn` /
> `numpy` / `pandas` já oferecem. Só adiamos o que exige biblioteca extra/nicho
> ou backend complexo.**

Nossa contribuição é a **interface uniforme** (Linha de Experimento), não os
algoritmos. Como o sklearn cobre a grande maioria dos métodos do daltoolbox, a v1
pode ser **ampla** sem custo de implementação alto.

**Adiado para o Bloco B** (lib extra ou complexidade real):
`xgboost` (dep compilada), `pam`/k-medoids (sklearn-extra), fuzzy `cmeans` (skfuzzy),
`louvain` (networkx), seleção `relief` (skrebate), imputação preditiva/por árvore,
smoothing avançado, autoencoders com backend de deep learning, mineração de padrões
(`mlxtend`; sequências em aberto) e gráficos (`matplotlib`).

**Dependências da v1:** `numpy`, `pandas`, `scikit-learn` apenas. O resto vira
**extra opcional** (`pip install pydaltoolbox[xgboost]`, etc.) quando entrar.

---

## 3. Superfície que os downstream usam (guia de prioridade)

Levantado do código de `harbinger` e `tspredit`. O que eles mais consomem ganha
prioridade dentro da v1:

| Símbolo do daltoolbox | tspredit | harbinger | Prioridade |
|---|---|---|---|
| `regression` (base + regressores) | muito | muito | **alta** |
| `classification` (base + modelos) | 1× | sim | **alta** |
| `dal_transform`, normalização | muito | sim | **alta** |
| `predictor`, `fit`/`transform`/`predict`/`inverse_transform` | muito | sim | **alta** (feito) |
| `dal_base`, `set_params`, `action` | sim | sim | feito |
| `sample_random`/`sample_stratified`/`train_test`/`k_fold` | sim | — | feito |
| `dal_tune` (tuning) | sim | — | média |
| `autoenc_base_e/ed` (identidade) | — | sim | média |
| clustering, padrões, gráficos | — | — | Bloco B |

*(Os `do_fit`/`do_predict` são do próprio tspredit, não do daltoolbox. Os
autoencoders-base são stubs de identidade — sem DL. harbinger usa `stats::kmeans`
direto, não o clusterer.)*

---

## 4. Decisões travadas

| Tema | Decisão |
|---|---|
| Escopo v1 | Tudo que é wrapper fino de sklearn/numpy/pandas (ver §2) |
| Ordem | **Classificação completa → Regressão** → transformações → clustering → tuning |
| Estilo de API | Métodos de instância; `fit` retorna `self` (encadeável) |
| Nomenclatura | **PascalCase** (PEP 8): `ClaKnn`, `RegRf`, `ClusterKmeans`, `MinMax` |
| Modelo de dados | `pandas.DataFrame` |
| Entrada do `predict` | Aceita frame inteiro **ou** só features; exemplos passam só features |
| Dependências v1 | `numpy`, `pandas`, `scikit-learn`; demais = extras opcionais |
| Publicação | **Um único release (`0.1.0`) ao final da Fase 9** (fecho da v1). Eduardo faz o upload (token próprio). PyPI sem prazo; versões imutáveis. Paramos a cada fase para ele conferir a documentação no GitHub. |
| Autoencoders | Só bases identidade na v1; sem backend de DL |
| Padrões / Gráficos | Adiados (Bloco B) |

---

## 5. Arquitetura (árvore de classes)

```
DalBase ─ fit / action / set_params
├── DalTransform ───── action → transform / inverse_transform
│   ├── DataSample (train_test, k_fold)                          [feito]
│   ├── MinMax, ZScore, DtPca, CategMapping, Imputation, ...     [v1]
│   └── AutoencBaseE / AutoencBaseEd (identidade)                [v1]
├── DalLearner ─────── action → predict ; + evaluate
│   ├── Predictor ───── fit registra as features (x)
│   │   ├── Classification(attribute, slevels) → acc/prec/recall/f1  [v1]
│   │   └── Regression(attribute)              → mse/smape/R2        [v1]
│   └── Clusterer ───── action → cluster ; evaluate = métricas      [v1 (sklearn)]
└── DalTune ────────── select_hyper (ClaTune, RegTune)             [v1]

PatternMiner (discover) ....................................... [Bloco B]
```

---

## 6. Mapeamento R → Python (por componente)

**Classificação:** dtree→`DecisionTreeClassifier` · knn→`KNeighborsClassifier` ·
nb→`GaussianNB` · rf→`RandomForestClassifier` · svm→`SVC` · mlp→`MLPClassifier` ·
glm/multinom→`LogisticRegression` · glmnet→`LogisticRegression(elasticnet)` ·
majority→`DummyClassifier` · bagging→`BaggingClassifier` · boosting→`AdaBoostClassifier` ·
rpart→`DecisionTreeClassifier`. *(xgboost→`xgboost` = extra, Bloco B.)*

**Regressão:** lm→`LinearRegression` · dtree→`DecisionTreeRegressor` ·
knn→`KNeighborsRegressor` · rf→`RandomForestRegressor` · svm→`SVR` · mlp→`MLPRegressor`.

**Clustering:** kmeans→`KMeans` · dbscan→`DBSCAN` · gmm→`GaussianMixture` ·
hclust→`AgglomerativeClustering`; métricas→`sklearn.metrics`
(silhouette, davies_bouldin, calinski_harabasz, adjusted_rand). *(pam/cmeans/louvain = Bloco B.)*

**Transformações:** minmax→`MinMaxScaler` · zscore→`StandardScaler` · pca→`PCA` ·
categ_mapping→`OneHotEncoder` · imputation_simple→`SimpleImputer` · na_removal→pandas ·
outliers→numpy/scipy · fs_corr→pandas · fs_lasso→`SelectFromModel(Lasso)` ·
fs_info_gain→`mutual_info_classif` · fs_stepwise→`SequentialFeatureSelector` ·
balanceamento simples→`sklearn.utils.resample`.

---

## 7. Estratégia de garantia (três camadas)

1. **Unitário** → `pytest` por componente (`fit → action → evaluate`).
2. **Integrado** → `pytest` ponta-a-ponta (tutoriais).
3. **Documentação executável** → `notebooks/**/*.md` (jupytext) renderizado em
   `examples/**/*.md` (saída real embutida) por `tools/build_examples.py`
   (`jupytext` + `nbconvert`). Falha se houver erro → é smoke test. Tabelas saem em
   texto puro (estilo `##` do R).

   **Convenção (como no daltoolbox):** **um exemplo didático por classe** —
   objetivo do método, quando usar e **referência (artigo)** + o ciclo
   `fit`/`predict`/`evaluate`. Além deles, um exemplo "guarda-chuva" por família
   (ex.: comparação de learners) para seleção de modelos.

**Comandos:** `pip install -e ".[docs]"` · `python tools/build_examples.py`.

**Ritual por fase:** implemento + testes → crio o notebook do tema → rodo o build →
você confere o `.md` processado no GitHub → escalamos.

---

## 8. Roadmap

Legenda: ✅ feito · 🔜 próximo · ⬜ pendente (v1) · 🔭 Bloco B (pós-v1)

### BLOCO A — v1 publicável (wrappers finos de sklearn/numpy/pandas)

#### ✅ Fase 0 — Scaffolding
`pyproject`, layout `src/`, `deploy.md`, `README` (com install via GitHub).

#### ✅ Fase 1 — Núcleo
`DalBase`, `DalTransform`, `DalLearner`, `Predictor`, helpers `adjust_*`.

#### ✅ Fase 2 — Amostragem
`SampleRandom`, `SampleStratified` (`train_test`, `k_fold`), `train_test_from_folds`.

#### ✅ Fase 3 — Slice vertical de classificação
`Classification` base + `evaluate`, `ClaDtree`, `datasets`, Tutorial 01.

#### 🔜 Fase 4 — Classificação completa
- [ ] `ClaKnn`, `ClaNb`, `ClaRf`, `ClaSvm`, `ClaMlp`
- [ ] `ClaGlm`, `ClaMultinom`, `ClaGlmnet` (sobre `LogisticRegression`)
- [ ] `ClaMajority` (baseline, `DummyClassifier`)
- [ ] `ClaBagging`, `ClaBoosting` (ensembles)
- [ ] `ClaRpart` (≈ árvore)
- Exemplos: `examples/classification/` (fundamentos, famílias de learners).

#### ✅ Fase 5 — Regressão completa
- [x] `Regression` base + `evaluate` (mse/smape/R2)
- [x] `RegLm`, `RegDtree`, `RegKnn`, `RegRf`, `RegSvm`, `RegMlp`
- [x] `datasets.diabetes()` (offline) como dataset de regressão
- [x] Exemplos: `examples/regression/` (um por classe + comparação).

#### ⬜ Fase 6 — Transformações essenciais *(em sub-blocos)*
- [x] **6a** Normalização: `MinMax`, `ZScore` (fit/transform/inverse_transform)
- [x] **6b** Limpeza: `NaRemoval`, `ImputationSimple`, `OutliersBoxplot`, `OutliersGaussian`
- [x] **6c** Codificação/projeção: `CategMapping` (one-hot), `DtPca` (+ `FitCurvatureMin/Max`, trazidos do Bloco B pois o PCA depende deles)
- [x] **6d** Seleção: `FeatureSelectionCorr`, `FeatureSelectionLasso`,
      `FeatureSelectionInfoGain`, `FeatureSelectionStepwise`
- [ ] **6e** Balanceamento simples (`SampleBalance`, over/subsampling via `resample`)
- Exemplos: `examples/transf/`.

#### ⬜ Fase 7 — Clustering (sklearn)
- [ ] `Clusterer` base + `cluutils` (métricas via `sklearn.metrics`)
- [ ] `ClusterKmeans`, `ClusterDbscan`, `ClusterGmm`, `ClusterHclust`
- [ ] `CluTune`
- Exemplos: `examples/clustering/`.

#### ⬜ Fase 8 — Tuning + bases de autoencoder
- [ ] `DalTune` base + `select_hyper`, `ClaTune`, `RegTune`
- [ ] `AutoencBaseE`, `AutoencBaseEd` (identidade) — harbinger herda
- Exemplos: tuning workflow.

#### ⬜ Fase 9 — Fechamento da v1  →  🚀 **Publicação (`0.1.0`)**
- [ ] Revisão de API pública, docstrings, `__all__`; suíte consolidada
- [ ] **Publicar `pydaltoolbox` 0.1.0 no PyPI** (via `deploy.md`; pipeline já
      validado: build + `twine check` + install em venv limpo). Eduardo faz o upload.

> 🎯 Concluído o Bloco A, `pytspredit` e `pyharbinger` têm fundação completa.

### BLOCO B — pós-v1 (requer lib extra ou complexidade)

- 🔭 **Classificação extra:** `ClaXgboost` (extra `[xgboost]`).
- 🔭 **Clustering extra:** `ClusterPam` (sklearn-extra), `ClusterCmeans` (skfuzzy),
  `ClusterLouvainGraph` (networkx).
- 🔭 **Transformações avançadas:** `FeatureSelectionRelief` (skrebate), imputação
  preditiva/por árvore, smoothing avançado, feature generation, aggregation,
  heurísticas de curvatura.
- 🔭 **Autoencoders reais:** backend de deep learning (pacote companheiro).
- 🔭 **Mineração de padrões:** `PatApriori`/`PatEclat` (`mlxtend`); sequências
  (`PatCspade`) em aberto; filtros de regras.
- 🔭 **Gráficos:** ~20 `plot_*` (matplotlib), aos poucos.

---

## 9. Próximos passos nos pacotes irmãos (visão macro)

1. **pytspredit** — `ts_data`/`ts_projection` (janela deslizante), `ts_norm_*`,
   regressores de TS sobre a `Regression`, `ts_tune`/`ts_integtune` sobre o `DalTune`,
   filtros e augmentation.
2. **pyharbinger** — base de detecção de eventos, detectores estatísticos/ML,
   ensembles, avaliação, reusando `Regression`/`Classification`/`AutoencBaseEd` daqui
   + regressores do pytspredit.

*(Planejamentos próprios quando chegarmos lá.)*

---

## 10. Como trabalhamos

- Incremental, um passo por vez; você confere o `examples/` processado antes de escalar.
- Verde sempre (`ruff` + `pytest` + build dos exemplos).
- Decisões sempre registradas aqui.
- Trabalho em **branch**; push para você revisar no GitHub. Sem merge/commit na `main`
  sem você pedir.

---

## 11. Pendências / decisões em aberto

- [x] ~~Dataset de regressão~~ → resolvido: `datasets.diabetes()` (offline) na Fase 5;
      `boston()` mantido só por compatibilidade (baixa da web, desencorajado).
- [ ] Backend de mineração de padrões sequenciais (Bloco B).
