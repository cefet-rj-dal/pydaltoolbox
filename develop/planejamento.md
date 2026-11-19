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

**Depois da v1 — dois ciclos temáticos até a 1.0.0:** **Ciclo 1** passa por cada tema só
com o que é **nativo** (sem dep nova) — entrega a largura do framework barato; **Ciclo 2**
volta em cada tema e acrescenta as classes de **maior dependência** como extras opcionais
(`[graph]`, `[xgboost]`, `[viz]`...). Releases podem ser bem pequenos (muitos ciclos são
bem-vindos). O roadmap detalhado, com a tabela de decisão *dependência vs. nativo*, está
na seção 8 (Bloco B). O Ciclo 2 será refinado em conjunto.

**Dependências do núcleo (até a 1.0.0):** `numpy`, `pandas`, `scikit-learn` apenas. Tudo
além disso é **extra opcional** (`pip install pydaltoolbox[graph,xgboost,viz]`).

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
| Publicação | `0.1.0` publicada (fim do Bloco A). Depois, **releases incrementais** até a `1.0.0` em **dois ciclos temáticos** (ver seção 8; muitos ciclos pequenos são bem-vindos). Eduardo autoriza/roda o upload (token `daldeploy`). PyPI sem prazo; versões imutáveis (sempre incrementar). Pausa a cada fase/sub-bloco para revisão no GitHub. |
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

#### ✅ Fase 6 — Transformações essenciais *(em sub-blocos)*
- [x] **6a** Normalização: `MinMax`, `ZScore` (fit/transform/inverse_transform)
- [x] **6b** Limpeza: `NaRemoval`, `ImputationSimple`, `OutliersBoxplot`, `OutliersGaussian`
- [x] **6c** Codificação/projeção: `CategMapping` (one-hot), `DtPca` (+ `FitCurvatureMin/Max`, trazidos do Bloco B pois o PCA depende deles)
- [x] **6d** Seleção: `FeatureSelectionCorr`, `FeatureSelectionLasso`,
      `FeatureSelectionInfoGain`, `FeatureSelectionStepwise`
- [x] **6e** Balanceamento: `SampleBalance`, `BalSubsampling`, `BalOversampling` (random + SMOTE local)
- Exemplos: `examples/transf/` (13 exemplos, um por transform).

#### ✅ Fase 7 — Clustering (sklearn) *(em sub-blocos)*
- [x] **7a** `Clusterer` base + métricas (`sklearn.metrics`) + `ClusterKmeans`
- [x] **7b** `ClusterDbscan` (eps via cotovelo kNN), `ClusterGmm` (G via BIC), `ClusterHclust`
- [x] **7c** `CluTune` — feito junto com a família de tuning (ver Fase 8)
- Exemplos: `examples/clustering/`.

#### ✅ Fase 8 — Tuning + bases de autoencoder
- [x] `DalTune` base + `ClaTune`, `RegTune`, `CluTune` (feito junto, a pedido — família completa)
- [x] `AutoencBaseE`, `AutoencBaseEd` (identidade) — harbinger herda
- Exemplos: tuning workflow.

#### ✅ Fase 9 — Fechamento da v1  →  🚀 **Publicação (`0.1.0`)**
- [x] Revisão de API pública (64 símbolos), docstrings, `__all__`; suíte consolidada
      (84 testes, ruff limpo); README atualizado para a v1 completa.
- [x] **Publicado `pydaltoolbox` 0.1.0 no PyPI** (2026-11-17) → https://pypi.org/project/pydaltoolbox/
      Próximo: caminho 0.2.0 → 1.0.0 (Bloco B, seção 8).

> 🎯 Concluído o Bloco A, `pytspredit` e `pyharbinger` têm fundação completa.

### BLOCO B — dois ciclos temáticos até a 1.0.0

Cobrir **todas** as classes do `daltoolbox` (R) em **duas rodadas** temáticas:

- **Ciclo 1 — baixa dependência (nativo):** passar por cada tema implementando só o que é
  nativo (núcleo `numpy`/`pandas`/`scikit-learn`, sem dep nova). Entrega a **largura** do
  framework barato: ao fim do ciclo, todos os temas têm cobertura, menos as classes que
  exigem lib pesada.
- **Ciclo 2 — completar (maior dependência):** voltar em cada tema e acrescentar as
  classes que precisam de **extra opcional** (`[graph]`, `[xgboost]`, `[viz]`, ...).
  **Este ciclo será refinado em conjunto** (escopo e ordem a decidir com o Eduardo).

Releases podem ser **tão pequenos quanto fizer sentido** (um tema, ou até um sub-bloco,
por release) — não há problema em chegar a muitos ciclos (20+) até a 1.0.0. Cada release
é um `0.x.0`; versões são imutáveis (sempre incrementar).

#### Princípio: dependência vs. implementação nativa

Regra: **implementar nativo** quando o algoritmo é simples/didático e a alternativa é uma
lib pesada, de nicho ou pouco mantida; **adicionar dependência (como extra opcional)**
quando é complexo/padronizado e existe lib leve e bem mantida. Decisões por componente:

| Componente | Opção com dependência | Opção nativa | Decisão | Ciclo |
|---|---|---|---|---|
| Smoothing, imputação (preditiva/árvore), feature generation, aggregation, hierarchy cut, sampling (simple/groups) | — | numpy/pandas/sklearn (já no núcleo) | **nativo** | 1 |
| `FeatureSelectionRelief` | `skrebate` (pouco mantido) | ReliefF (médio) | **nativo** | 1 |
| `ClusterPam` (k-medoids) | `scikit-learn-extra` | PAM/CLARA (médio) | **nativo** | 1 |
| `ClusterCmeans` (fuzzy) | `skfuzzy` | fuzzy c-means (~40 linhas) | **nativo** | 1 |
| `PatApriori`/`PatEclat` + regras | `mlxtend` (puxa matplotlib) | apriori/eclat (médio) | **nativo** | 1 |
| `PatCspade` (sequências) | `prefixspan` / SPMF (Java) | PrefixSpan (médio) | **nativo** | 1 |
| `ClusterLouvainGraph` | `networkx` (Louvain embutido) | Louvain (complexo) | **dep → `[graph]`** | 2 |
| `ClaXgboost` | `xgboost` (binário) | — (não reimplementar) | **dep → `[xgboost]`** | 2 |
| Gráficos (`plot_*`, ~23) | `matplotlib` | — | **dep → `[viz]`** | 2 |

---

### CICLO 1 — rodada temática de baixa dependência (nativo)

Núcleo permanece `numpy`/`pandas`/`scikit-learn`. Sequência sugerida (ajustável; cada
item pode virar um ou mais `0.x.0`):

- 🔭 **Transformações restantes:** `Smoothing` base (+ `evaluate`), `SmoothingFreq`,
  `SmoothingInter`, `SmoothingQuantization` (`KBinsDiscretizer`), `SmoothingCluster`
  (`KMeans`); `ImputationPredictive` (`IterativeImputer`), `ImputationTree`;
  `FeatureGeneration`, `Aggregation`, `HierarchyCut`; `SampleSimple`, `SampleGroups`.
- 🔭 **Clustering (nativo):** `ClusterPam` (k-medoids), `ClusterCmeans` (fuzzy c-means).
- 🔭 **Seleção de atributos (nativo):** `FeatureSelectionRelief` (ReliefF); reconciliar
  as variantes `fss`/`stepwise`.
- 🔭 **Mineração de padrões (nativo):** `PatternMiner` base + `discover` + `patutils`;
  `PatApriori`, `PatEclat`, filtros de regras (`RuleFilter*`), thresholds, `PatDara`;
  `PatCspade` (PrefixSpan nativo — se inviável, documentar e levar ao Ciclo 2).

> ✅ Fim do Ciclo 1: **todo o daltoolbox coberto, exceto Louvain, XGBoost e gráficos**,
> com o pacote ainda instalável só com o núcleo.

---

### CICLO 2 — rodada temática completa (classes de maior dependência) · *a refinar juntos*

Revisitar cada tema acrescentando o que exige **extra opcional**. Escopo/ordem a definir
com o Eduardo; esboço inicial:

- 🔭 **Classificação:** `ClaXgboost` → `[xgboost]`.
- 🔭 **Clustering:** `ClusterLouvainGraph` → `[graph]` (`networkx`).
- 🔭 **Gráficos:** família `plot_*` (~23) → `[viz]` (`matplotlib`), seguindo a skill
  `dataviz` (temas claro/escuro, paleta acessível). Tema inteiro é de Ciclo 2.
- 🔭 **Padrões (se necessário):** backend alternativo (`mlxtend`/`[patterns]`) caso o
  nativo do Ciclo 1 precise de mais eficiência.
- 🔭 *(outros itens de maior dependência que surgirem — decidir juntos.)*

> **Autoencoders reais** (backend de deep learning) permanecem **fora** do pydaltoolbox:
> as bases identidade bastam; a implementação real fica num pacote companheiro, como no R.

---

### 🔭 1.0.0 — Estabilização (completa)
- Cobertura de **todas** as classes do R (ou exceções documentadas).
- Congelar a API pública; revisar docstrings e `__all__`.
- Site de documentação (equivalente ao pkgdown); paridade com os tutoriais do R.
- Passada de performance/qualidade; **promover de beta para estável**.

> **Extras opcionais previstos:** `[graph]` (networkx), `[xgboost]`, `[viz]` (matplotlib),
> e, se adotado, `[patterns]` (mlxtend). Núcleo enxuto (numpy/pandas/scikit-learn) até a 1.0.0.

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
