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

#### ✅ Fase 4 — Classificação completa
- [x] `ClaKnn`, `ClaNb`, `ClaRf`, `ClaSvm`, `ClaMlp`
- [x] `ClaGlm`, `ClaMultinom`, `ClaGlmnet` (sobre `LogisticRegression`)
- [x] `ClaMajority` (baseline, `DummyClassifier`)
- [x] `ClaBagging`, `ClaBoosting` (ensembles)
- [x] `ClaRpart` (≈ árvore)
- [x] Exemplos: `examples/classification/` (13 por método + comparação).

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

#### Mapa de versões (um passo = uma versão = ~um dia)

**Cadência:** publicar **uma versão por dia**, trabalhando **um item por dia**. Cada passo
dos Ciclos 1 e 2 corresponde a um bump de minor.

| Passo | Versão | Tema |
|---|---|---|
| 1.1 | `0.2.0` ✅ | Smoothing |
| 1.2 | `0.3.0` ✅ | Imputação avançada |
| 1.3 | `0.4.0` ✅ | Transformações de atributos |
| 1.4 | `0.5.0` ✅ | Amostragem restante |
| 1.5 | `0.6.0` | Clustering nativo (Pam, Cmeans) |
| 1.6 | `0.7.0` | Seleção de atributos (Relief + fss/stepwise) |
| 1.7 | `0.8.0` | Padrões: base + itemsets |
| 1.8 | `0.9.0` | Padrões: regras e thresholds |
| 1.9 | `0.10.0` | Padrões: sequências (PrefixSpan) |
| 2.1 | `0.11.0` | `ClaXgboost` `[xgboost]` |
| 2.2 | `0.12.0` | `ClusterLouvainGraph` `[graph]` |
| 2.3 | `0.13.0` | Gráficos: base + distribuição + comparação `[viz]` |
| 2.4 | `0.14.0` | Gráficos: relação + temporais + estrutura `[viz]` |
| 2.5 | `0.15.0` | *(opcional)* Padrões: backend `mlxtend` `[patterns]` |
| — | `1.0.0` | Estabilização (cobertura completa, API congelada, beta→estável) |

*(`0.10.0` > `0.9.0` é válido em semver/PyPI — comparação numérica. Se o passo 2.5 for
descartado, a 1.0.0 vem após a `0.14.0`.)*

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

Núcleo permanece `numpy`/`pandas`/`scikit-learn` (sem dep nova). Pré-distribuição com
**todas as classes a migrar** no Ciclo 1 (~24 unidades), desmembrada em **9 passos**
(`1.1`–`1.9`); cada passo pode virar um ou mais `0.x.0`. Arquivo R indicado para
rastreabilidade.

#### ✅ 1.1 — Smoothing → `0.2.0` (publicado)
| Classe | Backend nativo | Arquivo R |
|---|---|---|
| `Smoothing` (base, + `evaluate`) | binning numpy/pandas | `trans_smoothing.R` |
| `SmoothingFreq` | binning por frequência | `trans_smoothing_freq.R` |
| `SmoothingInter` | binning por intervalo | `trans_smoothing_inter.R` |
| `SmoothingQuantization` | `KBinsDiscretizer` | `trans_smoothing_quantization.R` |
| `SmoothingCluster` | `KMeans` | `trans_smoothing_cluster.R` |

#### ✅ 1.2 — Imputação avançada → `0.3.0` (publicado)
| Classe | Backend nativo | Arquivo R |
|---|---|---|
| `ImputationPredictive` | `IterativeImputer` | `trans_imputation_predictive.R` |
| `ImputationTree` | árvore sklearn | `trans_imputation_tree.R` |

#### ✅ 1.3 — Transformações de atributos → `0.4.0` (publicado)
| Classe | Backend nativo | Arquivo R |
|---|---|---|
| `FeatureGeneration` | pandas | `trans_feature_generation.R` |
| `Aggregation` | pandas (groupby) | `trans_aggregation.R` |
| `HierarchyCut` | numpy/pandas | `trans_hierarchy_cut.R` |

#### ✅ 1.4 — Amostragem restante → `0.5.0` (publicado)
| Classe | Backend nativo | Arquivo R |
|---|---|---|
| `SampleSimple` | numpy | `trans_sample_simple.R` |
| `SampleGroups` | pandas | `trans_sample_groups.R` |

#### 1.5 — Clustering nativo → `0.6.0`
| Classe | Backend nativo | Arquivo R |
|---|---|---|
| `ClusterPam` (k-medoids) | PAM/CLARA | `clu_pam.R` |
| `ClusterCmeans` (fuzzy) | fuzzy c-means | `clu_cmeans.R` |

#### 1.6 — Seleção de atributos → `0.7.0`
| Classe | Backend nativo | Arquivo R |
|---|---|---|
| `FeatureSelectionRelief` (ReliefF) | numpy | `trans_feature_selection_relief.R` |
| reconciliar `fss` × `stepwise` | leaps-forward × `stats::step` | `trans_feature_selection_fss.R`, `trans_feature_selection_stepwise.R` |

*(Hoje existe `FeatureSelectionStepwise` baseado no `fss.R`; decidir se mantemos uma
classe ou criamos `FeatureSelectionFss` + `FeatureSelectionStepwise` distintas.)*

#### 1.7 — Mineração de padrões: base + itemsets → `0.8.0`
| Classe | Backend nativo | Arquivo R |
|---|---|---|
| `PatternMiner` (base, + `discover`) | — | `pat_pattern_miner.R` |
| `patutils` (utilitários) | — | `pat_utils.R` |
| `PatApriori` | apriori nativo | `pat_apriori.R` |
| `PatEclat` | eclat nativo | `pat_eclat.R` |

#### 1.8 — Mineração de padrões: regras e thresholds → `0.9.0`
| Classe/função | Backend nativo | Arquivo R |
|---|---|---|
| filtros de regras (`RuleFilterNone/Interest/Dara`, `filter_rules`, `rules_tidy`) | numpy/pandas | `pat_rule_filter.R` |
| thresholds (`pat_support_threshold`, `pat_confidence_threshold`) | numpy | `pat_threshold.R` |
| `PatDara` (+ `dara_rules`) | nativo | `pat_dara.R` |

#### 1.9 — Mineração de padrões: sequências → `0.10.0`
| Classe | Backend nativo | Arquivo R |
|---|---|---|
| `PatCspade` | PrefixSpan nativo | `pat_cspade.R` |

> ✅ Fim do Ciclo 1: **todo o daltoolbox coberto, exceto Louvain, XGBoost e gráficos**
> (que são o Ciclo 2), com o pacote ainda instalável só com o núcleo. Se o PrefixSpan
> (1.9) se mostrar inviável no prazo, documentar e levar ao Ciclo 2.

---

### CICLO 2 — rodada temática completa (classes de maior dependência) · *a refinar juntos*

Pré-distribuição com **todas as classes/funções** a migrar no Ciclo 2 (25 unidades:
2 learners + base de gráficos + 22 `plot_*`). Agrupadas em sub-releases `0.x.0`
(granularidade ajustável; ordem/escopo a refinar com o Eduardo). Cada arquivo R está
indicado para rastreabilidade.

#### 2.1 — Classificação extra → `[xgboost]` · `0.11.0`
| Classe | Backend | Arquivo R |
|---|---|---|
| `ClaXgboost` | `xgboost` | `cla_xgboost.R` |

#### 2.2 — Clustering extra → `[graph]` · `0.12.0`
| Classe | Backend | Arquivo R |
|---|---|---|
| `ClusterLouvainGraph` | `networkx` (Louvain) | `clu_louvain.R` |

#### 2.3 — Gráficos: base + distribuição + comparação → `[viz]` (`matplotlib`) · `0.13.0`
Base + 11 gráficos. Segue a skill `dataviz` (temas claro/escuro, paleta acessível).
| Função/base | Tipo | Arquivo R |
|---|---|---|
| `GraphicsDalGraphics` (base/utilitários) | base | `graphics_dal_graphics.R` |
| `plot_hist` | distribuição | `grf_hist.R` |
| `plot_density` | distribuição | `grf_density.R` |
| `plot_density_class` | distribuição | `grf_density_class.R` |
| `plot_boxplot` | distribuição | `grf_boxplot.R` |
| `plot_boxplot_class` | distribuição | `grf_boxplot_class.R` |
| `plot_bar` | comparação | `grf_bar.R` |
| `plot_groupedbar` | comparação | `grf_groupedbar.R` |
| `plot_stackedbar` | comparação | `grf_stackedbar.R` |
| `plot_lollipop` | comparação | `grf_lollipop.R` |
| `plot_pieplot` | comparação | `grf_pieplot.R` |
| `plot_radar` | comparação | `grf_radar.R` |

#### 2.4 — Gráficos: relação + temporais + estrutura → `[viz]` · `0.14.0`
11 gráficos.
| Função | Tipo | Arquivo R |
|---|---|---|
| `plot_scatter` | relação | `grf_scatter.R` |
| `plot_points` | relação | `grf_points.R` |
| `plot_correlation` | relação | `grf_correlation.R` |
| `plot_pair` | relação | `grf_pair.R` |
| `plot_pair_adv` | relação | `grf_pair_adv.R` |
| `plot_parallel` | relação | `grf_parallel.R` |
| `plot_pixel` | relação | `grf_pixel.R` |
| `plot_series` | temporal | `grf_series.R` |
| `plot_ts` | temporal | `grf_ts.R` |
| `plot_ts_pred` | temporal | `grf_ts_pred.R` |
| `plot_dendrogram` | estrutura | `grf_dendrogram.R` |

#### 2.5 — (opcional) Padrões: backend alternativo → `[patterns]` · `0.15.0`
Sem classes novas: trocar/aumentar o backend de `PatApriori`/`PatEclat` para `mlxtend`
**somente** se o nativo do Ciclo 1 precisar de mais eficiência. Decidir junto.

> **Resumo do Ciclo 2:** `ClaXgboost`, `ClusterLouvainGraph`, `GraphicsDalGraphics` + 22
> `plot_*`. Extras introduzidos: `[xgboost]`, `[graph]` (networkx), `[viz]` (matplotlib),
> e opcionalmente `[patterns]` (mlxtend).
>
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
