# Mapeamento R ↔ Python (daltoolbox ↔ pydaltoolbox)

Correspondência entre a API pública do pacote R **daltoolbox** e o port Python
**pydaltoolbox**. Convenções: classes R em `snake_case` (construtor S3) viram
**PascalCase** no Python; genéricos S3 (`fit`/`action`/`evaluate`/…) viram
**métodos de instância**; `data.frame` → `pandas.DataFrame`.

**Legenda:** ✅ portado · 🎓 movido para o curso `datamining` (extensão) · ⬜ não portado (ver §Pendências)

## Núcleo / contrato (Experiment Line)
| R (arquivo) | R (símbolo) | Python | Status |
|---|---|---|---|
| `dal_base.R` | `dal_base`, `set_params` | `DalBase`, `.set_params()` | ✅ |
| `dal_learner.R` | `dal_learner`, `action` | `DalLearner`, `.action()` | ✅ |
| `dal_transform.R`/`trans_transform.R` | `dal_transform`, `transform`, `inverse_transform` | `DalTransform`, `.transform()`, `.inverse_transform()` | ✅ |
| `dal_predictor.R` | `predictor`, `fit`, `evaluate`, `predict` | `Predictor`, `.fit()`, `.evaluate()`, `.predict()` | ✅ |
| `dal_predictor.R` | `predictor_prepare_fit` | (helper interno do `Predictor`) | ✅ |
| `dal_tune.R` | `dal_tune`, `select_hyper` | `DalTune`, `.select_hyper()` | ✅ |
| `dal_adjust.R` | `adjust_class_label`, `adjust_data.frame`, `adjust_factor`, `adjust_matrix` | `adjust_class_label`, `adjust_dataframe`, `adjust_factor`, `adjust_matrix` | ✅ |
| `data.R` | datasets (iris, …) | `datasets.iris/diabetes/boston` | ✅ |

## Amostragem
| R | Python | Status |
|---|---|---|
| `trans_sample.R` (`data_sample`), `train_test`, `k_fold`, `train_test_from_folds` | `DataSample`, `.train_test()`, `.k_fold()`, `train_test_from_folds` | ✅ |
| `trans_sample_random.R` (`sample_random`) | `SampleRandom` | ✅ |
| `trans_sample_strat.R` (`sample_stratified`) | `SampleStratified` | ✅ |
| `trans_sample_simple.R` (`sample_simple`) | `SampleSimple` | ✅ |
| `trans_sample_groups.R` (`sample_groups`) | `SampleGroups` | ✅ |

## Classificação
| R | Python | Status |
|---|---|---|
| `cla_classification.R` (`classification`) | `Classification` / `SklearnClassification` | ✅ |
| `cla_dtree.R` / `cla_rpart.R` | `ClaDtree` / `ClaRpart` | ✅ |
| `cla_knn.R` / `cla_nb.R` / `cla_majority.R` | `ClaKnn` / `ClaNb` / `ClaMajority` | ✅ |
| `cla_rf.R` / `cla_bagging.R` / `cla_boosting.R` | `ClaRf` / `ClaBagging` / `ClaBoosting` | ✅ |
| `cla_svm.R` / `cla_mlp.R` | `ClaSvm` / `ClaMlp` | ✅ |
| `cla_glm.R` / `cla_multinom.R` / `cla_glmnet.R` | `ClaGlm` / `ClaMultinom` / `ClaGlmnet` | ✅ |
| `cla_xgboost.R` (`cla_xgboost`) | `ClaXgboost` (extra `[xgboost]`) | ✅ |
| `cla_tune.R` (`cla_tune`) | `ClaTune` | ✅ |

## Regressão
| R | Python | Status |
|---|---|---|
| `reg_regression.R` (`regression`) | `Regression` / `SklearnRegression` | ✅ |
| `reg_lm.R`/`reg_dtree.R`/`reg_knn.R`/`reg_rf.R`/`reg_svm.R`/`reg_mlp.R` | `RegLm`/`RegDtree`/`RegKnn`/`RegRf`/`RegSvm`/`RegMlp` | ✅ |
| `reg_tune.R` (`reg_tune`) | `RegTune` | ✅ |

## Agrupamento
| R | Python | Status |
|---|---|---|
| `clu_clusterer.R` (`clusterer`), `cluster` | `Clusterer`, `.cluster()` | ✅ |
| `clu_kmeans.R`/`clu_dbscan.R`/`clu_gmm.R`/`clu_hclust.R` | `ClusterKmeans`/`ClusterDbscan`/`ClusterGmm`/`ClusterHclust` | ✅ |
| `clu_pam.R` (`cluster_pam`) | `ClusterPam` (nativo) | ✅ |
| `clu_cmeans.R` (`cluster_cmeans`) | `ClusterCmeans` (nativo) | ✅ |
| `clu_tune.R` (`clu_tune`) | `CluTune` | ✅ |
| `clu_utils.R` (`cluutils`, métricas) | `clustering.metrics` (`metric_*`, via `Clusterer.evaluate`) | ✅ |

## Transformações
| R | Python | Status |
|---|---|---|
| `trans_norm_minmax.R`/`trans_norm_zscore.R` | `MinMax` / `ZScore` | ✅ |
| `trans_na_removal.R` | `NaRemoval` | ✅ |
| `trans_imputation_simple.R`/`_predictive.R`/`_tree.R` | `ImputationSimple`/`ImputationPredictive`/`ImputationTree` | ✅ |
| `trans_outliers_boxplot.R`/`_gaussian.R` | `OutliersBoxplot`/`OutliersGaussian` | ✅ |
| `trans_categ_mapping.R` / `trans_dt_pca.R` | `CategMapping` / `DtPca` | ✅ |
| `trans_feature_selection_corr/_fss/_info_gain/_lasso/_relief/_stepwise.R` | `FeatureSelectionCorr/Fss/InfoGain/Lasso/Relief/Stepwise` | ✅ |
| `trans_feature_generation.R`/`trans_aggregation.R`/`trans_hierarchy_cut.R` | `FeatureGeneration`/`Aggregation`/`HierarchyCut` | ✅ |
| `trans_fit_curvature_max.R`/`_min.R` | `FitCurvatureMax`/`FitCurvatureMin` | ✅ |
| `trans_smoothing.R`/`_inter/_freq/_quantization/_cluster.R` | `Smoothing`/`SmoothingInter`/`SmoothingFreq`/`SmoothingQuantization`/`SmoothingCluster` | ✅ |
| `trans_sample_balance.R` (`sample_balance`) | `SampleBalance` | ✅ |
| `trans_sample_oversampling.R` (`bal_oversampling`) | `BalOversampling` | ✅ |
| `trans_sample_subsampling.R` (`bal_subsampling`) | `BalSubsampling` | ✅ |

## Mineração de padrões
| R | Python | Status |
|---|---|---|
| `pat_pattern_miner.R` (`pattern_miner`), `discover` | `PatternMiner`, `.discover()` | ✅ |
| `pat_utils.R` (`patutils`) | `PatUtils` | ✅ |
| `pat_apriori.R` (`pat_apriori`) | `PatApriori` (extra `[patterns]`) | ✅ |
| `pat_prefix_span.R` (`pat_prefix_span`) | `PatPrefixSpan` (nativo) | ✅ |
| `pat_dara.R` (`pat_dara`) | `dara`, `dara_score` | ✅ |
| `pat_rule_filter.R` (`pat_rule_filter_none/_dara/_interest`, `pat_filter_rules`, `transform.pat_rule_filter`) | `RuleFilter`, `RuleFilterNone`, `RuleFilterDara`, `RuleFilterInterest` | ✅ |
| `pat_threshold.R` (`pat_support_threshold`, `pat_confidence_threshold`) | `SupportThreshold`, `ConfidenceThreshold` | ✅ |
| `pat_dara.R` (`pat_rules_tidy`) | `pat_rules_tidy` | ✅ |

## Autoencoders
| R | Python | Status |
|---|---|---|
| `autoenc_base_e.R`/`autoenc_base_ed.R` | `AutoencBaseE`/`AutoencBaseEd` (bases identidade) | ✅ |

## Gráficos (`plot_*`, extra `[viz]`)
| R (`grf_*.R`) | Python (`viz/`) | Status |
|---|---|---|
| `grf_hist/density/density_class/boxplot/boxplot_class` | `plot_hist/density/density_class/boxplot/boxplot_class` | ✅ |
| `grf_bar/groupedbar/stackedbar/lollipop/pieplot/radar` | `plot_bar/groupedbar/stackedbar/lollipop/pieplot/radar` | ✅ |
| `grf_scatter/points/correlation/parallel/pixel` | `plot_scatter/points/correlation/parallel/pixel` | ✅ |
| `grf_series/ts/ts_pred` | **movidos p/ `pytspredit`** (`plot_series/ts/ts_pred`) | ➡️ |
| `grf_dendrogram` | `plot_dendrogram` | ✅ |

> Fora do escopo da biblioteca: `grf_pair`/`grf_pair_adv` (GGally) foram removidos de ambos
> os pacotes e viraram extensões do curso `datamining` (`codes/`, R + Python). O
> `graphics_dal_graphics.R` (página de overview sem código) também foi removido do R.

## Pendências de porte (⬜)
**Nenhuma** — a lista foi zerada no 0.19.0:
- `pat_rules_tidy` **portado** para o Python.
- `pat_dara_rules` **removido** do daltoolbox (atalho redundante; o fluxo é
  `PatApriori(target='rules', rule_filter=RuleFilterDara(...))` + `pat_rules_tidy`). Com ele saiu
  o único uso de `arules::is.redundant`, então não há mais essa lacuna.

**Importante (não confundir):** `pat_dara()` (o *ranqueamento* DARA, Baroni et al. 2020 — a
contribuição autoral do grupo) **está portado** como `dara`/`dara_score` e **permanece** nos dois
— o filtro `RuleFilterDara` depende dele. O `patutils`/`PatUtils` também **permanece**: fornece as
métricas do `evaluate()` do `PatternMiner` (usado por apriori/prefix_span).

(Diferença menor de superfície de API: as métricas de cluster do R são expostas como o
objeto `cluutils`; no Python são funções `metric_*` usadas via `Clusterer.evaluate` — mesma
funcionalidade.)
