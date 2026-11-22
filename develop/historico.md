# Histórico — pydaltoolbox (port Python do daltoolbox)

Relato cronológico do trabalho concluído. Os próximos passos estão em
`planejamento.md`; a correspondência de classes em `mapeamento.md`. Ao concluir uma
atividade, registramos aqui.

Projeto: port Python do pacote R **daltoolbox** (CEFET/RJ, lab DAL), preservando a
**Experiment Line** (API uniforme `fit`/`action`/`evaluate`/`set_params`). Princípios:
embrulhar scikit-learn/numpy/pandas/scipy com adaptadores finos e **fiéis ao default do R**
(quality-faithful); back-implementar nativamente só o que é nicho/atípico, sempre com
atribuição ao trabalho original + descrição da inovação; docstrings com Teoria + Referências
+ Porting; bug em qualquer lado corrige-se nos dois repositórios.

## Bloco A / v1 — fundação (0.1.0, 2026-11-17)
- Scaffolding, núcleo (Experiment Line), amostragem, classificação completa (12 learners
  sklearn), regressão completa, transforms essenciais, clustering sklearn, família de tuning,
  bases de autoencoder (identidade). Mapeamento quality-faithful validado contra o R
  (SVM escala como `e1071`, árvores pré-podadas como `tree`/`rpart`, MLP escalado como `nnet`).
- **0.1.0 publicado no PyPI** (conta `eogasawara`); sdist exclui `develop/`, `notebooks/`,
  `examples/`, `tests/`, `tools/`. Pipeline de docs executáveis: `notebooks/**/*.md` →
  `examples/**/*.md` via `tools/build_examples.py`.

## Ciclo 1 — passada nativa / baixa-dependência por tema
Breadth com deps do núcleo (numpy/pandas/scikit-learn). Releases pequenos e frequentes.
- **0.2.0** — família Smoothing (Inter/Freq/Quantization/Cluster), nativa; tune refaz o ajuste.
- **0.3.0** — imputação preditiva (`ImputationPredictive`, `ImputationTree`), nativa.
- **0.4.0** — engenharia de atributos (`FeatureGeneration`, `Aggregation`, `HierarchyCut`).
- **0.5.0** — `SampleSimple` + `SampleGroups` (amostragem por cluster).
- **0.6.0** — clustering nativo: `ClusterPam` (PAM BUILD+SWAP) e `ClusterCmeans` (fuzzy c-means).
  Overhaul de docstrings: Teoria + Referências + Porting em todas as classes.
- **0.7.0** — seleção de atributos: `FeatureSelectionFss` (renomeado), `FeatureSelectionStepwise`
  (GLM por AIC, nativo), `FeatureSelectionRelief` (ReliefF nativo).
- **0.9.0** — mineração de padrões: `PatternMiner` + `PatUtils` + `PatApriori` (via **mlxtend**,
  primeiro extra opcional `[patterns]`). Eclat removido dos dois repos (arules e mlxtend só
  compartilham apriori).
- **0.10.0** — `RuleFilter` (base/None/Dara) + **DARA** (`dara`, algoritmo do grupo do Eduardo —
  Baroni et al. 2020). **Back-implementation canônica:** Python canônico, R vira backport,
  ambos citam o paper.
- **0.11.0** — `RuleFilterInterest` + thresholds (`SupportThreshold`/`ConfidenceThreshold`,
  auto `supp=0`/`conf=0`).
- **0.12.0** — sequências: `PatPrefixSpan` (PrefixSpan nativo, Pei et al. 2001). Back-implementation
  por posse de dependência nos dois repos (R dropou `arulesSequences`). Validado idêntico ao
  cSPADE no exemplo clássico de Zaki. **Fim do Ciclo 1.**

### Consolidação (back-implementation, 2026-11-21)
PAM (FastPAM1, ~20x mais rápido), fss, Relief (ReliefF completo), SMOTE (Borderline/ADASYN/
SMOTE-NC), remoção do `caret` do R, c-means (init robusto por `nstart`) — todos backportados
Python→R, dropando deps de nicho do R (`cluster`, `leaps`, `caret`, `e1071` p/ cmeans). kNN
mantido de propósito (FNN é rápido e estável). `ClusterLouvainGraph`/`igraph` **removidos** do
roadmap e do R (mau encaixe tabular).

## Ciclo 2 — completar (maior dependência, extras opcionais)
- **0.13.0** — `ClaXgboost` (wrapper `XGBClassifier`), primeiro extra de classificação
  `[xgboost]`. Paridade com o R verificada no iris.
- **0.14.0** — gráficos de distribuição + comparação (11 `plot_*`, extra `[viz]` = matplotlib):
  `plot_hist/density/density_class/boxplot/boxplot_class/bar/groupedbar/stackedbar/lollipop/
  pieplot/radar`. Decisões (com Eduardo): retorno `matplotlib.axes.Axes`; paleta CVD-safe
  quando `colors=None`; nomes `plot_*` exportados **lazy (PEP 562)** para não exigir matplotlib
  no núcleo.
- **0.15.0** — gráficos de relação + temporais + estrutura (10 `plot_*`):
  `plot_scatter/points/correlation/pair/parallel/pixel/series/ts/ts_pred/dendrogram`.

### Enxugamento de dependência gráfica (GGally, 2026-11-22)
- **`plot_pair_adv`** e **`plot_pair`** (ambos `GGally::ggpairs`) saíram do daltoolbox e do
  pydaltoolbox → viraram **extensões do curso** `C:\Git\eogasawara\datamining` (pasta `codes/`,
  versões R + Python, carregadas por `source`/`exec` da URL publicada no GitHub).
- **`plot_parallel`** reescrito em **ggplot2 nativo** no R (z-score + `geom_line`, sem
  `ggparcoord`) e refinado no Python (z-score por coluna); **mantido** nos dois pacotes.
- **`GGally` removido** do DESCRIPTION do daltoolbox (única lib gráfica externa de nicho;
  `ggplot2` fica, é o backend canônico).
- **0.16.0** — pydaltoolbox: `plot_pair` removido (→ curso), `plot_parallel` com z-score →
  **20 `plot_*`**.
- Limpeza extra no R: `graphics_dal_graphics.R` (página de overview sem código) removido.

## Estado atual (base para o `planejamento.md`)
- Última release: **pydaltoolbox 0.16.0** (PyPI). daltoolbox sem `GGally`. Curso com `codes/`.
- Cobertura: todo o daltoolbox portado, exceto conveniências de regras (ver `mapeamento.md`
  §Pendências). Testes verdes (216) + ruff limpos; notebooks de gráficos renderizam.
- Extras opcionais: `[patterns]` (mlxtend), `[xgboost]`, `[viz]` (matplotlib). Núcleo:
  numpy/pandas/scikit-learn.
