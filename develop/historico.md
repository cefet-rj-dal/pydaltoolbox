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
  `examples/` e `tests/`. Pipeline de docs executáveis: `notebooks/**/*.md` →
  `examples/**/*.md` via `notebooks/conversor.py`.

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

### Limpezas / simplificação da API de padrões (0.19.0, 2026-11-22)
- **`pat_dara_rules` removido** do daltoolbox (atalho que agrupava minerar+redundância+tidy;
  redundante com `pat_apriori(target="rules") + pat_rules_tidy` e os rule filters). Com ele saiu
  o único uso de `arules::is.redundant`. Mantidos `pat_dara()` (ranqueamento DARA) e
  `pat_rules_tidy()`. Exemplos/tutorial do DARA reescritos para o fluxo apriori+tidy.
- **`pat_rules_tidy` portado para o Python** (`pydaltoolbox.pat_rules_tidy`): achata as regras do
  `PatApriori(target="rules")` em `lhs ⇒ rhs` + support/confidence/lift/count/kulc/ir, com parse
  opcional de `attribute=value` por coluna.
- Resultado: **Pendências de porte zeradas** (`mapeamento.md`). pydaltoolbox **0.19.0**.

### Documentação — organização no daltoolbox (0.20.0, 2026-11-22, só-R)
- **`Rmd/` renomeada para `notebooks/`** no daltoolbox (espelha o Python: `notebooks/` → `examples/`),
  via `git mv` (com `conversor.R`, `README.Rmd` e subpastas). `.Rbuildignore`: `^Rmd$` → `^notebooks$`.
- **`conversor.R` enxuto:** fonte agora `notebooks/`; **removida a geração de Word/.docx** (função
  `convert_rmd_docx` + fase `rmd->docx`); gera só `.md` + `.R` puro.
- **`examples/` regenerada do zero** com o pacote local instalado (reflete o código atual: sem
  `plot_pair`/`plot_pair_adv`/`pat_dara_rules`/GGally; `plot_parallel` nativo). Mantidos os ativos
  não-gerados: `seed.R`, `slides/`, `videos/`, `wine.RData`. **144/144 renderizados, zero falhas.**
- Sem release do pydaltoolbox (passo só-R; a versão Python segue 0.19.0).

### Exemplos 1:1 R↔Python — ML, transf, patterns, custom (0.21.0, 2026-11-22)
Reorganização dos exemplos Python para **um arquivo por classe**, espelhando a árvore de
subpastas e a didática do R (seções About/Didactic goal/setup/load/split/fit/eval/What to
observe/References). **71 notebooks Python** (todos buildam verdes):
- classification(15): fundamentals, knn, probabilistic, svm, neural-networks, linear-models,
  ensemble, model-selection; regression(7): linear, beyond-linear, ensemble, flexible,
  model-selection; clustering(8): partitional, density, advanced-views, model-selection;
  transf(30): sampling, scaling, cleaning, encoding, feature-selection, dimensionality,
  smoothing, balancing, aggregation; patterns(4): apriori-rules, apriori-itemsets,
  prefixspan-sequences, divergent-association-ranking; custom(7): transformation,
  classification, regression, clustering, autoencoder encoder / encoder-decoder, pattern-miner.
- `tuning/` dissolvido em `*/model-selection/`; `90-comparing-*` removidos; pasta `pattern/`→`patterns/`.
- **R (daltoolbox):** adicionado `data/Diabetes.RData` (= `datasets.diabetes()`, 442x11), e trocado
  **Boston→Diabetes / medv→target** em regressão, transf feature-selection (stepwise/lasso/fss),
  custom/20 e tutorial/21. Exemplos `patterns` renomeados: eclat→apriori-itemsets, cspade→prefixspan.
  `examples/` regenerada (144/144, zero falhas).
- Dev-only nos dois lados (notebooks/examples fora do pacote): sem release PyPI.

### Exemplos 1:1 R↔Python — gráficos + tutorial (0.22.0, 2026-11-22)
Fecha a iniciativa de exemplos 1:1. **36 notebooks Python** (todos buildam verdes):
- **graphics(23)**: `comparison/` (bar, bar-with-error, grouped-bar, stacked-bar, lollipop, pie,
  radar) + `distribution/` (histogram, density, boxplot, boxplot-by-class, density-by-class,
  scatter, points, correlation, parallel, pixel, dendrogram, time-series lines/basic/forecast,
  export-save pdf/jpg) — um por `plot_*`, retornando `Axes` (figura embutida via `ax.figure`).
- **tutorial(13)**: first-experiment, sampling-strategy, data-quality-and-cleaning,
  preprocessing-basics, baseline-classification, metrics-and-evaluation, model-comparison,
  tuning-workflow, end-to-end-classification-pipeline, regression-workflow (diabetes),
  clustering-workflow, visual-analysis-and-reporting, custom-extension.
- **Paridade final: 107 notebooks Python = 107 notebooks R**, família por família
  (classification 15, regression 7, clustering 8, transf 30, patterns 4, custom 7, graphics 23,
  tutorial 13). Dev-only: sem release PyPI.

### Testes pareados R↔Python + fix de qualidade (0.23.0, 2026-11-22)
Primeira passada de **testes pareados** (regressão). Exportado o split idêntico (diabetes,
`SampleRandom(seed=1)`) para CSV e rodados os mesmos modelos nos dois lados sobre as MESMAS
linhas (scripts em `daltoolbox/develop/parity/`). Resultado: **paridade excelente** em quase
tudo (RegLm 0.433=0.433, RegKnn 0.319=0.319, RegRf 0.351=0.351, RegSvm def 0.099≈0.100,
RegMlp 0.339 vs 0.290) — **exceto RegDtree**: Python overfitava para R2 **−0.003** enquanto o
R `tree::tree` generalizava (**0.238**).
- **Fix (pydaltoolbox 0.23.0, publicado):** `RegDtree` agora usa
  `min_impurity_decrease = mindev * var(y)` (`mindev=0.01`), o equivalente fiel do `mindev` do
  `tree::tree` → R2 **0.238**, idêntico ao R. Novo parâmetro `mindev`. Adicionado
  `tests/test_quality_regression.py` (guarda: RegDtree R2>0.15; ML regressores R2>0). 220 testes.
- Nota: em diabetes (dataset quase-linear, usado por Efron p/ métodos lineares) o modelo linear
  lidera legitimamente; o `reg_svm(cost=40)` do exemplo overfita nos DOIS (fiel), mantido.
- O "R parecia melhor em tudo" com splits diferentes era **variância do split** — na mesma
  partição a paridade é boa (menos o bug do dtree, agora corrigido).

### Testes pareados R↔Python — classificação + clustering (2026-11-22)
Dois lotes de paridade sobre as MESMAS linhas (scripts em `daltoolbox/develop/parity/`).

**Classificação** (dataset `breast_cancer`, mais difícil que iris: binário, mas atributos em
escalas muito heterogêneas — expõe fidelidade de escalonamento que iris mascara). Split
idêntico exportado p/ CSV. Paridade boa em tudo **exceto `cla_glm`**: Python 0.947, R **0.044**
(predições invertidas).
- **Bug real no R `cla_glm.R`:** o `glm` binomial modela P(último nível do fator), mas o
  `predict` atribuía `prob` à coluna de `positive` incondicionalmente → inversão quando
  `positive` não é o último nível (o exemplo da roxygen escapava por sorte). **Fix (R, com OK):**
  relevel da resposta em `fit.cla_glm` p/ `positive` ser o último nível → R 0.956 (≈ Python). O
  Python já estava correto (sklearn mapeia rótulos por coluna). Guarda: `tests/test_quality_classification.py`.

**Clustering** (iris, concordância da partição com `Species`: entropy/purity). Paridade boa em
Pam/Cmeans/Gmm/Hclust; **dois descasamentos corrigidos**:
1. **KMeans** — R `stats::kmeans` usava `nstart=1` (init único → ótimo local ruim, purity ~0.67
   em ~3/8 execuções); Python já robusto (`n_init=10`). **Fix (R, com OK):** `cluster_kmeans(k,
   nstart=10)`. Agora 0.893 nos dois.
2. **DBSCAN** — Python 0.687 vs R 0.927. Duas causas no **Python**: (a) off-by-one na estimativa
   de `eps` — `NearestNeighbors(n_neighbors=min_pts)` inclui o próprio ponto → `min_pts+1`
   (= `kNNdist(k=min_pts)` do R); (b) **`FitCurvatureMax/Min`** usava spline GCV
   (`make_smoothing_spline`), cuja 2ª derivada é ondulada e põe o máximo na cauda (cotovelo
   errado, x=145/eps=0.60). Reescrito com a formulação de **Reinsch (df=2)**, replicando
   `smooth.spline(x,y,df=2)` do R (penalidade resolvida p/ `trace(S)=2`; cap de 400 pts +
   eigendecomposição p/ ser O(n) em n grande). Cotovelo volta ao joelho (x=97/eps=0.374 ≈ R
   x=98/0.387) → DBSCAN 0.920 ≈ R 0.927. Fidelity da curvatura `approximate → exact`.
- **Ripple verificado:** a curvatura é usada também em PCA (seleção de componentes), threshold de
  padrões e tuning de smoothing. Testes de lá (frouxos) seguem verdes; a seleção de bins do
  `SmoothingInter(n=1:20)` em iris mudou 2→**8**, agora **idêntica ao R `smoothing_inter`** (8) —
  confirmando ganho de fidelidade, não regressão. Exemplos afetados regenerados pelo conversor
  (dimensionality/density/smoothing/patterns).
- Guarda: `tests/test_quality_clustering.py` (DBSCAN não-degenerado, partitional recupera
  estrutura, cotovelo de `FitCurvatureMax` no joelho). **225 testes verdes**, ruff limpo.
- Dev-only no Python quanto a exemplos; **mudança de código** Python (dbscan/curvatura) + R
  (cla_glm, cluster_kmeans). Sem bump de versão até fechar o ciclo de paridade.

### Testes pareados R↔Python — feature-selection (2026-11-22)
Lote de paridade dos seletores (comparando o CONJUNTO selecionado nas mesmas linhas;
diabetes p/ regressão, breast_cancer p/ classificação; scripts `featsel*` em
`daltoolbox/develop/parity/`). **Nenhum bug — port fiel.** Resultado:
- **Paridade exata (idêntica ao R):** `corr(0.9)`, `fss`, `relief(top10)`, `info_gain(top10)`
  e `stepwise(gaussian)` em diabetes. Os seletores nativos (numpy) e os de LinearRegression
  são determinísticos e estáveis entre versões.
- **Aproximados por natureza (não-bug):** `lasso` (penalidade `lambda` escolhida por CV —
  `cv.glmnet` nfolds=10 aleatório vs `LassoCV` cv=5; grids diferentes → conjunto difere mas
  ambos válidos) e `stepwise(binomial)` em breast_cancer (quase-separação logística → logLik→0,
  AIC mal-condicionado + solver sklearn vs IRLS do glm). **Prova de que o stepwise é fiel:**
  `stepwise(gaussian)` em diabetes dá idêntico nos dois (`bmi,bp,s1,s2,s5,sex`); a diferença no
  binomial é só numérica/separação.
- Guarda `tests/test_quality_feature_selection.py` (6 testes: conjuntos exatos dos
  determinísticos vs R; lasso/stepwise-binomial checados frouxos). **231 testes verdes.**
- Sem mudança de código (nos dois repos); só testes + scripts de paridade.

## Estado atual (base para o `planejamento.md`)
- Última release: **pydaltoolbox 0.23.0** (PyPI) — fix de qualidade do `RegDtree`. daltoolbox
  sem `GGally`; API de padrões simplificada; exemplos 1:1 (107=107); dataset do R = Diabetes.
- Cobertura: **todo o daltoolbox portado** (Pendências zeradas). **225 testes verdes** + ruff
  limpos; notebooks renderizam. Testes pareados: **regressão + classificação + clustering** feitos
  (parity ok; bugs achados e corrigidos: RegDtree, cla_glm, kmeans nstart, dbscan eps/curvatura).
- Extras opcionais: `[patterns]` (mlxtend), `[xgboost]`, `[viz]` (matplotlib). Núcleo:
  numpy/pandas/scikit-learn.
