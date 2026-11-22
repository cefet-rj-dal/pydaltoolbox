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
(`[xgboost]`, `[viz]`...). Releases podem ser bem pequenos (muitos ciclos são
bem-vindos). O roadmap detalhado, com a tabela de decisão *dependência vs. nativo*, está
na seção 8 (Bloco B). O Ciclo 2 será refinado em conjunto.

**Dependências do núcleo (até a 1.0.0):** `numpy`, `pandas`, `scikit-learn` apenas. Tudo
além disso é **extra opcional** (`pip install pydaltoolbox[graph,xgboost,viz]`).

---

## 2b. Princípio de longo prazo — back-implementation (Python = releitura, R = backport)

> **Direção estratégica (2026-11-20, Eduardo), comum aos três pacotes DAL
> (`pydaltoolbox`, `pytspredit`, `pyharbinger`).**

A família DAL adota, ao longo do tempo, uma **inversão de papéis**:

- **A versão Python é a releitura** — reimplementação própria (nativa) dos
  trabalhos anteriores, **dona dos algoritmos**, não apenas um *wrapper* de libs
  de terceiros.
- **A versão R passa a ser um backport da versão Python** — o Python vira a
  referência canônica e o R acompanha (invertendo a direção histórica R→Python).

Isso deixa o trabalho **sólido e autocontido**: a paridade fica **sob nosso
controle**, não refém dos defaults de pacotes de terceiros.

**Trilha de back-implementation (gradual e seletiva):** trocar dependências de
terceiros pela nossa implementação quando a lib **falta / não é mantida / causa
atrito (wheel/compilação) / impede a paridade**. Como `pydaltoolbox` é a
**fundação**, ele é o lugar natural para as implementações próprias dos
algoritmos-base; as libs canônicas e estáveis (`scikit-learn`, `scipy`, `numpy`,
`pandas`) **seguem embrulhadas** salvo atrito real.

**Ação 1 — gate "diferença pequena":** ao portar nativo, **medir o gap vs. o
pacote de referência** (R / lib original) nos exemplos; aceitar só se pequeno,
documentando a tolerância. Gap grande → investigar antes.

**Trade-off explícito:** isto flexiona o princípio "não reimplementar" acima —
reimplementamos nos casos citados e seguimos embrulhando as libs canônicas.

**⚠️ Não é regra cega — é planejamento de DISCUSSÃO (caso a caso).** A inversão
*Python = releitura / R = backport* é a **tendência**, não um absoluto, e **não se
aplica a tudo**. O alvo são **pacotes atípicos / de nicho** (sem equivalente mantido,
que causam atrito de build, ou cuja auto-seleção impede a paridade) — **não** as libs
canônicas e comuns. Quando já existe uma implementação **madura** de um dos lados e a
**diferença é pequena**, mantém-se o que há — ex.: o **MLP do `scikit-learn` não vira
back-implementation**, porque o R já tem um MLP nativo excelente e a diferença é
pequena. Cada candidato é **decidido em conjunto**, caso a caso, nunca por automação.

### 2b.1 — Duas fontes de valor + inversão de atribuição (2026-11-21, Eduardo)

Uma reimplementação nativa vale a pena por **uma** destas duas razões (basta uma):

1. **Inovação (++)** — otimização ou melhoria de qualidade **medível** vs. a
   referência. **Simplificar para ficar mais rápido conta como inovação** quando a
   perda de qualidade é nula ou documentadamente pequena (não precisa ser mais
   "esperto", pode ser mais enxuto/rápido com o mesmo resultado).
2. **Posse / remoção de dependência** — mesmo **sem** ++, portar do Python canônico
   para o R faz o R deixar de depender da lib de nicho (`e1071`, `cluster`, …) e virar
   **código nosso citando o artigo**. Autocontido, paridade sob nosso controle, sem
   atrito de build. (É o caso do **c-means**: tirar o `e1071` já se justifica; ++ de
   inicialização robusta — múltiplos restarts / seeding k-means++ — fica opcional.)

**Inversão de atribuição (importante).** Quando o R passa a ser backport do Python, a
documentação inverte o sentido:
- **citar o artigo** como fonte do algoritmo (Bezdek 1981 / Kaufman-Rousseeuw 1990 /
  Chawla 2002 / Kononenko 1994 …);
- declarar **"reimplementação independente"**, não "usa/embrulha o pacote xyz";
- "inspirado em `e1071`/`cluster`" vira nota de reconhecimento, não dependência.
- O `Porting:` do Python e o roxygen do R refletem isso (Fidelity: `adapted`, nativo).

**Template de doc "++" (aprovado 2026-11-21)** — toda componente back-implementada
documenta **original + inovação**, com atribuição invertida:
- Python `Porting:` → `Canonical implementation; the R <file>.R is a backport of this.
  Fidelity: adapted.` + bullet `Inspired by <X> and evolved with/to <Y> (see References).`
  + bullet `Innovation (++): <o que melhora> ... measured <evidência>` (números concretos
  quando houver, ex.: "~20x faster at k=20, identical result"). **Descrever o algoritmo, só.**
- R roxygen `@details` → `Inspired by <X> and evolved with/to <Y> (see references).` + a
  inovação + evidência; `@references` inclui os artigos.
- **PROIBIDO (Eduardo repetiu várias vezes — PARAR de reintroduzir):** nunca escrever
  "not a wrapper of X", "native implementation" / "native numpy", "no external/extra
  dependency", "no longer calls/uses/depends on X", "carries no dependency", "X runs a
  single start", nem qualquer alfinetada no que o pacote substituído faz. Tudo isso
  pressupõe cópia e é ruído auto-justificativo. Dizer SÓ: o que o algoritmo faz + "inspired
  by X" (isso sinaliza evolução, e é desejado) + References. O fato de ter removido uma
  dependência vai na **mensagem de commit**, não no docstring.
- **Citar só o artigo-base, não o pacote substituído (Eduardo 2026-11-21)** — porém apenas
  após **conferir que o nosso código difere da fonte do pacote** (outra linguagem / outra
  mecânica), para ser implementação-a-partir-da-literatura, não cópia. Conferido no fss: o
  `leaps` roda em **Fortran compilado** e o nosso é R puro com `lm` → citar só Hocking/Miller,
  sem "inspirado no leaps". Fazer essa conferência em toda back-implementation.

**Bônus de licença.** Reimplementar a partir do **artigo** (+ nosso código) é limpo:
não herdamos a licença do código de terceiros. Por isso toda back-implementation deve
ter **alguma diferença intencional e documentada** — nunca cópia do código-fonte alheio.

**Fluxo acordado:** *estudar otimização no Python → validar o ++ (ou a paridade) →
backport para o R tirando a dependência → atualizar atribuição nos dois lados.*

**Candidatos priorizados (ciclo de consolidação — ordem: PAM → fss → ~~kNN~~ → Relief → SMOTE → caret):**
| Candidato | Situação | Tipo / alvo | Status |
|---|---|---|---|
| **PAM** (`ClusterPam`) | SWAP de livro-texto | **++ FastPAM1** (mesmo resultado, ~20× em k=20) + remove `cluster` do R | ✅ feito (py 10677bd / R 583222b) |
| **fss** | leaps forward (= guloso) | **posse** (não é ++): greedy nativo remove `leaps` do R | ✅ feito (py 68d561c / R 6c8cd8e) |
| **Relief** | *"simplified RELIEF"* (1 hit/miss) | **++ ReliefF completo**: k vizinhos + prior multiclasse + seed (real>ruído 72%→100%) | ✅ feito (py fd7802e / R fc662f3) |
| **kNN** | `class::knn` + `FNN::knn.reg` | **MANTER** (como o nnet/MLP): FNN dá KD-tree rápido, nativo seria força-bruta mais lento; zero ++ | ⏭️ pulado (Eduardo 2026-11-21) |
| **SMOTE** (`BalOversampling`) | SMOTE local simplificado | **++ Borderline-SMOTE / ADASYN + SMOTE-NC** (F1 minoria ~0.47 vs ~0.44; sobretudo expansão de capacidade) | ✅ feito (py 9a6bda0 / R d488ea8) |
| **caret** | wrappers | **posse**: findCorrelation + up/downSample + createDataPartition removem `caret` **inteiro** (findCorrelation idêntico ao caret) | ✅ feito (py 4741f0e / R a0be70f) |
| **c-means** (`ClusterCmeans`) | FCM nativo | **posse + ++ init robusta** (`nstart`): single-start erro varia (769.5±6.2) vs nstart=15 (765.4±0.0); cmeans deixa de usar `e1071` (fica p/ SVM/NB) | ✅ feito (py 662850a / R bad803c) |

**Lição (2026-11-21):** nem toda back-implementation vira ++. O **fss** mostrou que o alvo
às vezes já é o mesmo algoritmo (leaps `method="forward"` é guloso) → valor é só **remover a
dependência**. E o **kNN** mostrou o oposto: às vezes é melhor **manter** a dep (o `FNN` dá
busca KD-tree que não superamos sem implementar uma KD-tree própria; sem ++ e com risco de
regressão). Decidir caso a caso, sem forçar um ++ inexistente nem remover dep útil.

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

**Ritual por fase (release):** implemento + testes → notebook do tema + build →
bump de versão (`pyproject.toml` + `__version__`) → **atualizar a lista de features do
`README.md` da raiz** → `ruff`+`pytest` verdes → commit + push → você confere no GitHub →
`twine upload`. *(A versão no README é mostrada pelo badge do PyPI — não fixar número no
texto; manter só a lista de componentes em dia.)*

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
  classes que precisam de **extra opcional** (`[xgboost]`, `[viz]`, ...).
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
| 1.5 | `0.6.0` ✅ | Clustering nativo (Pam, Cmeans) |
| 1.6 | `0.7.0` ✅ | Seleção de atributos (Relief + fss/stepwise) |
| 1.6b | `0.8.0` ✅ | **Consolidação ++/posse** (FastPAM, ReliefF, Borderline/ADASYN, c-means nstart; remove cluster/leaps/caret do R) |
| 1.7 | `0.9.0` ✅ | Padrões: base + itemsets + regras (`PatApriori` via mlxtend `[patterns]`; eclat removido dos dois repos) |
| 1.8a | `0.10.0` ✅ | Padrões: rule filters base (`RuleFilter`/`None`/`Dara`) + DARA (`dara`, divergent association rules) |
| 1.8b | `0.11.0` ✅ | Padrões: `RuleFilterInterest` + thresholds (`SupportThreshold`/`ConfidenceThreshold`, auto `supp=0`/`conf=0`) |
| 1.9 | `0.12.0` ✅ | Padrões: sequências (`PatPrefixSpan`, PrefixSpan nativo; fim do Ciclo 1) |
| 2.1 | `0.13.0` ✅ | `ClaXgboost` `[xgboost]` (wrapper `XGBClassifier`; paridade R no iris) |
| ~~2.2~~ | — | ~~`ClusterLouvainGraph` `[graph]`~~ **REMOVIDO** (má encaixe tabular; `clu_louvain`+`igraph` tirados do R também) |
| 2.3 | `0.14.0` ✅ | Gráficos: distribuição + comparação (11 `plot_*`, matplotlib `[viz]`) |
| 2.4 | `0.15.0` ✅ | Gráficos: relação + temporais + estrutura (10 `plot_*`). Python feito; **remoção do `plot_pair_adv` do R + porte p/ o curso PENDENTE** |
| ~~2.5~~ | — | ~~*(opcional)* Padrões: backend `mlxtend` `[patterns]`~~ (já antecipado no 1.7) |
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
| ~~`ClusterLouvainGraph`~~ | ~~`networkx`~~ | ~~Louvain~~ | **REMOVIDO** (má encaixe tabular; tirado do R também — 2026-11-22) | — |
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

#### 1.6 — Seleção de atributos → `0.7.0` ✅
| Classe | Backend nativo | Arquivo R |
|---|---|---|
| `FeatureSelectionFss` | numpy + sklearn `LinearRegression` (forward por R² ajustado) | `trans_feature_selection_fss.R` |
| `FeatureSelectionStepwise` | nativo: sklearn `LogisticRegression`/`LinearRegression` + AIC calculado à mão + busca gulosa | `trans_feature_selection_stepwise.R` |
| `FeatureSelectionRelief` (ReliefF) | numpy | `trans_feature_selection_relief.R` |

**Decisão (2026-11-21, com Eduardo):** quebra limpa de nome (pré-1.0). O antigo
`FeatureSelectionStepwise` (que implementava o `fss`) foi movido para
`FeatureSelectionFss`; `FeatureSelectionStepwise` agora é o stepwise-GLM correto
(espelho do `stats::step`: `direction` forward/backward/both, `family` binomial-default
ou gaussian, seleção por AIC). **Backend nativo, sem statsmodels** — AIC = `2k − 2·logLik`
com logistic/linear do sklearn. Sem dependência nova. Relief é ReliefF nativo em numpy
(sem `skrebate`).

#### ✅ 1.7 — Mineração de padrões: base + itemsets + regras → `0.9.0` (publicado)
| Classe | Backend | Arquivo R |
|---|---|---|
| `PatternMiner` (base, + `discover`) | — | `pat_pattern_miner.R` |
| `PatUtils` (utilitários: filtros + métricas) | — | `pat_utils.R` |
| `PatApriori` (itemsets **e** regras) | `mlxtend` → extra `[patterns]` | `pat_apriori.R` |

> **Decisões 1.7 (com Eduardo):** backend = **mlxtend** como extra opcional `[patterns]`
> (embrulho fino), não nativo; o antecipamos do passo 2.5. Entrada = **DataFrame one-hot**
> (colunas = itens). Escopo = itemsets **+ regras** (confidence/lift, lhs/rhs, include/
> exclude, quality_filter) já no 1.7 — só a auto-estimação de thresholds (`supp=0`) e os
> rule filters do `pat_rule_filter.R` ficam p/ 1.8.
> **ECLAT removido dos dois repos:** `arules` e `mlxtend` só compartilham o `apriori`
> (arules tem eclat mas não fpgrowth sem o nicho não-CRAN `fim4r`; mlxtend tem fpgrowth mas
> não eclat). Em vez de um "Eclat" com motor fpgrowth (nome enganoso) ou de puxar `fim4r`,
> ficamos **só com apriori nos dois**; o `pat_eclat.R` saiu do R. Desempenho de itemsets
> grandes: revisitar no futuro se virar gargalo.

#### ✅ 1.8a — Mineração de padrões: rule filters base + DARA → `0.10.0` (publicado)
| Classe/função | Backend | Arquivo R |
|---|---|---|
| `RuleFilter` base + `RuleFilterNone` | pandas | `pat_rule_filter.R` |
| `RuleFilterDara` + `dara()` + `dara_score()` (divergent association rules) | numpy/pandas | `pat_dara.R`, `pat_rule_filter.R` |
| integração: `PatApriori(rule_filter=...)` (só `target='rules'`) | — | `pat_apriori.R` |

> **DARA** (`[[back-implementation-initiative]]`): algoritmo **do grupo** (Baroni et al.,
> 2020, *J. Biomedical Informatics* 108:103512). Back-implementation **canônica** — Python
> vira a referência, R vira backport, atribuição cita o **paper** nos dois lados. Counter de
> divergência validado contra a Tabela 1 do artigo e contra o `pat_dara` do R (idênticos).
> Entrada = regras do `PatApriori` sobre itens `atributo=valor` (one-hot `prefix_sep='='`).

#### ✅ 1.8b — Mineração de padrões: rule filters de interesse + thresholds → `0.11.0` (publicado)
| Classe/função | Backend | Arquivo R |
|---|---|---|
| `RuleFilterInterest` (lift/kulc/ir/support/confidence/count) | mlxtend measures | `pat_rule_filter.R` |
| `SupportThreshold` (curvature/min_count/sqrt/quantile/density) | numpy + `FitCurvatureMax` | `pat_threshold.R` |
| `ConfidenceThreshold` (rhs_baseline/support_adaptive/fixed) | numpy | `pat_threshold.R` |
| `PatApriori`: `supp=0`/`conf=0` auto-estimados no `fit()` via estratégias | — | `pat_apriori.R` |

> Medidas de interesse (lift/kulc/ir) validadas **idênticas** ao `arules::interestMeasure`.
> `kulc`=kulczynski, `ir`=imbalance derivados das colunas do mlxtend; `count`=support×n (n em
> `rules.attrs['n_transactions']`). ⚠️ Delta pré-existente: o `association_rules` do mlxtend
> gera regras com consequente de vários itens que o `arules::apriori` não emite por padrão.

#### ✅ 1.9 — Mineração de padrões: sequências → `0.12.0` (publicado · fim do Ciclo 1)
| Classe | Backend | Arquivo R |
|---|---|---|
| `PatPrefixSpan` | PrefixSpan nativo (numpy/pandas) | `pat_prefix_span.R` |

> **Back-implementation por posse de dependência:** PrefixSpan nativo (Pei et al., 2001) nos
> **dois** repos; o R deixou de depender de `arulesSequences` (`pat_cspade.R` removido,
> `arulesSequences` fora do DESCRIPTION). Entrada = sequências de itemsets (lista) ou
> DataFrame `sequenceID`/`eventID`/`item` (como `read_baskets`). Validado **idêntico ao
> `arulesSequences::cspade`** no exemplo clássico de Zaki: support=0.5 → 18 padrões;
> maxgap=5 → 11; maxgap=10 → 17; mingap=10 → 12 (projeção por conjunto de posições, não
> primeira-ocorrência, para não perder co-ocorrências de itens em elementos posteriores).

> ✅ **Fim do Ciclo 1 (0.12.0): todo o daltoolbox coberto, exceto XGBoost e gráficos**
> (que são o Ciclo 2), com o pacote instalável só com o núcleo (PrefixSpan é numpy/pandas
> puro; `mlxtend` só p/ apriori/regras). PrefixSpan (1.9) entregue e validado idêntico ao
> cSPADE. (Louvain/`ClusterLouvainGraph` foi **removido** da família — ver §2b.1.)

---

### CICLO 2 — rodada temática completa (classes de maior dependência) · *a refinar juntos*

Ciclo 2 restante = **gráficos `plot_*`** (classificação extra `ClaXgboost` já entregue em
2.1/`0.13.0`). Cada arquivo R está indicado para rastreabilidade.

**Decisões de desenho dos gráficos (2026-11-22, com o Eduardo):**
- **Backend = `matplotlib`** (único dep novo no extra `[viz]`). O R é ggplot2 de ponta a
  ponta, mas a superfície de customização usada nos exemplos do daltoolbox é pequena e padrão
  (`colors`, `alpha`, `label_x/label_y`, `title`, `bin`, etc.) → matplotlib cobre tudo. Os
  casos "difíceis" têm backend matplotlib já no **core**: `pandas.plotting.scatter_matrix`
  (pair), `pandas.plotting.parallel_coordinates` (parallel), `scipy.cluster.hierarchy.dendrogram`
  (dendrogram), `imshow` (pixel/correlation), projeção `polar` (radar). Preferiu-se matplotlib
  a `plotnine` (porta do ggplot2) para não puxar statsmodels/mizani/patsy sem necessidade.
- **API = funções `plot_*` retornando o objeto do gráfico** (`Figure`/`Axes`), espelhando o R
  (onde `grf_*` devolve um ggplot que o usuário imprime). Sem estado, sem classe `DalTransform`.
- **Enxugamento (regra do Eduardo):** `plot_*` que **não é ggplot2 padrão E não tem
  correspondência limpa no matplotlib** sai do daltoolbox e vira **extensão do curso de
  mineração** em `C:\Git\eogasawara\datamining` (tutorial `03-DataVisualization`), deixando o
  pacote mais leve — mesmo padrão de Louvain/eclat/arulesViz. **Único caso: `plot_pair_adv`**
  (`GGally::ggpairs`, painéis mistos densidade+correlação+cor-por-classe; sem equivalente padrão
  no matplotlib) → **remover de daltoolbox (R) + portar para o `datamining`**. `plot_pair`
  (scatter_matrix básica) e `plot_radar` (polar) **ficam e são portados**.
- A skill `dataviz` guia paleta/tema (claro/escuro, acessível) para coesão visual.

#### 2.1 — Classificação extra → `[xgboost]` · `0.13.0` ✅ PUBLICADO
`ClaXgboost` (wrapper `XGBClassifier`, extra `[xgboost]`, paridade R no iris). Ver memória.

#### (histórico) 2.1 — mapeamento original
| Classe | Backend | Arquivo R |
|---|---|---|
| `ClaXgboost` | `xgboost` | `cla_xgboost.R` |

#### ~~2.2 — Clustering extra → `[graph]`~~ **REMOVIDO**
`ClusterLouvainGraph` (Louvain via networkx/igraph) foi **descartado** da família (2026-11-22):
má encaixe na Linha de Experimento tabular (opera sobre grafo, não DataFrame), não é consumido
a jusante (tspredit/harbinger não usam clusterer), e seria só um embrulho de dep de nicho. O
`clu_louvain.R` + `igraph` foram tirados **do R** também. Grafos podem virar extensão dedicada
no futuro, se vierem a ser modalidade de primeira classe.

#### 2.3 — Gráficos: distribuição + comparação → `[viz]` (`matplotlib`) · `0.14.0` ✅
11 gráficos (funções `plot_*` → **`Axes`**). Segue a skill `dataviz` (paleta CVD-safe de
8 hues quando `colors=None`). Subpacote `src/pydaltoolbox/viz/` (`_helpers` + `distribution`
+ `comparison`); exportados lazy (PEP 562) no `__init__` principal para não exigir matplotlib
no core. Decidido com o Eduardo: retorno `Axes`, paleta validada da skill `dataviz`.
| Função | Tipo | Arquivo R | Backend matplotlib |
|---|---|---|---|
| `plot_hist` | distribuição | `grf_hist.R` | `ax.hist` |
| `plot_density` | distribuição | `grf_density.R` | `scipy.stats.gaussian_kde` + `fill` |
| `plot_density_class` | distribuição | `grf_density_class.R` | kde por classe |
| `plot_boxplot` | distribuição | `grf_boxplot.R` | `ax.boxplot` |
| `plot_boxplot_class` | distribuição | `grf_boxplot_class.R` | boxplot agrupado |
| `plot_bar` | comparação | `grf_bar.R` | `ax.bar` |
| `plot_groupedbar` | comparação | `grf_groupedbar.R` | `ax.bar` deslocado |
| `plot_stackedbar` | comparação | `grf_stackedbar.R` | `ax.bar(bottom=)` |
| `plot_lollipop` | comparação | `grf_lollipop.R` | `ax.vlines` + `scatter` |
| `plot_pieplot` | comparação | `grf_pieplot.R` | `ax.pie` |
| `plot_radar` | comparação | `grf_radar.R` | projeção `polar` |

> Como os `plot_*` são independentes (sem classe-base com estado), não há "base de gráficos"
> a portar; utilitários comuns (aplicação de paleta, tema) ficam num `_helpers` interno do
> subpacote `viz/`. (O R `graphics.R`/`graphics_dal_graphics.R` é só agregador de `@include`.)

#### 2.4 — Gráficos: relação + temporais + estrutura → `[viz]` · `0.15.0` ✅ (Python)
10 gráficos (o `plot_pair_adv` foi **removido** — ver abaixo). Python feito: módulos
`viz/relation.py` (scatter/points/correlation/pair/parallel/pixel) + `viz/temporal.py`
(series/ts/ts_pred) + `viz/structure.py` (dendrogram); retorno `Axes` (exceto `plot_pair`,
que devolve a grade de Axes do scatter-matrix); `plot_dendrogram` recebe uma linkage do SciPy.
**Enxugamento de dependência gráfica (feito 2026-11-22):** `plot_pair_adv` e `plot_pair`
(ambos `GGally::ggpairs`) saíram do daltoolbox e viraram extensões do curso `datamining`
(pasta `codes/`, sourceadas por URL do GitHub; versões R + Python). No daltoolbox o
`plot_parallel` foi reescrito em **ggplot2 nativo** (z-score + geom_line, sem `GGally::ggparcoord`)
e o **`GGally` foi removido do DESCRIPTION**. No pydaltoolbox (**0.16.0**): `plot_pair` removido
(fica no curso), `plot_parallel` refinado com z-score por coluna → 20 `plot_*`.
| Função | Tipo | Arquivo R | Backend matplotlib |
|---|---|---|---|
| `plot_scatter` | relação | `grf_scatter.R` | `ax.scatter` |
| `plot_points` | relação | `grf_points.R` | `ax.plot`/`scatter` |
| `plot_correlation` | relação | `grf_correlation.R` | `imshow` + `scipy` linkage (`reorder="hclust"`) |
| `plot_pair` | relação | `grf_pair.R` | `pandas.plotting.scatter_matrix` |
| `plot_parallel` | relação | `grf_parallel.R` | `pandas.plotting.parallel_coordinates` |
| `plot_pixel` | relação | `grf_pixel.R` | `imshow`/`pcolormesh` |
| `plot_series` | temporal | `grf_series.R` | `ax.plot` multi-série |
| `plot_ts` | temporal | `grf_ts.R` | `ax.plot` |
| `plot_ts_pred` | temporal | `grf_ts_pred.R` | `ax.plot` sobreposto |
| `plot_dendrogram` | estrutura | `grf_dendrogram.R` | `scipy.cluster.hierarchy.dendrogram` |

#### Removido do daltoolbox → curso `datamining`
| Função | Motivo | Destino |
|---|---|---|
| `plot_pair_adv` | `GGally::ggpairs` (painéis mistos densidade+correlação+cor-por-classe); sem correspondência padrão no matplotlib | `C:\Git\eogasawara\datamining` (tutorial `03-DataVisualization`), como extensão de curso. Remover `grf_pair_adv.R` + man/examples do R e `devtools::document()`. |

#### ~~2.5 — (opcional) Padrões: backend alternativo → `[patterns]` · `0.15.0`~~ (antecipado p/ 1.7)
Antecipado: o `PatApriori` já nasceu sobre `mlxtend` no extra `[patterns]` (1.7/`0.9.0`),
em vez de nativo. Não há passo 2.5 separado.

> **Resumo do Ciclo 2:** `ClaXgboost` (2.1, feito) + **21** `plot_*` (matplotlib). Extras
> introduzidos: `[xgboost]`, `[viz]` (matplotlib). (`ClusterLouvainGraph`/`[graph]` removido;
> `plot_pair_adv` movido para o curso; `[patterns]`/mlxtend antecipado no 1.7.)
>
> **Autoencoders reais** (backend de deep learning) permanecem **fora** do pydaltoolbox:
> as bases identidade bastam; a implementação real fica num pacote companheiro, como no R.

---

### 🔭 1.0.0 — Estabilização (completa)
- Cobertura de **todas** as classes do R (ou exceções documentadas).
- Congelar a API pública; revisar docstrings e `__all__`.
- Site de documentação (equivalente ao pkgdown); paridade com os tutoriais do R.
- Passada de performance/qualidade; **promover de beta para estável**.

> **Extras opcionais previstos:** `[xgboost]`, `[viz]` (matplotlib), e `[patterns]` (mlxtend,
> já no 1.7). Núcleo enxuto (numpy/pandas/scikit-learn) até a 1.0.0.

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
