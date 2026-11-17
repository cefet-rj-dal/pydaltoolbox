# Planejamento — Port do daltoolbox (R) para pydaltoolbox (Python)

> Documento vivo. Atualizado a cada checkpoint. Objetivo: você saber **todos os
> passos**, o que já foi feito, o que vem, e por quê — sem surpresas.

Última atualização: 2026-11-17

---

## 1. O todo: o ecossistema DAL

`pydaltoolbox` **não é um fim em si**. Ele é a **fundação** de uma família:

```
          pydaltoolbox  (fundação: Linha de Experimento)
                 │
        ┌────────┴────────┐
   pytspredit         pyharbinger
 (previsão de TS)   (detecção de eventos em TS)
```

No R: `harbinger` depende de `daltoolbox` **e** de `tspredit`; `tspredit` depende de
`daltoolbox`. Portanto a estratégia é: **publicar o `pydaltoolbox` primeiro, expondo
exatamente a superfície que os dois downstream precisam**, para depois viabilizar
`pytspredit` e `pyharbinger`.

Consequência prática: **priorizamos o que harbinger/tspredit consomem** e **adiamos
o resto** para versões incrementais posteriores.

---

## 2. Superfície do daltoolbox que os downstream realmente usam

Levantado por análise do código-fonte de `C:\Git\dal\harbinger` e `C:\Git\dal\tspredit`:

| Símbolo do daltoolbox | tspredit | harbinger | Precisa na v1? |
|---|---|---|---|
| `dal_transform` (base) | muito (58×) | sim (4×) | ✅ **sim** |
| `regression` (base + evaluate) | muito (17×) | muito (11×) | ✅ **sim — crítico** |
| `predictor` (base) | sim (10×) | sim | ✅ **sim** |
| `classification` (base) | 1× | sim (4×) | ✅ **sim** (já feito) |
| `dal_base`, `set_params`, `action` | sim | sim | ✅ **sim** (feito) |
| `fit`/`transform`/`predict`/`inverse_transform` | muito | sim | ✅ **sim** (feito) |
| `sample_random`, `train_test`, `k_fold` | sim (3×) | — | ✅ **sim** (feito) |
| `dal_tune` (base de tuning) | sim (3×) | — | ✅ **sim** |
| `minmax` / `zscore` | conceito-base | `minmax` (1×) | ✅ **sim** |
| `autoenc_base_e` / `autoenc_base_ed` | — | sim (5×) | ✅ **sim** (só a base identidade) |
| `adjust_data.frame` / `adjust_*` | sim | sim | ✅ **sim** (feito) |
| **clusterer / cluster_\*** | — | — (usa `stats::kmeans` direto) | ❌ adiar |
| **pattern mining (pat_\*)** | — | — | ❌ adiar |
| **gráficos (grf_\* / plot_\*)** | tem o próprio | tem o próprio | ❌ adiar |
| **feature selection / imputation / outliers / smoothing** | — | — | ❌ adiar |

Observações que **resolveram pendências**:
- `do_fit`/`do_predict` são do **próprio tspredit** (em `ts_reg.R`), não do daltoolbox.
- `autoenc_base_e/ed` no daltoolbox são **stubs de identidade** (sem treino). O backend
  real (torch/keras) fica num pacote companheiro → **pydaltoolbox não precisa de deep
  learning na v1**.
- harbinger e tspredit têm **plotagem própria** → os gráficos do daltoolbox não são
  dependência de ninguém → podem ser 100% adiados.

---

## 3. Princípios

- **Não reimplementamos algoritmos.** Reusamos `scikit-learn`, `numpy`, `pandas`.
  Nossa contribuição é a **interface uniforme** (`fit`/`action`/`evaluate`) via
  *adapters finos*.
- **Dependências mínimas na v1.** Só `numpy`, `pandas`, `scikit-learn`. Tudo que for
  mais pesado ou de nicho (xgboost, gráficos, mineração de padrões) vira **extra
  opcional** ou **versão futura** — equivalente ao `Suggests:` do R.
- **Incremental.** Uma fase/sub-bloco por vez; você confere antes de escalar. Sempre
  terminando com `ruff` limpo e `pytest` verde.
- **Fiéis à semântica do R; idiomáticos na superfície** quando o público Python justifica.

---

## 4. Decisões travadas

| Tema | Decisão |
|---|---|
| Estilo de API | Métodos de instância; `fit` retorna `self` (encadeável) |
| Nomenclatura | **PascalCase** (PEP 8): `ClaDtree`, `SampleRandom`, `MinMax` |
| Modelo de dados | `pandas.DataFrame` |
| Entrada do `predict` | Aceita frame inteiro **ou** só features (ignora o alvo); exemplos passam só features |
| Dependências v1 | `numpy`, `pandas`, `scikit-learn` apenas |
| Gráficos | **Adiados** para versão futura (downstream têm plotagem própria) |
| Mineração de padrões | **Adiada**; quando vier, libs mais usadas/leves (`mlxtend`); sequências = ponto futuro |
| Autoencoders | Só as classes-base identidade; **sem backend de DL** na v1 |
| Publicação | PyPI (ver [deploy.md](deploy.md)) |

---

## 5. Arquitetura (árvore de classes)

```
DalBase ─ fit / action / set_params
├── DalTransform ───── action → transform / inverse_transform
│   ├── DataSample (train_test, k_fold)
│   ├── MinMax, ZScore                         [v1]
│   └── AutoencBaseE / AutoencBaseEd (identidade) [v1]
├── DalLearner ─────── action → predict ; + evaluate
│   ├── Predictor ───── fit registra as features (x)
│   │   ├── Classification(attribute, slevels) → acc/prec/recall/f1   [v1]
│   │   └── Regression(attribute)              → mse/smape/R2         [v1 crítico]
│   ├── Clusterer ───── action → cluster                              [futuro]
│   └── PatternMiner ── action → discover                            [futuro]
└── DalTune ────────── select_hyper (ClaTune, RegTune)                [v1]
```

---

## 6. Mapeamento de bibliotecas (R → Python)

| daltoolbox (R) | Python | Quando |
|---|---|---|
| `tree`, `rpart`, `class`, `FNN`, `e1071`, `randomForest`, `nnet` | `scikit-learn` | v1 |
| `caret`, `prcomp` | `sklearn.model_selection/preprocessing/decomposition` | v1 |
| data.frame / factor / genéricos S3 | `pandas` / classes + herança | v1 |
| `xgboost`, `glmnet`, `adabag`, `ipred` | `xgboost`, `sklearn` | futuro / extra |
| `cluster`, `dbscan`, `mclust` | `sklearn.cluster`, `GaussianMixture` | futuro |
| `arules`, `arulesSequences` | `mlxtend`, (sequências: ponto futuro) | futuro |
| `ggplot2` | `matplotlib` | futuro |

---

## 7. Estratégia de garantia (três camadas)

1. **Testes unitários** → `pytest` por componente: `fit → action → evaluate`.
2. **Testes integrados** → `pytest` ponta-a-ponta (ex.: `test_tutorial_01.py`).
3. **Documentação executável (conferência visual)** → notebooks que viram `.md`
   processado com a **saída real embutida**, para você inspecionar o efeito.

### Camada 3 — documentação executável (análoga ao Rmd/examples do R)

| Papel | daltoolbox (R) | pydaltoolbox |
|---|---|---|
| Fonte executável | `Rmd/**/*.Rmd` | `notebooks/**/*.md` (jupytext markdown) |
| Doc. processada | `examples/**/*.md` | `examples/**/*.md` |
| Conversor | `Rmd/conversor.R` | `tools/build_examples.py` |
| Motor | `knitr` | `jupytext` + `nbconvert --execute` |

**Comandos:**
```bash
pip install -e ".[docs]"              # ferramentas (jupytext, nbconvert, ipykernel)
python tools/build_examples.py            # reconstrói tudo
python tools/build_examples.py tutorial   # só uma subpasta
```
A execução **falha se houver erro** → serve de smoke test; depois entra no CI.

### Ritual a cada fase (como você confere)
1. Implemento componentes + testes `pytest`.
2. Crio/atualizo o notebook do tema em `notebooks/`.
3. Rodo `python tools/build_examples.py` → gera `examples/.../*.md`.
4. Reinstalo (`pip install -e .`) e te aponto o `.md` renderizado.
5. **Você confere o efeito real por lá** antes de escalarmos.

---

## 8. Roadmap

Legenda: ✅ feito · 🔜 próximo · ⬜ pendente (v1) · 🔭 versão futura

### BLOCO A — Fundação v1 (desbloqueia pytspredit + pyharbinger)

#### ✅ Fase 0 — Scaffolding
- [x] `pyproject.toml`, layout `src/`, `__init__.py`
- [x] `deploy.md` (guia PyPI), `README.md`

#### ✅ Fase 1 — Núcleo da Linha de Experimento
- [x] `DalBase`, `DalTransform`, `DalLearner`, `Predictor` (`base.py`)
- [x] Helpers `adjust_*` (`adjust.py`)

#### ✅ Fase 2 — Amostragem
- [x] `DataSample`, `SampleRandom` (`train_test`, `k_fold`), `train_test_from_folds`
- [ ] `SampleStratified` *(restante)*

#### ✅ Fase 3 — Slice vertical de classificação
- [x] `Classification` base + `evaluate`
- [x] `ClaDtree`, `datasets.iris()/boston()`
- [x] Teste integrado **Tutorial 01**

#### 🔜 Fase 4 — Normalização (transformações base)
- [ ] `MinMax` (fit/transform/inverse_transform) ← `trans_norm_minmax.R`
- [ ] `ZScore` ← `trans_norm_zscore.R`
- **Por quê agora:** harbinger usa `minmax`; é a base conceitual do `ts_norm` do tspredit.

#### ⬜ Fase 5 — Regressão **(crítico p/ downstream)**
- [ ] `Regression` base + `evaluate` (mse/smape/R2) ← `reg_regression.R`
- [ ] `RegLm`, `RegDtree`, `RegKnn`, `RegRf`, `RegSvm`, `RegMlp` ← `reg_*.R`
- **Por quê:** os preditores de série do tspredit **estendem `regression`** e envelopam
  regressores do sklearn. É a dependência mais usada pelos dois pacotes.

> 🚀 **Marco de publicação:** o **primeiro deploy real no PyPI** (`0.1.0`) acontece
> **ao concluir esta fase** (decisão de 2026-11-17). O pipeline de release já foi
> validado (build + `twine check` + instalação em venv limpo). Eduardo faz o upload
> (`twine`, token próprio); publicação é instantânea e sem prazo no PyPI. Releases
> seguintes (`0.2.0`...) acompanham as fases posteriores.

#### ⬜ Fase 6 — Tuning
- [ ] `DalTune` base + `select_hyper` ← `dal_tune.R`
- [ ] `ClaTune` ← `cla_tune.R` · `RegTune` ← `reg_tune.R`
- **Por quê:** o `ts_tune`/`ts_integtune` do tspredit estende `dal_tune`.

#### ⬜ Fase 7 — Bases de autoencoder (identidade)
- [ ] `AutoencBaseE`, `AutoencBaseEd` como `DalTransform` identidade ← `autoenc_base_*.R`
- **Por quê:** harbinger (`han_autoencoder`) herda de `autoenc_base_ed`. Trivial, sem DL.

#### ⬜ Fase 8 — Fechamento da v1
- [ ] Classificadores que harbinger `hanc_ml` precise (provável: 1–2 além do dtree)
- [ ] Revisão da API pública, docstrings, `__all__`
- [ ] Suíte de testes unit + integrados consolidada (`ruff`/`pytest` verdes)
- [ ] **Release de consolidação da v1** no PyPI (ex.: `0.2.0`), via `deploy.md`

> 🎯 **Marco:** concluído o Bloco A, `pytspredit` e `pyharbinger` já têm fundação para começar.

### BLOCO B — Versões futuras (incrementais, pós-v1)

#### 🔭 Classificação completa
- `ClaKnn`, `ClaNb`, `ClaRf`, `ClaSvm`, `ClaMlp`, `ClaGlm/Glmnet/Multinom`,
  `ClaMajority`, `ClaBagging`, `ClaBoosting`, `ClaRpart`, `ClaXgboost` (extra).

#### 🔭 Clustering
- `Clusterer` + `cluutils` (métricas) + `ClusterKmeans/Pam/Dbscan/Gmm/Hclust/Cmeans/Louvain` + `CluTune`.

#### 🔭 Transformações avançadas
- Encoding, imputação, outliers, smoothing, seleção de atributos, PCA, balanceamento,
  heurísticas de curvatura.

#### 🔭 Mineração de padrões
- `PatApriori`/`PatEclat` (via `mlxtend`); **sequências** (`PatCspade`) = ponto a decidir
  (lacuna no ecossistema Python); `PatDara`, filtros de regras.

#### 🔭 Gráficos
- ~20 funções `plot_*` (matplotlib). Entregues aos poucos conforme a necessidade.

---

## 9. Próximos passos nos pacotes irmãos (visão macro)

Depois do Bloco A:
1. **pytspredit** — portar `ts_data`/`ts_projection` (janela deslizante), `ts_norm_*`,
   regressores de TS (knn/mlp/rf/svm/elm/lm) sobre a `Regression` do pydaltoolbox,
   `ts_tune`/`ts_integtune` sobre o `DalTune`, filtros e augmentation.
2. **pyharbinger** — portar o `harbinger` base (detecção de eventos), detectores
   estatísticos/ML, ensembles, avaliação (`har_eval`), reusando `Regression`,
   `Classification` e `AutoencBaseEd` do pydaltoolbox + os regressores do pytspredit.

*(Detalhamento desses dois será feito em planejamentos próprios quando chegarmos lá.)*

---

## 10. Como trabalhamos

- Incremental, um passo por vez; você confere antes de escalar.
- Verde sempre (`ruff` + `pytest`).
- Decisões sempre registradas aqui.
- Sem commits/push sem você pedir.

---

## 11. Pendências / decisões em aberto

- [ ] `datasets.boston()` usa `fetch_openml` (baixa da web). Trocar por `diabetes` do
      sklearn (offline) como dataset de regressão da Fase 5?
- [ ] `xgboost` como extra opcional (quando chegar na classificação completa).
- [ ] Backend de mineração de padrões sequenciais (versão futura).
