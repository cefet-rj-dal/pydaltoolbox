# Kickoff — pydaltoolbox step 2.1 → release 0.13.0 (ClaXgboost, extra `[xgboost]`)

Cole este prompt numa conversa limpa para continuar o port. É auto-contido; o resto do
contexto está na memória e em `develop/planejamento.md`.

---

## Contexto (leia primeiro)

- Repo: `c:\Git\dal\pydaltoolbox` — port Python do pacote R **daltoolbox**
  (`c:\Git\dal\daltoolbox`), liderado por Eduardo Ogasawara (CEFET/RJ, lab DAL).
- Leia a memória do projeto: `[[pydaltoolbox-port]]`, `[[back-implementation-initiative]]`,
  `[[docstring-theory-and-references]]`, `[[fix-bugs-in-both-r-and-python]]`.
- Estado: **0.12.0 é a última release** e **o Ciclo 1 está COMPLETO**. Estamos no **Ciclo 2**
  (classes de maior dependência, como extras opcionais). Confirme que
  `https://pypi.org/pypi/pydaltoolbox/0.12.0/json` está no ar — se o upload ao PyPI tiver
  ficado pendente, os artefatos estão em `dist/` e o Eduardo publica (token em
  `C:\Users\eduar\.pypirc`, seção `[pypi]`).
- Plano completo: `develop/planejamento.md` (§8 = roadmap; §2b/§2b.1 = back-implementation).

## O que já foi feito (não refazer)

- **Bloco A / v1** e **Cycle 1 (passos 1.1–1.9)** concluídos e publicados (0.1.0 → 0.12.0),
  incluindo o ciclo de consolidação (0.8.0: PAM/fss/Relief/SMOTE/caret/c-means).
- **Classificação (Cycle 1):** `Classification` base + `ClaDtree`, `ClaRpart`, `ClaKnn`,
  `ClaNb`, `ClaRf`, `ClaBagging`, `ClaBoosting`, `ClaSvm`, `ClaMlp`, `ClaGlm`, `ClaMultinom`,
  `ClaGlmnet`, `ClaMajority` (todos sklearn). **Falta só `ClaXgboost`** (este passo).
- **Mineração de padrões completa:** `PatApriori` (itemsets+regras+auto-thresholds), rule
  filters (`None`/`Interest`/`Dara`, todos `DalTransform`), `dara`, `PatPrefixSpan`
  (sequências nativas). `mlxtend` é extra `[patterns]` (só apriori/regras).
- **Limpeza no R (2026-11-22):** `ClusterLouvainGraph`/`igraph` **removidos** da família
  (má encaixe tabular); `arulesViz` (Suggests morto) removido. Então o **Ciclo 2 restante é
  só `ClaXgboost` + os gráficos `plot_*`**.

## Princípios do port (valem sempre)

- **Envolver** scikit-learn/numpy/pandas/scipy/mlxtend com a interface da Linha de
  Experimento (adaptadores finos). Nativo só p/ dep de nicho ou ganho real (++), caso a caso
  com o Eduardo (`[[back-implementation-initiative]]`). **XGBoost NÃO se reimplementa** — é
  embrulho do `xgboost`, como extra opcional.
- **Quality-faithful** (`[[pydaltoolbox-port]]`): o adaptador deve reproduzir o comportamento
  DEFAULT do backend R e ter qualidade comparável — medir vs R quando possível.
- API idiomática Python: métodos de instância, `fit` retorna `self`; **PascalCase**; pandas.
- Núcleo = numpy/pandas/scikit-learn. Extras opcionais por tema; `[patterns]` já existe.
- Docstrings OBRIGATÓRIAS (`[[docstring-theory-and-references]]`): Teoria + `References:` +
  `Porting:` (`Mirrors cla_xgboost.R. Fidelity: approximate|adapted`). ⛔ PROIBIDO "native
  implementation", "not a wrapper of X", "no dependency", "no longer uses X", alfinetadas no
  pacote substituído. Remoção de dep vai no COMMIT.

## Fluxo de trabalho

- **R roda aqui** (`C:\R\R-4.5.0`). Rodar de **arquivo `.R`** (`Rscript --vanilla script.R`),
  não `-e` inline (segfaults); repetir em segfault. `devtools::document()` regenera NAMESPACE/man.
- **Bug em qualquer lado → corrigir nos dois** (`[[fix-bugs-in-both-r-and-python]]`),
  pedindo antes de editar o R.
- **Branches:** trabalhar em `main` nos dois repos; manter só `main` sem branches órfãs.
  Uploads externos (PyPI, push de daltoolbox) podem ser bloqueados pelo auto-mode → deixar
  pronto e pedir pro Eduardo.

## Tarefa do 0.13.0 (step 2.1) — `ClaXgboost` (extra `[xgboost]`)

Fonte R: `c:\Git\dal\daltoolbox\R\cla_xgboost.R` — `cla_xgboost(attribute, params=list(),
nrounds=20)`: embrulho do `xgboost::xgb.train`; `binary:logistic` p/ 2 classes,
`multi:softprob` p/ multiclasse; `predict` devolve matriz de probabilidades por classe.

- Criar `src/pydaltoolbox/classification/xgboost.py`: **`ClaXgboost(Classification)`**
  espelhando o padrão de `ClaRf`/`ClaSvm` (ver `classification/ensemble.py`, `svm.py`).
  Embrulhar **`xgboost.XGBClassifier`** (API sklearn) — `fit`/`predict`/`predict_proba` já
  se encaixam na base `Classification`. Params sugeridos: `attribute`, `params=None`
  (dict extra p/ o XGBClassifier), `nrounds=20` (→ `n_estimators`). Objetivo é inferido
  pelo XGBClassifier (binário vs multiclasse), como no R.
- **Import lazy** com dica de instalação (igual ao `_require_mlxtend` do pattern): só importar
  `xgboost` dentro do `fit`, com `ImportError` claro apontando `pip install
  pydaltoolbox[xgboost]`.
- Adicionar o extra em `pyproject.toml`: `xgboost = ["xgboost>=1.7"]`.
- Registrar em `classification/__init__.py` e no `__init__.py` principal (`__all__`).
- **Paridade/qualidade:** comparar com o R `cla_xgboost` no iris (ou diabetes) — acurácia
  comparável. Confirmar com o Eduardo se há default do R a espelhar (nrounds=20).
- Testes pytest (`tests/test_classification.py` ou novo): fit+predict no iris, probabilidades
  somam 1, multiclasse e binário. Usar `pytest.importorskip("xgboost")`.
- Notebook `notebooks/classification/14-xgboost.md` → `python tools/build_examples.py
  classification`.

Confirme o desenho com o Eduardo antes de implementar (sobretudo o mapeamento de params
`nrounds`/`params` → `XGBClassifier`, e se mantém `[xgboost]` como extra).

> **Depois do 2.1:** gráficos `plot_*` (~22, `grf_*.R`) → extra `[viz]` (`matplotlib`),
> seguindo a skill `dataviz`. Dividir em 2+ releases (2.3/2.4). Opcional/minor: conveniências
> só-R de patterns (`rules_tidy`, `dara_rules`, `is.redundant`). Depois: **1.0.0**
> (estabilização — API congelada, docs, beta→estável).

## Checklist de release 0.13.0

1. `pip install -e .[xgboost]` (local pode ter o build do PyPI em cache).
2. Implementar `ClaXgboost` + registrar nos `__init__` + extra no `pyproject.toml`.
3. Docstrings (Teoria + References + Porting, respeitando o ⛔).
4. `python -m pytest -q` (hoje 187 verdes) + `ruff check src tests` (limpo).
5. (R não muda neste passo — `cla_xgboost.R` já existe e fica como está.)
6. Bump de versão para `0.13.0` em `pyproject.toml` e `src/pydaltoolbox/__init__.py`.
7. Atualizar `README.md` e `develop/planejamento.md` (marcar 2.1 ✅).
8. `python tools/build_examples.py classification` (reverter churn cosmético se houver).
9. Commit em `main` (só pydaltoolbox muda), terminando com a linha `Co-Authored-By:` do ambiente.
10. Build/publicar: `rm -rf dist build *.egg-info src/*.egg-info` → `python -m build` →
    `python -m twine check dist/*` → conferir `develop/` fora do sdist →
    `python -m twine upload dist/*` (pode bloquear → Eduardo). Verificar em
    `https://pypi.org/pypi/pydaltoolbox/0.13.0/json`.
11. `git push origin main` + atualizar a memória `[[pydaltoolbox-port]]`.

## Observações

- `gh` CLI não está instalado; git via linha de comando.
- Shell primário é PowerShell; a ferramenta Bash também está disponível (sintaxe POSIX).
- Confirmar cada decisão de design com o Eduardo antes de implementar.
