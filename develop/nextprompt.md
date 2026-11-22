# Kickoff — pydaltoolbox step 1.8b → release 0.11.0 (mineração de padrões: RuleFilterInterest + thresholds)

Cole este prompt numa conversa limpa para continuar o port. É auto-contido; o resto do
contexto está na memória e em `develop/planejamento.md`.

---

## Contexto (leia primeiro)

- Repo: `c:\Git\dal\pydaltoolbox` — port Python do pacote R **daltoolbox**
  (`c:\Git\dal\daltoolbox`), liderado por Eduardo Ogasawara (CEFET/RJ, lab DAL).
- Leia a memória do projeto: `[[pydaltoolbox-port]]`, `[[back-implementation-initiative]]`,
  `[[docstring-theory-and-references]]`, `[[fix-bugs-in-both-r-and-python]]`.
- Estado: **0.10.0 é a última release** (step 1.8a, rule filters base + DARA).
  Confirme que `https://pypi.org/pypi/pydaltoolbox/0.10.0/json` está no ar — se o upload ao
  PyPI tiver ficado pendente (o classificador de auto-mode bloqueia `twine upload`), os
  artefatos já estão em `dist/` e o Eduardo publica (token em `C:\Users\eduar\.pypirc`,
  seção `[pypi]`).
- Plano completo: `develop/planejamento.md` (§8 = roadmap; §2b/§2b.1 = back-implementation).

## O que já foi feito (não refazer)

- **Bloco A / v1** e **Cycle 1 passos 1.1–1.6** concluídos e publicados (0.1.0 → 0.7.0).
- **Ciclo de consolidação (0.8.0)** concluído nos dois repos (PAM/fss/Relief/SMOTE/caret/c-means).
- **1.7 (0.9.0)** concluído: subpacote `src/pydaltoolbox/pattern/` com `PatternMiner` base +
  `PatUtils` (filtros `quality_min`/`quality_max` + métricas) + `PatApriori`.
  - **Backend = `mlxtend`** como **extra opcional `[patterns]`** (embrulho fino, import
    lazy), NÃO nativo — decisão do Eduardo (antecipou o antigo passo 2.5).
  - `PatApriori` faz **itemsets E regras** (`target='frequent itemsets'` → `mlxtend.apriori`;
    `target='rules'` → `association_rules`, com confidence/lift, lhs/rhs, include/exclude,
    quality_filter). Entrada = **DataFrame one-hot booleano** (colunas = itens).
  - **ECLAT removido dos dois repos**: `arules` e `mlxtend` só compartilham `apriori` (arules
    não tem fpgrowth sem o nicho não-CRAN `fim4r`; mlxtend não tem eclat). `pat_eclat.R` +
    man/examples saíram do R; `devtools::document()` rodado.
- **1.8a (0.10.0)** concluído: base de rule filters + DARA.
  - `pattern/rule_filter.py`: `RuleFilter` base + `RuleFilterNone` (no-op) + `RuleFilterDara`.
  - `pattern/dara.py`: `dara()` (ranking de divergência, Algoritmo 2 do paper) + `dara_score()`
    + `parse_items()`. **Back-implementation canônica** (algoritmo do grupo, Baroni et al.
    2020 JBI 108:103512): Python canônico, R backport, ambos citam o paper.
  - `PatApriori` ganhou `rule_filter=` (default `RuleFilterNone`; só aplica em `target='rules'`,
    pula itemsets). DARA opera sobre itens `atributo=valor` (one-hot `pd.get_dummies(prefix_sep='=')`).
  - Validado: counter == Tabela 1 do paper e == `pat_dara` do R.

## Princípios do port (valem sempre)

- **Envolver** scikit-learn/numpy/pandas/scipy/mlxtend com a interface da Linha de
  Experimento (adaptadores finos). Implementação **nativa** só para dep de nicho ou ganho
  real (++), decidido **caso a caso com o Eduardo** (`[[back-implementation-initiative]]`).
- API idiomática Python: métodos de instância, `fit` retorna `self`; **PascalCase**; pandas.
- Core deps = numpy/pandas/scikit-learn. `mlxtend` é o primeiro extra opcional (`[patterns]`).

## Convenções de docstring (OBRIGATÓRIAS — `[[docstring-theory-and-references]]`)

Toda classe nova: **(1) Teoria**; **(2) `References:`** (citações acadêmicas verbatim);
**(3) `Porting:`** auditável (`Mirrors <arquivo_r>.R. Fidelity: exact|approximate|adapted`
+ bullets das adaptações). **⛔ PROIBIDO** no docstring: "native implementation", "not a
wrapper of X", "no external/extra dependency", "no longer uses/depends on X", ou qualquer
alfinetada no pacote substituído. Dizer só: o que o algoritmo faz + "inspired by X" +
References. Remoção de dependência vai na MENSAGEM DE COMMIT, não no docstring.

## Fluxo de trabalho (importante)

- **R roda neste ambiente** (`C:\R\R-4.5.0`). Rodar R de **arquivo `.R`**
  (`Rscript --vanilla script.R`), não `-e` inline (segfaults); repetir em segfault.
  `devtools::document()` regenera NAMESPACE/man.
- **Bug em qualquer lado → corrigir nos dois** (`[[fix-bugs-in-both-r-and-python]]`),
  pedindo antes de editar o R.
- **Uploads externos (PyPI, push de daltoolbox) podem ser bloqueados pelo auto-mode** —
  deixar pronto e pedir pro Eduardo.

## Tarefa do 0.11.0 (step 1.8b) — RuleFilterInterest + thresholds

Fontes R (`c:\Git\dal\daltoolbox\R\`): `pat_rule_filter.R`, `pat_threshold.R`.

- **`RuleFilterInterest`** (`pat_rule_filter.R`, `pat_rule_filter_interest`): filtra regras
  por medidas clássicas de interesse — lift, Kulczynski (`kulc`), imbalance ratio (`ir`),
  support, confidence, count. Herdar de `RuleFilter` (base já existe) e encaixar no
  `PatApriori(rule_filter=...)`. Precisa calcular kulc/ir/count a partir das regras do
  mlxtend (o `association_rules` já traz lift; kulc = (conf(A→B)+conf(B→A))/2; ir e count
  derivam dos suportes/antecedent-consequent supports — ver `pat_rules_tidy`/`interestMeasure`).
- **Thresholds** (`pat_threshold.R`): `pat_support_threshold` (estratégia "curvature", usando
  `FitCurvatureMax` já portado) e `pat_confidence_threshold` — habilitam a auto-estimação
  `supp=0`/`conf=0` no `fit()` do `PatApriori` (hoje exige `supp>0`).
- Registrar nos `__init__`. Testes pytest. Notebook `notebooks/pattern/03-rule-filters.md` →
  `python tools/build_examples.py pattern`.

> **Já pronto (não refazer):** base `RuleFilter`/`RuleFilterNone`/`RuleFilterDara`, `dara()`,
> `dara_score()`, e o slot `PatApriori(rule_filter=...)` (só `target='rules'`, pula itemsets).

Confirme o desenho com o Eduardo antes de implementar (sobretudo a forma das estratégias de
threshold e se `RuleFilterInterest` é embrulho de medidas do mlxtend ou cálculo próprio).

## Checklist de release 0.11.0

1. `pip install -e .[patterns]` (o Python local pode ter o build do PyPI em cache).
2. Implementar as classes + registrar nos `__init__`.
3. Docstrings com Teoria + References + Porting (respeitando o ⛔ PROIBIDO acima).
4. `python -m pytest -q` (hoje 155 verdes) e `ruff check src tests` (limpo).
5. Backport/ajuste no R (branch), testar via `Rscript` de arquivo, `devtools::document()`.
6. Bump de versão para `0.11.0` em `pyproject.toml` e `src/pydaltoolbox/__init__.py`.
7. Atualizar `README.md` e `develop/planejamento.md` (marcar 1.8b ✅).
8. `python tools/build_examples.py pattern` (confirmar render). ⚠️ `pip install mlxtend`
   pode subir numpy/sklearn — se re-renderizar outros exemplos por drift de versão, reverter
   os não relacionados e commitar só os de padrões.
9. Commit em `main` nos dois repos ("Release 0.11.0 — ..."), terminando com a linha
   `Co-Authored-By:` que o ambiente indicar.
10. Build/publicar: `rm -rf dist build *.egg-info src/*.egg-info` → `python -m build` →
    `python -m twine check dist/*` → conferir `develop/` fora do sdist →
    `python -m twine upload dist/*` (pode ser bloqueado → Eduardo publica). Verificar em
    `https://pypi.org/pypi/pydaltoolbox/0.11.0/json`.
11. `git push origin main` nos dois repos. Atualizar a memória `[[pydaltoolbox-port]]`.

## Observações

- `gh` CLI não está instalado; git via linha de comando.
- Shell primário é PowerShell; a ferramenta Bash também está disponível (sintaxe POSIX).
- Confirmar cada decisão de design com o Eduardo antes de implementar.
