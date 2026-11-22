# Kickoff — pydaltoolbox Ciclo 2 (planejar com o Eduardo) · próxima release `0.13.0`

Cole este prompt numa conversa limpa para continuar o port. É auto-contido; o resto do
contexto está na memória e em `develop/planejamento.md`.

---

## Contexto (leia primeiro)

- Repo: `c:\Git\dal\pydaltoolbox` — port Python do pacote R **daltoolbox**
  (`c:\Git\dal\daltoolbox`), liderado por Eduardo Ogasawara (CEFET/RJ, lab DAL).
- Leia a memória do projeto: `[[pydaltoolbox-port]]`, `[[back-implementation-initiative]]`,
  `[[docstring-theory-and-references]]`, `[[fix-bugs-in-both-r-and-python]]`.
- Estado: **0.12.0 é a última release** e **o Ciclo 1 está COMPLETO**. Confirme que
  `https://pypi.org/pypi/pydaltoolbox/0.12.0/json` está no ar — se o upload ao PyPI tiver
  ficado pendente, os artefatos estão em `dist/` e o Eduardo publica (token em
  `C:\Users\eduar\.pypirc`, seção `[pypi]`).
- Plano completo: `develop/planejamento.md` (§8 = roadmap; §2b/§2b.1 = back-implementation).

## O que já foi feito (não refazer)

- **Bloco A / v1** e **Cycle 1 (passos 1.1–1.9)** concluídos e publicados (0.1.0 → 0.12.0),
  incluindo o ciclo de consolidação (0.8.0: PAM/fss/Relief/SMOTE/caret/c-means).
- **Mineração de padrões completa:** `PatternMiner`, `PatUtils`, `PatApriori` (itemsets+regras,
  auto-thresholds via `SupportThreshold`/`ConfidenceThreshold`), rule filters
  (`RuleFilter`/`None`/`Interest`/`Dara`, todos `DalTransform`), `dara`/`dara_score`, e
  **`PatPrefixSpan`** (sequências, PrefixSpan nativo — validado idêntico ao cSPADE).
  `mlxtend` é o único extra opcional (`[patterns]`, só p/ apriori/regras); PrefixSpan é
  numpy/pandas puro.
- **Backports no R concluídos:** eclat removido; DARA com atribuição canônica; rule filters
  viraram `dal_transform`; `arulesSequences` removido (PrefixSpan nativo no R, `pat_cspade`
  fora). Os dois repos estão **uniformes** em patterns.

## Princípios do port (valem sempre)

- **Envolver** scikit-learn/numpy/pandas/scipy/mlxtend com a interface da Linha de
  Experimento (adaptadores finos). Implementação **nativa** só para dep de nicho ou ganho
  real (++), decidido **caso a caso com o Eduardo** (`[[back-implementation-initiative]]`).
- API idiomática Python: métodos de instância, `fit` retorna `self`; **PascalCase**; pandas.
- Núcleo = numpy/pandas/scikit-learn. Extras opcionais por tema (`[patterns]` já existe).
- Docstrings OBRIGATÓRIAS (`[[docstring-theory-and-references]]`): Teoria + `References:` +
  `Porting:`. ⛔ PROIBIDO "native implementation", "not a wrapper of X", "no dependency",
  "no longer uses X", ou alfinetadas no pacote substituído. Remoção de dep vai no COMMIT.

## Fluxo de trabalho

- **R roda aqui** (`C:\R\R-4.5.0`). Rodar de **arquivo `.R`** (`Rscript --vanilla script.R`),
  não `-e` inline (segfaults); repetir em segfault. `devtools::document()` regenera NAMESPACE/man.
- **Bug em qualquer lado → corrigir nos dois** (`[[fix-bugs-in-both-r-and-python]]`),
  pedindo antes de editar o R.
- **Branches:** trabalhar em `main` nos dois repos (ou branch + merge); manter só `main` sem
  branches órfãs. Uploads externos (PyPI, push de daltoolbox) podem ser bloqueados pelo
  auto-mode → deixar pronto e pedir pro Eduardo.

## Tarefa — CICLO 2: planejar com o Eduardo ANTES de implementar

O Ciclo 2 é a rodada de **classes de maior dependência** (extras opcionais) e estava marcado
como **"a refinar juntos"**. **Comece conversando com o Eduardo** sobre ordem e escopo.
Candidatos (planejamento §8, Ciclo 2):

- **`ClaXgboost`** → extra `[xgboost]` (`cla_xgboost.R`). Embrulho do `xgboost` (binário).
- **`ClusterLouvainGraph`** → extra `[graph]` (`clu_louvain.R`). `networkx` tem Louvain embutido;
  avaliar nativo vs networkx.
- **Gráficos** `plot_*` (~23, `grf_*.R`) → extra `[viz]` (`matplotlib`). Base + distribuição +
  comparação + relação + temporais + estrutura (dividir em 2+ releases).
- **Conveniências de patterns só-R** (minor, opcional): `rules_tidy`, `dara_rules`,
  `is.redundant`.
- Depois: **1.0.0** (estabilização — cobertura completa, API congelada, beta→estável).

Confirme **com o Eduardo** qual tema atacar primeiro, nativo-vs-extra em cada caso, e os nomes
das classes, antes de implementar.

## Checklist de release (padrão, ajuste a versão)

1. `pip install -e .` (local pode ter o build do PyPI em cache). 2. Implementar + registrar
   nos `__init__`. 3. Docstrings (Teoria+References+Porting, respeitando o ⛔). 4.
   `python -m pytest -q` (hoje 187 verdes) + `ruff check src tests`. 5. Backport/ajuste no R
   (se aplicável) via `Rscript` de arquivo + `devtools::document()`. 6. Bump de versão em
   `pyproject.toml` e `src/pydaltoolbox/__init__.py`. 7. `README.md` + `develop/planejamento.md`.
   8. `python tools/build_examples.py <tema>` (reverter churn cosmético de frozenset). 9. Commit
   em `main` nos dois repos, terminando com a linha `Co-Authored-By:` do ambiente. 10.
   `rm -rf dist build *.egg-info src/*.egg-info` → `python -m build` → `twine check dist/*` →
   `develop/` fora do sdist → `twine upload dist/*` (pode bloquear → Eduardo). 11. `git push
   origin main` nos dois repos + atualizar `[[pydaltoolbox-port]]`.

## Observações

- `gh` CLI não está instalado; git via linha de comando.
- Shell primário é PowerShell; a ferramenta Bash também está disponível (sintaxe POSIX).
- Confirmar cada decisão de design com o Eduardo antes de implementar.
