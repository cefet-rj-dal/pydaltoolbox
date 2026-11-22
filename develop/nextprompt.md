# Kickoff — fechamento do Ciclo 2: ajustes no daltoolbox (R) + rumo ao 1.0.0

Cole este prompt numa conversa limpa para continuar. É auto-contido; o resto do contexto
está na memória e em `develop/planejamento.md`.

---

## Contexto (leia primeiro)

- Repos: `c:\Git\dal\pydaltoolbox` (port Python) e `c:\Git\dal\daltoolbox` (original R),
  liderados por Eduardo Ogasawara (CEFET/RJ, lab DAL). Curso: `C:\Git\eogasawara\datamining`.
- Leia a memória: `[[pydaltoolbox-port]]`, `[[back-implementation-initiative]]`,
  `[[docstring-theory-and-references]]`, `[[fix-bugs-in-both-r-and-python]]`.
- Estado: **0.15.0 é a última release** — a **família de gráficos está COMPLETA no Python**
  (21 `plot_*`, extra `[viz]`). Ciclo 1 e Ciclo 2 (lado Python) concluídos. Confirme
  `https://pypi.org/pypi/pydaltoolbox/0.15.0/json` no ar.

## Tarefa 1 (combinada com o Eduardo) — ajustes no daltoolbox (R)

Metade pendente do step 2.4: **remover `plot_pair_adv` do daltoolbox R** e **portá-lo para o
curso**. (`plot_pair_adv` = `GGally::ggpairs` com painéis mistos densidade+correlação+cor-por
-classe; sem correspondência limpa no matplotlib → não entra no port, vira extensão de curso,
mesmo padrão de Louvain/eclat/arulesViz.)

Passos (⚠️ **R muda → pedir OK ao Eduardo antes de editar**, `[[fix-bugs-in-both-r-and-python]]`):
1. No `c:\Git\dal\daltoolbox`: remover `R/grf_pair_adv.R`, o man page (`man/plot_pair_adv.Rd`),
   e qualquer exemplo/Rmd que use `plot_pair_adv`. Conferir se `GGally` fica órfão como dep
   (ainda é usado por `grf_pair.R`/`grf_parallel.R`? → `plot_pair` usa `GGally::ggpairs`,
   `plot_parallel` usa `GGally::ggparcoord`; então **GGally continua Suggests, NÃO remover**).
2. `C:\R\R-4.5.0\bin\Rscript --vanilla` de arquivo `.R` (não `-e` inline): rodar
   `devtools::document()` (regenera NAMESPACE/man) e `devtools::load_all(".")` p/ validar.
   Segfaults intermitentes em carga pesada → reexecutar/isolar.
3. Portar `plot_pair_adv` para `C:\Git\eogasawara\datamining` (tutorial `03-DataVisualization`)
   como extensão de curso. **Confirmar o formato/escopo disso com o Eduardo** (pode ser função
   R no material do curso, ou célula de notebook). Ler como o curso organiza os tutoriais antes.
4. Commit em cada repo. Push ao `daltoolbox origin/main` costuma ser **bloqueado pelo
   auto-mode** (pacote CRAN sensível) → mergear local e pedir ao Eduardo empurrar.
5. Atualizar `develop/planejamento.md` (tirar o "PENDENTE" do 2.4) e a memória
   `[[pydaltoolbox-port]]`.

> **Auditoria oportuna:** aproveitar p/ conferir se há outros `grf_*` no R sem par no Python
> que deveriam sair (ou entrar). O Python tem 21 `plot_*`; o R tinha ~23 `grf_*` (um deles era
> `plot_pair_adv`; o outro delta provável é algum agregador/`graphics_dal_graphics.R`).

## Tarefa 2 — rumo ao 1.0.0 (estabilização)

Depois do fechamento acima, o próximo e último marco é **1.0.0**:
- **API freeze:** revisar os símbolos públicos (`__all__`), nomes, assinaturas. Hoje ~85
  símbolos + 21 `plot_*`. Garantir consistência (PascalCase classes, snake_case funções).
- **Cobertura:** rodar `python -m pytest -q` (hoje **218 verdes**), `ruff check src tests`,
  e `python tools/build_examples.py` (todos os notebooks) — reverter churn cosmético (PIDs de
  ipykernel, bytes de PNG) se aparecer.
- **Docs:** README completo; conferir que todo componente tem docstring Theory+References+
  Porting (`[[docstring-theory-and-references]]`).
- **Classifier:** mudar `Development Status :: 3 - Alpha` → `5 - Production/Stable` (ou
  `4 - Beta`) no `pyproject.toml`, conforme o Eduardo decidir.
- Bump p/ `1.0.0` em `pyproject.toml` E `src/pydaltoolbox/__init__.py` (os dois! lição do
  0.15.0: esqueci o pyproject e o build saiu 0.14.0). Build + twine check + upload (vai pedir
  confirmação; o user autoriza). Verificar `https://pypi.org/pypi/pydaltoolbox/1.0.0/json`.
- Confirmar o escopo do 1.0.0 COM o Eduardo (o que congela, o que fica fora — autoencoders
  reais seguem fora, em pacote companheiro).

## Observações

- `gh` CLI não instalado; git via linha de comando. Shell primário PowerShell; Bash também.
- **Upload PyPI:** o `twine upload` foi bloqueado 1x pelo auto-mode mas passou quando o user
  re-pediu explicitamente na sessão — se bloquear, tentar de novo após o OK do user.
- Bump de versão: **mexer nos DOIS arquivos** (`pyproject.toml` + `__init__.py`).
- Convenções de gráficos (fechadas com Eduardo): retorno `Axes` (exceto `plot_pair` → grade
  ndarray de Axes), paleta CVD-safe de `viz/_helpers.py` quando `colors=None`, exports lazy
  PEP 562, docstrings com Porting Fidelity approximate.
