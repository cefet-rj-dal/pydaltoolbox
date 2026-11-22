# Kickoff — pydaltoolbox step 2.4 → release 0.15.0 (gráficos: relação + temporais + estrutura, extra `[viz]`)

Cole este prompt numa conversa limpa para continuar o port. É auto-contido; o resto do
contexto está na memória e em `develop/planejamento.md`.

---

## Contexto (leia primeiro)

- Repo: `c:\Git\dal\pydaltoolbox` — port Python do pacote R **daltoolbox**
  (`c:\Git\dal\daltoolbox`), liderado por Eduardo Ogasawara (CEFET/RJ, lab DAL).
- Leia a memória do projeto: `[[pydaltoolbox-port]]`, `[[back-implementation-initiative]]`,
  `[[docstring-theory-and-references]]`, `[[fix-bugs-in-both-r-and-python]]`.
- Estado: **0.14.0 é a última release** (step 2.3 ✅: 11 `plot_*` de distribuição +
  comparação, extra `[viz]` = matplotlib). O **Ciclo 1 está COMPLETO**, **2.1 e 2.3 também**.
  Falta só o **2.4** (este passo) para fechar o Ciclo 2 de gráficos. Confirme que
  `https://pypi.org/pypi/pydaltoolbox/0.14.0/json` está no ar — senão os artefatos ficam em
  `dist/` e o Eduardo publica (o upload do 0.14.0 foi **bloqueado pelo auto-mode**; token em
  `C:\Users\eduar\.pypirc`, seção `[pypi]`).
- Plano completo: `develop/planejamento.md` (§8 = roadmap; seção "Decisões de desenho dos
  gráficos" no Ciclo 2 + o detalhe "#### 2.4").

## Decisões de desenho JÁ FECHADAS com o Eduardo (não reabrir — valem do 2.3)

- **Backend = `matplotlib`** (extra opcional `[viz]`, já criado em `pyproject.toml`).
- **API = funções `plot_*` retornando um `matplotlib.axes.Axes`** (não Figure; a figura é
  `ax.figure`). Sem classe/estado. Utilitários comuns já em `src/pydaltoolbox/viz/_helpers.py`
  (paleta CVD-safe de 8 hues quando `colors=None`, tema `theme_bw`-like claro, `new_axes`,
  `resolve_colors`, `apply_theme`). **Reaproveitar `_helpers` — não duplicar.**
- Nomes R preservados (snake_case, são funções). Exports **lazy (PEP 562)** no `__init__`
  principal (bloco `_VIZ_EXPORTS` + `__getattr__`) para não exigir matplotlib no core —
  **adicione os 10 novos nomes lá E em `viz/__init__.py` E nos dois `__all__`**.
- Seguir a skill **`dataviz`** ANTES de escrever qualquer gráfico. Carregar no começo.

## Princípios do port (valem sempre)

- **Envolver** matplotlib/pandas/scipy com adaptadores finos; reproduzir o DEFAULT do R
  (quality-faithful). Não precisa pixel-perfect — mesma figura conceitual.
- Docstrings OBRIGATÓRIAS (`[[docstring-theory-and-references]]`): Teoria + `References:` +
  `Porting:` (`Mirrors grf_<nome>.R. Fidelity: approximate|adapted`). ⛔ PROIBIDO "native
  implementation", "not a wrapper of X", alfinetadas no pacote substituído.
- **Bug em qualquer lado → corrigir nos dois** (`[[fix-bugs-in-both-r-and-python]]`), pedindo
  antes de editar o R. **Neste passo O R MUDA** (remoção do `plot_pair_adv` — ver abaixo).
- **Branches:** trabalhar em `main`; sem branches órfãs. Upload PyPI é bloqueado pelo
  auto-mode → deixar `dist/` pronto e pedir pro Eduardo. Push ao `daltoolbox origin/main`
  também costuma ser bloqueado (pacote CRAN) → mergear local e pedir pro Eduardo empurrar.

## Tarefa do 0.15.0 (step 2.4) — 10 gráficos de relação + temporais + estrutura

Fontes R em `c:\Git\dal\daltoolbox\R\grf_*.R`. Adicionar ao subpacote `src/pydaltoolbox/viz/`
(ex.: módulos `relation.py`, `temporal.py`, `structure.py`), exportando as 10 funções.

| Função | Tipo | Arquivo R | Backend matplotlib |
|---|---|---|---|
| `plot_scatter` | relação | `grf_scatter.R` | `ax.scatter` (cor por classe opcional) |
| `plot_points` | relação | `grf_points.R` | `ax.plot`/`scatter` |
| `plot_correlation` | relação | `grf_correlation.R` | `imshow` + anotação; `reorder="hclust"` via scipy linkage; `scale_fill_gradient2` → diverging azul↔vermelho, midpoint 0, limites [-1,1] |
| `plot_pair` | relação | `grf_pair.R` | `pandas.plotting.scatter_matrix` (R usa GGally::ggpairs, mas é matriz simples) |
| `plot_parallel` | relação | `grf_parallel.R` | `pandas.plotting.parallel_coordinates` |
| `plot_pixel` | relação | `grf_pixel.R` | `imshow`/`pcolormesh` |
| `plot_series` | temporal | `grf_series.R` | `ax.plot` multi-série |
| `plot_ts` | temporal | `grf_ts.R` | `ax.plot` |
| `plot_ts_pred` | temporal | `grf_ts_pred.R` | `ax.plot` sobreposto (real vs previsto) |
| `plot_dendrogram` | estrutura | `grf_dendrogram.R` | `scipy.cluster.hierarchy.dendrogram` |

Notas de porte:
- Ler cada `grf_*.R` e espelhar assinatura + defaults (igual fiz no 2.3). `colors=None` →
  `resolve_colors` do `_helpers`.
- `plot_correlation` tem muitos params (method/triangle/reorder/digits/show_diag/title) —
  portar fielmente; é o gráfico sequencial/diverging, cuidar da paleta conforme a skill.
- Entrada = **pandas DataFrame**.

## ⚠️ Enxugamento OBRIGATÓRIO deste passo: remover `plot_pair_adv` (R) + portar p/ o curso

- `plot_pair_adv` (`grf_pair_adv.R`, `GGally::ggpairs` com painéis mistos densidade+correlação
  +cor-por-classe) **NÃO tem correspondência limpa no matplotlib** → **sai do daltoolbox** e
  vira extensão do curso. Mesmo padrão de Louvain/eclat/arulesViz.
- No **R** (pedir ao Eduardo antes de editar): remover `R/grf_pair_adv.R` + `man/` + exemplos,
  rodar `devtools::document()` (NAMESPACE/man limpos), testar `devtools::load_all(".")`. Merge
  local, pedir push ao Eduardo (origin/main bloqueado p/ CRAN).
- Portar `plot_pair_adv` para `C:\Git\eogasawara\datamining` (tutorial `03-DataVisualization`)
  como extensão de curso. Confirmar o escopo disso com o Eduardo.

## Checklist de release 0.15.0

1. `pip install -e .[viz]` (reinstalar editable; o local pode ter o build do PyPI em cache).
2. Adicionar os 10 `plot_*` a `src/pydaltoolbox/viz/` (reusando `_helpers`). Registrar em
   `viz/__init__.py`, no `_VIZ_EXPORTS`/`__getattr__` e nos dois `__all__`.
3. Docstrings (Teoria + References + Porting, respeitando o ⛔). Ref comum: Wickham (2016).
4. `python -m pytest -q` (hoje **202 verdes**; backend `Agg` + `pytest.importorskip("matplotlib")`,
   assertar tipo `Axes` + nº de artistas). `ruff check src tests` (limpo).
5. **R muda:** remover `plot_pair_adv` + `devtools::document()` (pedir ao Eduardo). Portar p/
   o `datamining`.
6. Bump de versão para `0.15.0` em `pyproject.toml` e `src/pydaltoolbox/__init__.py`.
7. Atualizar `README.md` (listar os 10 novos) e `develop/planejamento.md` (marcar 2.4 ✅).
8. Notebooks `notebooks/graphics/03-relation.md` + `04-temporal.md` (ou por gráfico) →
   `python tools/build_examples.py graphics`. **Não** ponha `matplotlib.use("Agg")` no
   notebook (suprime a captura PNG inline — aprendizado do 2.3); o build usa ipykernel inline.
9. Commit em `main`, terminando com a linha `Co-Authored-By:` do ambiente.
10. Build/publicar: `rm -rf dist build *.egg-info src/*.egg-info` → `python -m build` →
    `python -m twine check dist/*` → conferir `viz/` no wheel e `develop/` fora do sdist →
    `python -m twine upload dist/*` (vai bloquear → Eduardo). Verificar em
    `https://pypi.org/pypi/pydaltoolbox/0.15.0/json`.
11. `git push origin main` + atualizar a memória `[[pydaltoolbox-port]]`.

## Observações

- `gh` CLI não está instalado; git via linha de comando.
- Shell primário é PowerShell; a ferramenta Bash também está disponível (sintaxe POSIX).
- R roda aqui (`C:\R\R-4.5.0`), de arquivo `.R` via `Rscript --vanilla` (não `-e` inline);
  segfaults intermitentes em cargas pesadas → reexecutar/isolar.

> **Depois do 2.4:** **Ciclo 2 COMPLETO** (21 `plot_*` + `ClaXgboost`). Próximo e último
> marco: **1.0.0** (estabilização — cobertura completa, API congelada, beta→estável).
