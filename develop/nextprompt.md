# Kickoff — pydaltoolbox step 2.3 → release 0.14.0 (gráficos: distribuição + comparação, extra `[viz]`)

Cole este prompt numa conversa limpa para continuar o port. É auto-contido; o resto do
contexto está na memória e em `develop/planejamento.md`.

---

## Contexto (leia primeiro)

- Repo: `c:\Git\dal\pydaltoolbox` — port Python do pacote R **daltoolbox**
  (`c:\Git\dal\daltoolbox`), liderado por Eduardo Ogasawara (CEFET/RJ, lab DAL).
- Leia a memória do projeto: `[[pydaltoolbox-port]]`, `[[back-implementation-initiative]]`,
  `[[docstring-theory-and-references]]`, `[[fix-bugs-in-both-r-and-python]]`.
- Estado: **0.13.0 é a última release** (`ClaXgboost`, extra `[xgboost]`, step 2.1 ✅). O
  **Ciclo 1 está COMPLETO** e o **2.1 também**. Falta só o **Ciclo 2 de gráficos** (`plot_*`),
  dividido em **2.3** (este passo) e **2.4**. Confirme que
  `https://pypi.org/pypi/pydaltoolbox/0.13.0/json` está no ar — senão os artefatos ficam em
  `dist/` e o Eduardo publica (token em `C:\Users\eduar\.pypirc`, seção `[pypi]`).
- Plano completo: `develop/planejamento.md` (§8 = roadmap; seção "Decisões de desenho dos
  gráficos" no Ciclo 2 = o combinado abaixo).

## Decisões de desenho JÁ FECHADAS com o Eduardo (não reabrir)

- **Backend = `matplotlib`** (único dep novo, no extra opcional `[viz]`). NÃO usar plotnine.
  Os casos "difíceis" têm backend matplotlib já no **core**: `pandas.plotting.scatter_matrix`,
  `pandas.plotting.parallel_coordinates`, `scipy.cluster.hierarchy.dendrogram`, `imshow`
  (pixel/correlation), projeção `polar` (radar). `scipy`/`pandas` já são deps do core.
- **API = funções `plot_*` retornando o objeto do gráfico** (`matplotlib.figure.Figure` ou
  `Axes`), espelhando o R (`grf_*` devolve um ggplot que o usuário imprime). **Sem** classe
  `DalTransform`, **sem** estado, **sem** "base de gráficos" (o R `graphics_dal_graphics.R` é
  só agregador de `@include`). Utilitários comuns (aplicar paleta, tema) num `_helpers`
  interno do subpacote `viz/`.
- **Nomes:** manter os nomes R `plot_bar`, `plot_density`, ... (snake_case, são funções, não
  classes — exceção consciente ao PascalCase que vale para classes).
- Seguir a skill **`dataviz`** ANTES de escrever qualquer gráfico (paleta acessível, temas
  claro/escuro, coesão visual). Carregar a skill no começo.
- **Enxugamento:** `plot_pair_adv` será **removido** do daltoolbox e virá extensão do curso
  (`C:\Git\eogasawara\datamining`) — mas isso é do **2.4**, não deste passo.

## Princípios do port (valem sempre)

- **Envolver** matplotlib/pandas/scipy com adaptadores finos; reproduzir o comportamento
  DEFAULT do R (`[[pydaltoolbox-port]]`, quality-faithful). Comparar com o R quando fizer
  sentido (mesma figura conceitual; não precisa pixel-perfect).
- Docstrings OBRIGATÓRIAS (`[[docstring-theory-and-references]]`): Teoria + `References:` +
  `Porting:` (`Mirrors grf_<nome>.R. Fidelity: approximate|adapted`). ⛔ PROIBIDO "native
  implementation", "not a wrapper of X", alfinetadas no pacote substituído.
- **Bug em qualquer lado → corrigir nos dois** (`[[fix-bugs-in-both-r-and-python]]`), pedindo
  antes de editar o R. (Neste passo o R não muda — os `grf_*.R` já existem.)
- **Branches:** trabalhar em `main`; sem branches órfãs. Uploads externos (PyPI) podem ser
  bloqueados pelo auto-mode → deixar pronto e pedir pro Eduardo.

## Tarefa do 0.14.0 (step 2.3) — 11 gráficos de distribuição + comparação

Fontes R em `c:\Git\dal\daltoolbox\R\grf_*.R`. Criar subpacote `src/pydaltoolbox/viz/` com
`_helpers.py` (paleta/tema) e um módulo por tema, exportando as 11 funções. Registrar em
`viz/__init__.py` e no `__init__.py` principal (`__all__`).

| Função | Entrada (igual ao R) | Backend matplotlib |
|---|---|---|
| `plot_hist` | DataFrame/coluna numérica | `ax.hist` (color, alpha, bin) |
| `plot_density` | DataFrame 1+ col numéricas (sobrepõe se >1) | `scipy.stats.gaussian_kde` + `fill_between` |
| `plot_density_class` | DataFrame + `class_label` | kde por classe |
| `plot_boxplot` | DataFrame (colunas = caixas) | `ax.boxplot` |
| `plot_boxplot_class` | DataFrame + `class_label` | boxplot agrupado por classe |
| `plot_bar` | 2 col (categoria, valor) | `ax.bar` |
| `plot_groupedbar` | DataFrame (linhas=grupo, cols=séries) | `ax.bar` deslocado |
| `plot_stackedbar` | idem groupedbar | `ax.bar(bottom=)` |
| `plot_lollipop` | 2 col (categoria, valor) | `ax.vlines` + `scatter` (+ texto) |
| `plot_pieplot` | 2 col (rótulo, valor) | `ax.pie` |
| `plot_radar` | 2 col (nome, valor), ≥3 vars | projeção `polar` (polígono) |

Assinaturas R a espelhar (defaults inclusos):
- `plot_bar(data, label_x="", label_y="", colors=None, alpha=1)`
- `plot_boxplot(data, label_x="", label_y="", colors=None, barwidth=0.25)`
- `plot_boxplot_class(data, class_label, label_x="", label_y="", colors=None)`
- `plot_density(data, label_x="", label_y="", colors=None, bin=None, alpha=0.25)`
- `plot_density_class(data, class_label, label_x="", label_y="", colors=None, bin=None, alpha=0.5)`
- `plot_groupedbar(data, label_x="", label_y="", colors=None, alpha=1)`
- `plot_hist(data, label_x="", label_y="", color="white", alpha=0.25)`
- `plot_lollipop(data, label_x="", label_y="", colors=None, color_text="black", size_text=3, size_ball=8, alpha_ball=0.2, min_value=0, max_value_gap=1)`
- `plot_pieplot(data, label_x="", label_y="", colors=None, textcolor="white", bordercolor="black")`
- `plot_radar(data, label_x="", label_y="", colors=None)`
- `plot_stackedbar(data, label_x="", label_y="", colors=None, alpha=1)`

Notas de porte:
- `colors=None` → usar a paleta do `_helpers` (derivada da skill `dataviz`); se vier, usar as
  cores dadas. Honrar `alpha`, `label_x/label_y`, `bin`, `max_value_gap`, etc.
- Entrada = **pandas DataFrame** (o `data.frame`/`reshape::melt` do R vira operação pandas).
- Cada função cria sua própria `fig, ax = plt.subplots()` e retorna `fig` (ou `ax`; escolher um
  e ser consistente — recomendo `Axes` para compor, documentar a escolha).

## Checklist de release 0.14.0

1. `pip install -e .[viz]` (reinstalar editable; o local pode ter o build do PyPI em cache).
2. Criar `src/pydaltoolbox/viz/` (`_helpers.py` + módulos por tema) com as 11 funções.
3. Extra no `pyproject.toml`: `viz = ["matplotlib>=3.6"]`. Registrar nos `__init__` (`__all__`).
4. Docstrings (Teoria + References + Porting, respeitando o ⛔). Ref comum: Wickham (2016),
   ggplot2; e a referência específica de cada tipo quando couber.
5. `python -m pytest -q` (hoje 190 verdes; usar `Agg` backend nos testes — `matplotlib.use("Agg")`
   — e `pytest.importorskip("matplotlib")`; assertar tipo do retorno, nº de artistas, etc.).
   `ruff check src tests` (limpo).
6. (R não muda neste passo.)
7. Bump de versão para `0.14.0` em `pyproject.toml` e `src/pydaltoolbox/__init__.py`.
8. Atualizar `README.md` e `develop/planejamento.md` (marcar 2.3 ✅).
9. Notebooks `notebooks/graphics/*.md` (um por grupo, ou um por gráfico) →
   `python tools/build_examples.py graphics` (reverter churn cosmético se houver).
10. Commit em `main` (só pydaltoolbox muda), terminando com a linha `Co-Authored-By:` do ambiente.
11. Build/publicar: `rm -rf dist build *.egg-info src/*.egg-info` → `python -m build` →
    `python -m twine check dist/*` → conferir `develop/` fora do sdist →
    `python -m twine upload dist/*` (pode bloquear → Eduardo). Verificar em
    `https://pypi.org/pypi/pydaltoolbox/0.14.0/json`.
12. `git push origin main` + atualizar a memória `[[pydaltoolbox-port]]`.

## Observações

- `gh` CLI não está instalado; git via linha de comando.
- Shell primário é PowerShell; a ferramenta Bash também está disponível (sintaxe POSIX).
- R roda aqui (`C:\R\R-4.5.0`), de arquivo `.R` via `Rscript --vanilla` (não `-e` inline).
- Confirmar o desenho/visual com o Eduardo (paleta, retorno `Axes` vs `Figure`) antes de
  escalar para os 11.

> **Depois do 2.3:** step 2.4 (`0.15.0`) = relação + temporais + estrutura (10 `plot_*`:
> scatter/points/correlation/pair/parallel/pixel/series/ts/ts_pred/dendrogram). No 2.4 também
> **remover `plot_pair_adv` do daltoolbox (R)** + `devtools::document()` e portá-lo como
> extensão do curso em `C:\Git\eogasawara\datamining`. Depois: **1.0.0** (estabilização).
