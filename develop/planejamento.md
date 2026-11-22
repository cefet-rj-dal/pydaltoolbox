# Planejamento — pydaltoolbox (próximos passos)

Somente o que vem a seguir. O que já foi feito está em `historico.md`; a correspondência
de classes R↔Python em `mapeamento.md`. Ao concluir um passo, mover o relato para
`historico.md` e atualizar este arquivo.

**Estado de partida:** última release pydaltoolbox **0.19.0**. Passos de documentação **0.20.0 e
0.21.0 feitos** (ver `historico.md`): daltoolbox `notebooks/`→`examples/` sem docx; e **exemplos
1:1 R↔Python** para ML+transf+patterns+custom (71 notebooks Python, um por classe, espelhando a
árvore do R) + dataset do R trocado p/ Diabetes. Núcleo numpy/pandas/scikit-learn; extras
`[patterns]`, `[xgboost]`, `[viz]`. 218 testes verdes, ruff limpo.

**Rumo ao 1.0.0:** os passos abaixo são de **qualidade**. Vamos planejando e testando; **o 1.0.0
(estabilização/freeze) só acontece quando o Eduardo disser que está satisfeito.** (Os passos de
exemplos/doc são dev-only — sem release PyPI, pois não mudam código do pacote.)

---

## 0.22.0 — Exemplos 1:1: gráficos + tutorial
Fecha a iniciativa de exemplos 1:1 (ver `historico.md` p/ as famílias já feitas).
- **graphics**: um notebook por `plot_*` (≈21), no agrupamento do R (`comparison/` + `distribution/`),
  espelhando `daltoolbox/notebooks/graphics/**` (inclui bar-with-error, export-save, etc.).
- **tutorial**: 13 tutoriais com a didática do R (`daltoolbox/notebooks/tutorial/**`): first-experiment,
  sampling, data-quality, preprocessing, baseline/metrics/comparison/tuning, end-to-end pipelines,
  regression/clustering workflows, visual analysis, custom extension.
- Regenerar `examples/` (Py `build_examples.py`) e conferir paridade final de cobertura.

## 0.23.0 — Testes pareados R↔Python
Objetivo: confiança de que o Python reproduz o R, caso a caso.
- Para cada família (começar por classificação e regressão), rodar o mesmo exemplo nos dois
  (`Rscript` + pytest) e comparar o resultado conceitual (predição/seleção/partição). Registrar
  em `tests/parity/` (Python) + scripts R em `develop/parity/`.
- Estender às demais famílias (transforms, clustering, padrões, gráficos por tipo de retorno).
- Fechar com o checklist de release padrão.

## 0.24.0 — Auditoria de documentação justa (crédito + inovação)
Objetivo: garantir documentação **justa com as obras originais** e que **apresenta a inovação**.
- Varrer todas as classes (Python + roxygen R): cada docstring com **Teoria + `References:` +
  `Porting:`**; referências ao **paper/autor original**; onde houve back-implementation com ganho,
  o `Porting:` descreve a **inovação (++)** com evidência.
- Conferir ausência de frases proibidas (ver convenções). Atribuição invertida onde o Python é
  canônico (DARA, PrefixSpan, PAM, Relief, …).
- Fechar com o checklist de release padrão.

## (passos seguintes, conforme necessário)
Continuar o ciclo de qualidade (mais paridade, mais exemplos, ajustes que o Eduardo pedir) em
releases 0.25.0, 0.26.0, … Cada um move seu relato para `historico.md`.

## 1.0.0 — Estabilização (somente quando o Eduardo autorizar)
- Congelar a API pública (revisar `__all__`, nomes, assinaturas; PascalCase classes /
  snake_case funções).
- `Development Status` de `3 - Alpha` → `4 - Beta`/`5 - Production/Stable` no `pyproject.toml`.
- Documentação final (README completo, talvez site de docs), cobertura e paridade consolidadas.
- Bump para `1.0.0` nos **dois** arquivos; build + twine check + publicar; verificar no PyPI.

---

## Convenções permanentes (valem em todo passo)
- **Docstrings obrigatórias:** Teoria + `References:` (paper/autor original) + `Porting:`
  (`Mirrors <arquivo>.R. Fidelity: exact|approximate|adapted`). Para back-implementation com
  ganho, citar o paper (não o pacote substituído) e descrever a inovação (++) com evidência.
  ⛔ **Proibido**: "native implementation", "not a wrapper of X", "no external dependency",
  alfinetadas — a remoção de dependência vai na **mensagem de commit**, não na docstring.
- **Bug em qualquer lado → corrigir nos dois** (pedir OK antes de editar o R).
- **Branches:** trabalhar em `main`; sem branches órfãs.
- **Deploy:** bump nos **dois** arquivos de versão (`pyproject.toml` **e**
  `src/pydaltoolbox/__init__.py`). `twine upload` e push ao `daltoolbox` (repo CRAN) podem ser
  bloqueados pelo auto-mode → se o Eduardo autorizar na sessão, tentar de novo; senão deixar
  pronto (`dist/`, commit local) e ele publica/empurra. Verificar sempre com o endpoint
  versionado `pypi.org/pypi/pydaltoolbox/<ver>/json` (o agregado fica em cache).
- **Ambiente R:** rodar de arquivo `.R` via `Rscript --vanilla` (não `-e` inline); segfaults
  intermitentes → reexecutar/isolar. `devtools::document()` regenera NAMESPACE/man.
- **Gráficos:** retorno `matplotlib.axes.Axes`; paleta CVD-safe de `viz/_helpers` quando
  `colors=None`; exports lazy PEP 562.
- **Extensões de curso:** o que depende de lib externa sem correspondência limpa (ex.: GGally)
  vai para `C:\Git\eogasawara\datamining` (`codes/`, R + Python), carregado por `source`/`exec`
  da URL publicada no GitHub.
