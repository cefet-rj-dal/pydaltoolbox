# Planejamento — pydaltoolbox (próximos passos)

Somente o que vem a seguir. O que já foi feito está em `historico.md`; a correspondência
de classes R↔Python em `mapeamento.md`. Ao concluir um passo, mover o relato para
`historico.md` e atualizar este arquivo.

**Estado de partida:** última release **0.19.0** (limpezas: `pat_rules_tidy` portado p/ Python,
`pat_dara_rules` removido do R — **Pendências de porte zeradas**, ver `mapeamento.md`). Núcleo
numpy/pandas/scikit-learn; extras `[patterns]`, `[xgboost]`, `[viz]`. 218 testes verdes, ruff limpo.

**Rumo ao 1.0.0:** os passos abaixo são de **qualidade** — documentação, exemplos e paridade de
testes com o R. Vamos planejando e testando; **o 1.0.0 (estabilização/freeze) só acontece quando
o Eduardo disser que está satisfeito.**

---

## 0.20.0 — Documentação: ajustar a organização no daltoolbox
Objetivo: deixar a organização da documentação do pacote R coerente e enxuta, como base para
melhorar os exemplos depois.
- Revisar a estrutura de `examples/` e `Rmd/` do daltoolbox (numeração, agrupamento por tema,
  índices/README por pasta), corrigir referências quebradas e regenerar os artefatos via o
  `conversor.R` (reconstruir os `examples/**/*.md` e `r/*.R` que ficaram pendentes das limpezas).
- Conferir que todo `plot_*`/classe tem exemplo e man page coerentes após as remoções recentes.
- Fechar com o checklist de release padrão (lado Python sem mudança de código, só versão/docs se
  necessário) — ou tratar como passo só-R, conforme o caso.

## 0.21.0 — Documentação: melhoria dos exemplos
Objetivo: elevar a qualidade didática dos exemplos (R e notebooks Python).
- Melhorar os notebooks `notebooks/**` do pydaltoolbox e os `Rmd/**` do daltoolbox: contexto do
  problema, o que o gráfico/modelo responde, limitações, saída comentada.
- Garantir paridade de cobertura: todo componente com exemplo executável nos dois lados.
- Fechar com o checklist de release padrão.

## 0.22.0 — Documentação: refinamento da melhoria dos exemplos
Objetivo: segunda passada de refino sobre 0.21.0.
- Uniformizar estilo/linguagem, revisar textos, remover redundância, afinar figuras e tabelas.
- Fechar com o checklist de release padrão.

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
