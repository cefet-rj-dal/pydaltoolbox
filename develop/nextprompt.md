# Kickoff — pydaltoolbox, próximo passo 0.17.0 (documentação + testes pareados com o R)

Cole numa conversa limpa para continuar. Brief curto; o detalhe está nos arquivos abaixo.

## Leia primeiro
- `develop/historico.md` — o que já foi feito (0.1.0 → 0.16.0 + limpeza do GGally).
- `develop/planejamento.md` — **os próximos passos** (0.17.0, 0.18.0, 0.19.0 → 1.0.0) e as
  convenções permanentes (docstrings, deploy, R, gráficos, extensões de curso).
- `develop/mapeamento.md` — tabela R↔Python + as pendências de porte.
- Memória do projeto: `[[pydaltoolbox-port]]`, `[[back-implementation-initiative]]`,
  `[[docstring-theory-and-references]]`, `[[fix-bugs-in-both-r-and-python]]`.

## Estado
- Última release **0.16.0** (PyPI). Repos: `c:\Git\dal\pydaltoolbox` (Python),
  `c:\Git\dal\daltoolbox` (R, sem GGally), `C:\Git\eogasawara\datamining` (curso, `codes/`).
- 216 testes verdes, ruff limpo. Núcleo numpy/pandas/sklearn; extras `[patterns]`/`[xgboost]`/`[viz]`.

## Próximo = 0.17.0 (ver `planejamento.md`)
Documentação + **testes pareados R↔Python** (começar por classificação/regressão): mesmo
exemplo nos dois, comparar o resultado conceitual. Seguir o checklist de release do
`planejamento.md`. **1.0.0 só quando o Eduardo autorizar.**

> Ao concluir um passo: mover o relato para `historico.md`, atualizar `planejamento.md` e
> `mapeamento.md`, e a memória `[[pydaltoolbox-port]]`.
