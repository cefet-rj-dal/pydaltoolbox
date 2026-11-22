# Planejamento — pydaltoolbox (próximos passos)

Somente o que vem a seguir. O que já foi feito está em `historico.md`; a correspondência
de classes R↔Python em `mapeamento.md`. Ao concluir um passo, mover o relato para
`historico.md` e atualizar este arquivo.

**Estado de partida:** última release **0.16.0** (PyPI). Núcleo numpy/pandas/scikit-learn;
extras `[patterns]`, `[xgboost]`, `[viz]`. Todo o daltoolbox portado exceto 3 conveniências
de regras (ver `mapeamento.md` §Pendências). 216 testes verdes, ruff limpo.

**Rumo ao 1.0.0:** os passos abaixo (0.17.0, 0.18.0, 0.19.0, …) são de **qualidade** —
documentação, paridade de testes com o R e porte das pendências. Vamos planejando e testando;
**o 1.0.0 (estabilização/freeze) só acontece quando o Eduardo disser que está satisfeito.**

---

## 0.17.0 — Documentação + testes pareados com o R (início)
Objetivo: elevar a confiança de que o Python reproduz o R, caso a caso.
- **Testes pareados R↔Python:** para cada família (começar por classificação e regressão, os
  mais usados a jusante), rodar o mesmo exemplo nos dois (`Rscript` + pytest) e comparar o
  resultado conceitual (predição/seleção/partição), não pixel/bit. Registrar as comparações
  (ex.: `tests/parity/` no Python + scripts R em `develop/parity/`).
- **Documentação:** revisar os notebooks `notebooks/**` e o `README.md`; garantir que cada
  componente tem exemplo executável e que o texto explica *o que faz* e *quando usar*.
- Fechar com: pytest + ruff verdes, notebooks construindo, bump nos **dois** arquivos de versão
  (`pyproject.toml` **e** `src/pydaltoolbox/__init__.py`), build + twine check, publicar, push.

## 0.18.0 — Auditoria de documentação justa (crédito + inovação)
Objetivo: garantir que a documentação é **justa com as obras originais** e **apresenta a
inovação** quando há (regra em `historico.md` / convenção abaixo).
- Varrer todas as classes: cada docstring tem **Teoria + `References:` + `Porting:`**; as
  referências citam o **paper/autor original** (não o pacote embrulhado), e onde houve
  back-implementation com ganho, o `Porting:` descreve a **inovação (++)** com evidência.
- Conferir que **não há** frases proibidas ("native implementation", "not a wrapper of X",
  alfinetadas no pacote substituído) — crédito à evolução ("inspired by X"), nunca dig.
- Alinhar a mesma auditoria no roxygen do R (atribuição invertida onde o Python é canônico:
  DARA, PrefixSpan, PAM, Relief, etc.).
- Fechar com o checklist de release padrão.

## 0.19.0 — Portar pendências + fechar cobertura de testes
Objetivo: zerar as lacunas do `mapeamento.md` e completar a paridade de testes.
- Portar as conveniências de regras ainda só-R (decidir caso a caso com o Eduardo):
  `pat_rules_tidy` (regras legíveis em DataFrame), `pat_dara_rules` (minera + ranqueia DARA
  numa chamada), e avaliar remoção de redundância de regras (equivalente a `arules::is.redundant`).
- Estender os testes pareados às demais famílias (transforms, clustering, padrões, gráficos —
  estes por tipo de retorno/artistas).
- Fechar com o checklist de release padrão.

## (passos seguintes, conforme necessário)
Continuar o ciclo de qualidade (mais paridade, mais exemplos, ajustes que o Eduardo pedir) em
releases 0.20.0, 0.21.0, … Cada um move seu relato para `historico.md`.

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
- **Deploy:** bump nos **dois** arquivos de versão. `twine upload` e push ao `daltoolbox`
  (repo CRAN) podem ser bloqueados pelo auto-mode → se o Eduardo autorizar na sessão, tentar
  de novo; senão deixar pronto (`dist/`, commit local) e ele publica/empurra. Verificar sempre
  com o endpoint versionado `pypi.org/pypi/pydaltoolbox/<ver>/json` (o agregado fica em cache).
- **Ambiente R:** rodar de arquivo `.R` via `Rscript --vanilla` (não `-e` inline); segfaults
  intermitentes → reexecutar/isolar. `devtools::document()` regenera NAMESPACE/man.
- **Gráficos:** retorno `matplotlib.axes.Axes`; paleta CVD-safe de `viz/_helpers` quando
  `colors=None`; exports lazy PEP 562.
- **Extensões de curso:** o que depende de lib externa sem correspondência limpa (ex.: GGally)
  vai para `C:\Git\eogasawara\datamining` (`codes/`, R + Python), carregado por `source`/`exec`
  da URL publicada no GitHub.
