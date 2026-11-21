# Kickoff — pydaltoolbox step 1.6 → release 0.7.0 (feature selection)

Cole este prompt numa conversa limpa para continuar o port. Ele é auto-contido; o
resto do contexto está na memória e no `develop/planejamento.md`.

---

## Contexto (leia primeiro)

- Repo: `c:\Git\dal\pydaltoolbox` — port Python do pacote R **daltoolbox**
  (`c:\Git\dal\daltoolbox`), liderado por Eduardo Ogasawara (CEFET/RJ, lab DAL).
- Leia a memória do projeto: `[[pydaltoolbox-port]]`, `[[docstring-theory-and-references]]`,
  `[[fix-bugs-in-both-r-and-python]]` (bug no port OU no R deve ser corrigido nos dois —
  **peça antes de editar o R**).
- Estado: **0.6.0 publicado no PyPI** (2026-11-21) e em `origin/main`. Árvore limpa.
  Cycle 1 do roadmap: passos 1.1–1.5 concluídos. **Este é o passo 1.6.**
- Plano completo: `develop/planejamento.md` (§8 = roadmap; §1.6 = este passo).

## Princípios do port (valem sempre)

- **Não reimplementar algoritmos clássicos** — envolver scikit-learn/numpy/pandas/scipy
  com a interface uniforme da Linha de Experimento (adaptadores finos). Implementação
  **nativa** só quando evita uma dependência niche (é o caso do ReliefF).
- **Mapeamento fiel de qualidade**: reproduzir o comportamento/DEFAULTS do backend R,
  não só chamar sklearn ingenuamente. Comparar resultado vs R.
- API idiomática Python: métodos de instância, `fit` retorna `self` (encadeável);
  nomes em **PascalCase**; dados em pandas DataFrame.
- Dependências centrais ficam em numpy/pandas/scikit-learn (sem novas no Cycle 1).

## Princípio estendido — ++ & redução de dependência (2026-11-21)

Eduardo estendeu a iniciativa de back-implementation (ver memória
`[[back-implementation-initiative]]`) para **todo o stack DAL, incluindo este
pacote**. Vale a partir de agora, em paralelo ao "não reimplementar — envolver":

1. **Incrementar métodos para "++"** quando fizer sentido: avanço algorítmico real,
   gate da **Ação 1++** = prever/selecionar/agrupar a verdade **tão bem ou melhor**
   que a referência (não mera paridade). **Não é regra cega** — decidido caso a caso
   com o Eduardo; libs canônicas/estáveis (scikit-learn/scipy) continuam embrulhadas.
2. **Reduzir a pegada de dependências** back-implementando deps **atípicas/niche**
   nativamente (Python = releitura canônica / R = backport 100%-compatível).
   pydaltoolbox é a **fundação** — lar natural dos algoritmos-base nativos que
   pyharbinger e pytspredit reusam.

**TODO antes de escolher o próximo marco** (não é o 0.7.0, que já está decidido):
fazer a **auditoria de imports** deste pacote e montar a **lista priorizada de
candidatos ++ / redução de dep** numa seção do `develop/planejamento.md`. O 0.7.0
(ReliefF nativo + avaliar stepwise sem statsmodels) **já encarna o princípio** — não
precisa mudar de escopo.

## Tarefa do 0.7.0 (step 1.6)

Portar seleção de atributos restante e **reconciliar `fss` × `stepwise`**. Arquivos R:
`trans_feature_selection_relief.R`, `trans_feature_selection_fss.R`,
`trans_feature_selection_stepwise.R` (em `c:\Git\dal\daltoolbox\R\`).

Situação atual no Python (`src/pydaltoolbox/transform/feature_selection.py`): existe
`FeatureSelectionStepwise`, mas ele **na verdade implementa o `fss`** (busca forward
mantendo o maior R² ajustado — o análogo de `leaps::regsubsets`). O nome está trocado.

Entregar três classes distintas, alinhadas ao R:

1. **`FeatureSelectionFss`** — renomear/mover o atual `FeatureSelectionStepwise` para cá
   (busca forward por R² ajustado; alvo numérico / regressão). Fonte R:
   `trans_feature_selection_fss.R`.
   - Manter `FeatureSelectionStepwise` como alias **temporário deprecado** apontando para
     a nova classe `stepwise`? NÃO — ver item 2. Decidir com cuidado para não quebrar
     `__init__`/README/notebooks que hoje exportam `FeatureSelectionStepwise`.
2. **`FeatureSelectionStepwise`** (novo, comportamento correto) — mirror de
   `stats::step`: direção forward/backward/both, família GLM (default binomial →
   classificação binária). Backend: statsmodels GLM + stepwise por AIC, OU sklearn +
   seleção por AIC/validação. **Avaliar se dá para fazer nativo sem nova dependência**
   (preferível); se exigir statsmodels, tratar como extra opcional e confirmar com Eduardo.
3. **`FeatureSelectionRelief`** — ReliefF **nativo** em numpy (para cada instância
   amostrada, compara vizinhos nearest-hit/nearest-miss e atualiza pesos dos atributos;
   params do R: `attribute`, `features`, `top`, `cutoff`, `m=50`). Fonte R:
   `trans_feature_selection_relief.R`. Sem dependência nova (não usar skrebate).

Confirme o desenho das 3 classes com Eduardo antes de implementar (especialmente o
destino do nome `FeatureSelectionStepwise` e se `stepwise` pode ser nativo).

## Convenções de docstring (OBRIGATÓRIas — ver [[docstring-theory-and-references]])

Toda classe nova precisa, no docstring:
1. **Teoria**: como o algoritmo funciona + base teórica (o `@details` do R).
2. **`References:`**: citações acadêmicas (o `@references` do R; copiar verbatim; se o R
   não tiver, usar a referência canônica do algoritmo — ex.: Kira & Rendell 1992 /
   Kononenko 1994 para Relief/ReliefF).
3. **`Porting:`** auditável, formato exato:
   ```
   Porting:
       Mirrors ``<arquivo_r>.R``. Fidelity: exact|approximate|adapted.
       - <cada adaptação/ajuste deliberado e o porquê>   # só quando approximate/adapted
   ```
   **Só o nome do arquivo R** — sem versão, sem "daltoolbox". Fidelity: `exact` (igual),
   `approximate` (mesmo algoritmo, diferença por backend/solver/RNG), `adapted` (mudança
   deliberada, reimplementação nativa ou correção de bug).

## Testes e docs (espelham o R)

- Testes unitários em `tests/` (pytest). Adicionar casos para as 3 classes
  (recuperação de atributos informativos no iris; ranking; top/cutoff; relief distingue
  atributos relevantes de ruído).
- Notebooks executáveis em `notebooks/transf/*.md` (jupytext) → renderizados para
  `examples/` por `python tools/build_examples.py` (demora; roda todos os notebooks).
  Criar/atualizar o notebook de feature selection.

## Checklist de release 0.7.0

1. `pip install -e .` (o Python local pode ter o build do PyPI em cache).
2. Implementar as 3 classes + registrar em `src/pydaltoolbox/transform/__init__.py` e
   em `src/pydaltoolbox/__init__.py` (`__all__`).
3. Docstrings com Teoria + References + Porting.
4. Testes: `python -m pytest -q` (hoje 113 verdes) e `ruff check src tests` (limpo).
5. Bump de versão para `0.7.0` em `pyproject.toml` e `src/pydaltoolbox/__init__.py`.
6. Atualizar `README.md` (lista de feature selection) e `develop/planejamento.md`
   (marcar 1.6 ✅ no mapa da §8; ajustar §1.6 conforme a decisão fss/stepwise).
7. `python tools/build_examples.py` (confirmar que todos os exemplos renderizam).
8. Commit em `main` (padrão das releases 0.2.0–0.6.0 = commit direto em main), mensagem
   "Release 0.7.0 — ...". Terminar a mensagem com:
   `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
9. Publicar (Eduardo autoriza por release): `rm -rf dist build *.egg-info src/*.egg-info`
   → `python -m build` → `python -m twine check dist/*` → conferir que `develop/` NÃO
   está no sdist → `python -m twine upload dist/*` (token no `C:\Users\eduar\.pypirc`,
   seção `[pypi]`). Verificar no endpoint versão-específico
   `https://pypi.org/pypi/pydaltoolbox/0.7.0/json` (o agregado fica em cache logo após).
10. `git push origin main`. Atualizar a memória `[[pydaltoolbox-port]]` marcando 0.7.0
    publicado e o próximo passo (1.7 → 0.8.0: mineração de padrões base + itemsets).

## Observações

- `gh` CLI não está instalado; operações de git via linha de comando.
- Shell primário é PowerShell; a ferramenta Bash também está disponível (sintaxe POSIX).
- Confirmar cada decisão de design com Eduardo antes de implementar.
