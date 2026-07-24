# 05-experimentos

Um experimento = uma pasta. Regras (detalhadas na skill `experimento-protocolo`):

1. **README.md antes do código**: hipótese, predição falsificável (a condição que MATA a hipótese), e baseline nulo — escritos antes da primeira execução.
2. **Determinismo**: seed explícita em tudo; config em arquivo, não hardcoded.
3. **Resultados em `results/`** (jsonl + figuras). Resultado negativo é resultado: fica registrado, nunca apagado.
4. **`ferramentas/`** é o lugar do tooling que nós mesmos criarmos (estimadores de r_N, medidores de contato, wrappers de LLC etc.) — reutilizável entre experimentos.

## Experimentos

| pasta | pergunta | status |
|---|---|---|
| `p5-cruz/` | a dinâmica SGLD na cruz L=w1²w2² exibe assinatura de contato unilateral? | planejado, não rodado |
