# Revisão adversarial do desenho v4 — 2026-07-24 (5º parecer)

**Veredito original: NÃO APTO** — 1 blocker de código + 1 blocker de desenho +
2 should-fix. TODAS as correções aplicadas antes do commit; APTO nos termos do
parecer.

- **Blocker de código:** `A["P1_mesmo_sinal"]` nunca atribuído ⟹ KeyError APÓS
  a run inteira (o smoke não exercita a análise). Corrigido + análise testada
  com dados sintéticos antes do commit (regra nova: analise() sempre testada
  sinteticamente no smoke-gate).
- **Blocker de desenho (com a conta):** sinais puro dispara a 9.9% (não >50%
  como o README supunha — premissa refutada quantitativamente), mas a DIREÇÃO
  está decidida a priori (viés −0.016 = B-clipping do preditor); combinado
  13.7% de falso-MATA (3× a calibração). Piso |Δp|≥0.04 adotado
  (P1_PISO_SINAL): falso-MATA → 4.3%; potência ≈1 contra o fracasso da
  geometria (célula +0.18 daria z≈+7.7).
- **Erro factual corrigido:** não há softplus em nenhuma confirmatória (relu-duro
  idêntico preditor↔experimento); o candidato do −0.016 é B-clipping
  (+ ≤0.004 de bloco). As DUAS pontes que a run testa: B linear-modal vs B-relu
  completo; B em blocos COARSE vs passo fino.
- **Armadilha de caminho:** escada apontava para config inexistente e funcionava
  por sorte do fallback — caminho direto para config_v3.json.
- **Verificados limpos:** seeds 1500–1649 virgens; HARKing zero; réguas
  herdadas intactas; censura 0.4–9.2%; rótulo "1.0" da escada = predições
  determinísticas do preditor-v2 (proveniência conferida).
- Runner paralelizado (Pool 4, nice) após o parecer — mudança de execução, não
  de desenho.
