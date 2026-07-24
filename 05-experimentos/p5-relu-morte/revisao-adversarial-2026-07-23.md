# Revisão adversarial do pré-registro — 2026-07-23

**Veredito: NÃO APTO PARA COMMIT como teste falsificável.** Revisor independente
(agente adversarial, protocolo da casa: "um resultado só existe quando sobreviveu a
uma tentativa honesta de destruí-lo"). Nenhuma trajetória confirmatória foi executada
pela revisão (só contas analíticas e simulações de processos genéricos — BM em cone, OU).

## FATAL

**F1 — As três predições P-a/b/c são decidíveis a priori; nenhuma pode MATAR a
hipótese.** Mesma patologia do p5-cruz num nível mais sutil:

- **P-b**: após a saída do cone, a força de L2 mantém a coordenada normal em
  `z_j ≈ y_j` com desvio estacionário `√(nT) = 0.045` ⟹ re-entrada exige 11σ (face 0)
  a 22σ (face 1). `f_SGLD ≈ 0` garantido; braço nulo (BM 2D recorrente) dá
  `f_nulo ≈ 0.5`. A razão ≫ 5 em qualquer mundo em que os lemas valem.
- **P-c**: a deriva do softplus em D é `∝ σ(βz)` com z<0, estritamente decrescente
  em β ⟹ monotonicidade por construção; convergência segue do limite uniforme
  softplus_β → relu. Remedição de fato conhecido.
- **P-a**: o alvo protocolar `salto*` é derivado da mesma teoria que gera o efeito;
  desvio esperado ~2–8% contra tolerância de 25% (folga ~12×); razão vs softplus_2
  ≈ 3.98 com limiar 3 logo abaixo (o salto do softplus_2 é essencialmente o piso de
  ruído 2σ_c = 0.20). Passa por construção dado K=200.

**E o ponto de fundo:** nenhuma das três discrimina "sweeping process /
complementaridade" (P5) de "a loss tem um kink e um vale" — fato trivial. Um sistema
kinkado qualquer, não-sweeping, satisfaria P-a/b/c igualmente.

## GRAVE

- **G1** — `SALTO_PROT` erra a esperança do que o código mede em até ~9%: (i) Jensen
  (√E‖·‖² vs E‖·‖, distribuição de Rice: fator 0.984 face 1 / 0.923 face 0);
  (ii) janelas truncadas (MIN=100 admite L∈[100,200) com κ e piso de ruído diferentes).
  Correção: alvo via média de Rice por comprimento real de janela (ou Monte Carlo
  forward do estimador) e `MIN_PASSOS_VALIDOS_JANELA = K_EVENTO`.
- **G2** — Censura verdadeira ≈ 18.7% em 500k passos (first-passage de BM no cone);
  `P(≥7/20 censurados) ≈ 6.4%` ⟹ ~1 em 15 execuções invalida o braço primário por
  azar. Correção: N_PASSOS = 1e6 (censura ~10%) ou mais seeds.

## MENOR

- M1 — eventos face 0 (~17%) têm razão individual ~1.5 (<3); só o agregado dominado
  pela face 1 salva; razão calculada sobre IQMs não pareados. Reportar por face/seed.
- M2 — `salto_prot` gravado com T_PRIMARIA nas linhas exploratórias do jsonl.
- M3 — `v_ant` com janela curta se `t_res < K` (na prática não dispara; mediana de
  saída ~53k passos).
- M4 — passos válidos não-contíguos na janela pós vs κ contíguo (desprezível).
- Conferido correto: pareamento de seeds entre braços (ruído gerado antes do gradf;
  relu ≡ nulo dentro de D até a primeira saída), instrumento P-b(i), convenção
  `morto()` com ≤, e toda a aritmética de `SALTO_PROT` (ρ, κ, σ_c, fator 4σ_c²).

## Caminho para APTO (indicações do revisor + síntese)

1. Ou **rebaixar honestamente** o artefato a DEMONSTRAÇÃO + lemas (sem linguagem de
   falsificação; não sustenta sozinho nota de "dinâmica É contato");
2. ou **redesenhar com risco real**: regime onde contato-vs-suave compete de verdade
   (σ_c maior / tolerância ~5% onde o modelo do instrumento pode falhar) e/ou uma
   predição que um sistema kinkado NÃO-sweeping violaria;
3. direção mais promissora (síntese pós-revisão): **≥2 neurônios** — para um neurônio
   morto, o piso continua exatamente plano (∇ próprio = 0), mas a FRONTEIRA se move
   porque o resíduo muda com o aprendizado dos outros neurônios ⟹ o conjunto de
   restrição vira C(t) MÓVEL — o ingrediente definidor do sweeping process que o caso
   1-neurônio não tem, e onde existem predições genuinamente não decidíveis a priori
   (ressurreição vs morte permanente conforme o resíduo esvazia a reação).
