# Piloto-pressão: mecanismo de ressurreição OBSERVADO em MLP; lei de área 2/2; eventos raros (2026-07-25, exploratório, pilotos-4a2/pressao)

**TL;DR (3 linhas):** Em regime subparametrizado (pressão: platô ~0.15) a T=0,
a ressurreição-por-mecanismo EXISTE em MLP e a ReLU-dura dá ZERO estrutural
(16 seeds, 0 revividas de 25 mortes). Os 2 episódios dinâmicos ressuscitados
têm o trio ORDENADO t_morte < t_área < t_res (lei de área qualitativa 2/2).
Gargalo quantificado: eventos raros (~0.1/run) — desenho 4a.3 = engenharia de
taxa de eventos.

**Status epistêmico: [EVIDÊNCIA NUMÉRICA exploratória]** — piloto com seeds
42–49, SEM pré-registro; NADA aqui é confirmatório. Serve para desenhar a 4a.3.

## Setup

64 runs: m∈{4,6} × {softplus β10/β30/β300, relu} × seeds 42–49; n=16 dados,
alvo rico y=sin(2πx)+0.3sin(5πx) (subparam ⟹ platô 0.14–0.16), η=1e-3, T=0,
H=2e5; régua dupla v2/v3 + ground-truth independente de régua
(morte-real = zmax<0 em todos os dados). Cadeia de descoberta: célula lenta
(n=64, η=3e-4) mostrou portão-de-área aberto sem volta ⟹ conta de timescale
(deriva ~η·v·R·σ(βz_park)·‖x‖²/n, ~3000× mais lenta que o toy) ⟹ probe
confirmou na 1ª tentativa (Fable, commit desta nota).

## Resultados

1. **Controle ReLU: zero estrutural.** 16 runs, 20 nascidas-mortas + 5 mortes
   dinâmicas, 0 ressurreições (ground-truth e réguas concordam). O ramo
   congelado (grad≡0) é exato em rede real a T=0.
2. **Softplus ressuscita por mecanismo.** β=10: 1 dinâmica (s42: morte 781 →
   volta 139 179) + 1 nascida-morta (s43); β=300: 1 dinâmica (s47-m6: morte
   3315 → volta 152 977 — estacionou RASO o bastante para o canal e^{βz} não
   zerar). β=30: 0 (com 5 mortes dinâmicas — flutuação com n=5 eventos).
3. **Lei de área qualitativa: 2/2 trios ordenados** (45k e 147k de abertura
   antes da volta). O gap portão→volta é o tempo de subida a β finito
   (~|z_park|/deriva), NÃO a correção O(1/β) assintótica — modelar na 4a.3.
4. **Gargalo = taxa de eventos:** mortes dinâmicas ~0.3/run; retorno
   ~1.4–1.5e5 ≈ H ⟹ censura de mortes tardias (2 flagradas). (ii') "≥metade
   das seeds" NÃO atingido; (iv') sem potência com 1–2 eventos por braço.

## O que mataria a 4a.3 (herança)

Desenho confirmatório precisa de ≥10–15 eventos esperados no braço primário:
H=5e5 (+ retorno ~1.5e5 cabe 3×), mais seeds (≥40–100), e/ou taxa de morte
dinâmica maior (m/alvo a explorar EM PILOTO, nunca pós-hoc). Estatística:
contagem Poisson vs zero-estrutural da relu (pareada por seed) + fração de
trios ordenados + gap vs z_park. Nulo estratificado herdado se T>0 voltar.

## Reprodução

pilotos-4a2/pressao/: treina_pressao.py (paridade bit-a-bit com harness
verificada), roda_*.py, results/*-rapida.jsonl.gz (65 arquivos). Análise desta
nota: inline no transcript (chaves n_born_dead/n_dinamicas/episodios[classe]).
