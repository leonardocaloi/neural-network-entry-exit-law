# Etapa-4a: P-NULL dispara — instrumento viciado, experimento INVÁLIDO (2026-07-24, exp05, pré-reg 6411ac8)

**TL;DR (3 linhas):** A grade confirmatória de 600 runs rodou íntegra e a análise
pré-registrada deu o veredito que congelamos para este caso: o braço de difusão
pura passou no teste calibrado (m=16, p=0.002) ⟹ **INSTRUMENTO VICIADO — nenhum
veredito científico de P-A/P-B/P-D pode ser reivindicado**. Diagnóstico: a régua
mede majoritariamente *chatter de fronteira* (59% dos episódios ≤5 passos), não
a morte física do teorema.

**Status epistêmico: [NEGATIVO]** — negativo de INSTRUMENTO (não falseia a lei
de área; falseia a capacidade desta régua de testá-la neste regime).

## Setup

MLP rasa m∈{16,64,256}, 1D-reg n=32, SGLD η=3e-4, 10 braços × 20 seeds
(1700–1719) × 1e5 passos, CPU pinada, tudo congelado em `config_etapa4.json`
(pré-registro 6411ac8; 6º/7º pareceres + 2ª rodada APTO no repositório).
Detalhes: `05-experimentos/etapa4-mlp-populacao/`.

## Resultado (vereditos pré-registrados, na ordem em que caíram)

| Predição | Veredito congelado | Números |
|---|---|---|
| **P-NULL** | **INSTRUMENTO VICIADO ⟹ experimento INVÁLIDO** | semgrad m=16: p=0.002 (T_obs=0.0172, 3132 limpos); m=64: p=0.46; m=256: p=0.49 |
| P-A | (anulado pelo P-NULL) registrado: parcial — só m=64 (p=0.004); m=16 p=0.041, m=256 p=0.53 | T_obs minúsculos (0.017–0.052): regime dominado por empates |
| P-B | (anulado) registrado: universalidade in-class sustentada | — |
| P-D | (anulado) registrado: "sem back-reaction" com Δ≡0 e piso≡0 — degenerado pelo chatter | rev_freeze = rev_controle = 0.17 EXATOS |
| P-E | descritivo válido como diagnóstico | abaixo |

## Diagnóstico (exploratório, pós-hoc, rotulado)

1. **Duração mediana dos episódios: 2–4 passos** em TODOS os braços T>0
   (toy calibrado: ~3×10³). 59% ≤5 passos; só ~10% ≥100 passos.
2. **Prova de que é chatter, não física:** ReLU-dura "ressuscita" na MESMA taxa
   e com a MESMA distribuição de durações que softplus em T>0 (med 4, 10.2% ≥100)
   — impossível por gradiente (morto de ReLU tem gradiente exatamente zero);
   logo são cruzamentos de limiar por ruído.
3. **O braço T=0 contém a física real:** únicos episódios longos genuínos
   (mediana 1110 passos), 99.6% censurados no horizonte de 1e5 — o fenômeno do
   teorema existe na MLP, mas vive em T→0 e em escalas ≥1e5 passos.
4. **Por que P-NULL disparou em m=16 e não em m=64/256:** com poucos neurônios,
   cada um contribui MUITOS episódios repetidos com assinatura própria (dur,
   atraso) — heterogeneidade por neurônio viola o H0 iid do nulo calibrado;
   com centenas de neurônios a mistura aproxima iid. O nulo segurou FPR≈α em
   mundos iid (demonstrado no --sintetico e no ataque do 7º parecer com 7
   famílias), mas o mundo real não é iid dentro da célula.
5. Tendência temporal comum (caveat pré-registrado): corr(dur,t_morte) e
   corr(atraso,t_morte) ≈ 0 em todas as células — essa objeção está morta.

## O que este negativo NÃO diz

Não falseia a lei de área (T2′): o teste nunca chegou a acontecer no regime
onde a lei vive. Não invalida o toy (calibrações v1/v2 íntegras). Não invalida
a infraestrutura (600/600 runs determinísticas, manifest SHA256, análise sem
nenhuma decisão pós-hoc).

## O que mataria a PRÓXIMA tentativa (herança para etapa-4a.2)

Experimento NOVO com pré-registro novo (regra da casa: hipótese nova = exp
novo): (1) **debounce por permanência/histerese de dois limiares** na régua —
separar morte física de chatter; recalibrar no toy COM ruído; (2) **nulo
estratificado por neurônio** (permutação dentro-do-neurônio) — mata a
heterogeneidade que enganou o iid; (3) regime: T-sweep para BAIXO + horizonte
≥5e5 no braço T pequeno (os episódios reais têm ~1e3+ passos e 99.6% censura
em 1e5); η menor se necessário.

## Reprodução

`cd 05-experimentos/etapa4-mlp-populacao && python run.py && python analise.py`
(determinístico; raw local com SHA256 em results/manifest.json; agregados em
resultados.json commitado).
