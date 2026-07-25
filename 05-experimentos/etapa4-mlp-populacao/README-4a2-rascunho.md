# etapa-4a.2 — ESTUDO DE POTÊNCIA: a física da área existe e reproduz na MLP?

**STATUS: RASCUNHO DE DESENHO — para 8º parecer adversarial. NÃO congelado,
NÃO rodar.** Herdeiro do NEGATIVO da 4a (P-NULL; NEGATIVOS.md) e da
fase-instrumento verificada (commit c2a4d1f).

Etiqueta global: **[ESPECULAÇÃO]** — e mais fraca que a da 4a: aqui NEM O SINAL
está estabelecido. O piloto viu razão física/difusão 3.76 em (T=1e-5, m=64)
**dirigida por 1 seed de 3**; sem ela, 1.25. Este experimento NÃO é
confirmatório de nada: é um estudo de potência/existência com desfechos
pré-registrados nos DOIS sentidos.

## 0. O que a fase-instrumento estabeleceu (verificado, commit c2a4d1f)

1. **Régua v3** (histerese+dwell, 1 botão n_σ=10): rejeita 97% do chatter,
   preserva o toy (erro-área 0.586 pareado), t_morte exato. **Profundidade
   DESLIGADA** (a morte do T2′ é rasa: exigir z_min≤−lim rejeita 24/26 da
   física genuína e ganha só +1.8pp contra chatter).
2. **T=1e-3 é regime MORTO**: pós-debounce, relu/softplus = 0.95 (verificado);
   os sobreviventes são difusão. **Fome de gradiente a β=300**: softplus
   morta-profunda tem σ(βz)≈0 — tão congelada quanto ReLU; em T=0, 0≡0.
3. **Nulo estratificado por neurônio** (permutação de atrasos dentro-do-k,
   truncamento reaplicado): pooled quebra 30–92×α sob heterogeneidade/
   autocorrelação (o mecanismo do P-NULL real); estratificado segura ≈α
   (verificado). Custo: potência exige ~10 episódios/neurônio (TPR 0.16 com
   ~2; 1.00 com ~10).
4. **Regime**: janela estreita — T=1e-5 único T com separação (3.76 vs 1.34
   em 1e-4, ~1.0 em ≥1e-3); m=64 único m (m=16 faminto de episódios; m≥128
   washout 0.90–0.98); horizonte 3e5 (p99 da ressurreição ~1.6e5; censura
   ~27–48%). T=0 nunca cumpre censura<50%: âncora de sobrevivência, não célula
   de contagem.

## 1. Perguntas e desfechos pré-registrados (estrutura)

- **Q1 (EXISTÊNCIA — o desfecho primário).** Na célula primária (softplus
  β=300, T=1e-5, m=64, H=3e5), o excesso de episódios físicos sobre a difusão
  reproduz entre seeds? Estatística: por seed s (pareada por seed/init/ruído
  entre braços), N_fis(softplus,s) vs N_fis(semgrad,s) — episódios v3 limpos
  completos. Teste de sinais entre S=40 seeds + razão agregada com CI
  bootstrap-por-seed. **Desfecho A (sinal existe):** razão significativa >1 e
  ≥k* seeds positivas. **Desfecho B (MATA da janela):** CI∋1 ⟹ o excesso do
  piloto era flutuação de 1 seed ⟹ física não mensurável nesta janela com
  este instrumento — NEGATIVO registrado, e a etapa-4 encerra com o mapa.
- **Q2 (MEMÓRIA — condicional a Q1=A, senão só descritiva).** Nos episódios
  softplus da célula primária: teste ESTRATIFICADO por neurônio (área vs
  placebo dentro-do-k, truncamento reaplicado), p<α=0.01. Gate de potência
  PRÉ-REGISTRADO: só é MATA se a mediana de episódios/neurônio ≥ 8 (senão
  reporta "sem potência", sem veredito — lição TPR 0.16).
- **Q3 (JANELA/WASHOUT — estimação).** Razão softplus/semgrad vs
  m∈{16,64,128,256} em T=1e-5: a curva com CI. A janela estreita, SE
  reproduzir, é resultado científico (lei de m corpos lava a física).
- **Q4 (FOME DE GRADIENTE — mecanismo, braço secundário).** softplus β=30
  (+semgrad pareado) na célula primária: se a fome de gradiente é real, β=30
  deve ter MAIS episódios físicos que β=300 (gradiente do morto ~σ(βz) não
  zera). Estimação com CI; informa a 4a.3 e a ponte GELU (β efetivo ~O(1)).
- **P-NULL (validade, obrigatório).** O teste de Q2 no braço semgrad não pode
  passar em nenhuma célula (nulo estratificado já demonstrado ≈α em 3 famílias
  de mundos; aqui é a checagem no mundo real).
- **Âncora T=0** (softplus m=64): curvas de sobrevivência KM da duração;
  sem MATA (censura estrutural), referência de física pura.

## 2. Grade (a congelar em preditor_4a2)

| Bloco | Braços | m | T | H | seeds |
|---|---|---|---|---|---|
| Primário | softplus-β300, semgrad | 64 | 1e-5 | 3e5 | 40 |
| Washout | softplus-β300, semgrad | 16, 128, 256 | 1e-5 | 3e5 | 30 |
| β-moderado | softplus-β30, semgrad-β30 | 64 | 1e-5 | 3e5 | 30 |
| Âncora | softplus-β300 | 64 | 0 | 3e5 | 30 |

≈ 80+180+60+30 = **350 runs**; ~2.0–3.6 min/run ⟹ ~15 core-h ⟹ **~4 h no
Pool 4 nice-19** (CPU pinada; device único como na 4a). Seeds virgens
1800–1839/1859. η=3e-4. Régua v3 n_σ=10, profundidade OFF, Δsnapshot=1.
Filtro-limpo herdado (q=2, q_j=0, φ_v=1/(4√m)) sobre episódios v3.

## 3. Análise (analise_4a2.py, a construir APÓS o parecer)

Testada em sintético COM os três mundos-nulo (iid, hetero, autocorr — regra
endurecida pelo 7º parecer: o nulo fabricado tem de conter o mecanismo que
quebrou o anterior). Q1: sinais+bootstrap por seed; Q2: p_estratificado (de
nulo_estratificado.py, já verificado); censura como na 4a; sem NENHUM número
fora do config.

## 4. Riscos declarados

1. Desfecho B é o mais provável a priori (1 seed de 3 no piloto) — o desenho
   trata B como resultado pleno (mapa da janela + negativo), não fracasso.
2. Episódios/neurônio podem ficar <8 na célula primária ⟹ Q2 sem veredito —
   por isso o gate; reportar a distribuição.
3. Runtime de semgrad ≠ softplus (57–95 vs 122–144 s): pareamento é por
   seed/init, não por passo-a-passo entre braços — declarar que a comparação
   Q1 é de CONTAGEM no mesmo H, robusta a isso.
4. β=30 muda lim_morte? Não (limiar não depende de β), mas muda a escala de
   profundidade da morte — reportar z_min por braço.

## 5. Perguntas explícitas ao 8º parecer

(a) Q1 é decidível a priori? (o semgrad pareado é controle forte — mas o
excesso pode ser trivialmente ≥0? Não: back-reaction pode SUPRIMIR episódios;
avaliar). (b) o gate de 8 ep/neurônio em Q2 é honesto ou é escape? (c) o teste
de sinais de Q1 precisa de piso de efeito (fio-de-cabelo)? (d) β=30 como
estimação está bem rotulado ou vira MATA disfarçado? (e) o que falta congelar.
