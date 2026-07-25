# Piloto pressão (etapa-4a.3) — tabela e veredito

_Runs carregados: 9 (esperado 36). T=0, η=3e-4, H=3e5, n=64, alvo rico, seeds 42-45, n_σ=10._

## Tabela (m × ativ), agregado nos seeds presentes

| ativ | m | seeds | loss_plato (min–max) | Σmortos_real | Σrev_real (neur) | Σv2_comp | Σv3_comp | Σv3_zmin<0 | v3 dur med/p90 |
|---|--:|--:|---|--:|--:|--:|--:|--:|---|
| softplus-b300 | 4 | 1 | 0.3046–0.3046 | 3 | 0 (0) | 0 | 0 | 0 | — |
| softplus-b300 | 8 | 1 | 0.2621–0.2621 | 1 | 0 (0) | 0 | 0 | 0 | — |
| softplus-b300 | 16 | 1 | 0.0440–0.0440 | 3 | 0 (0) | 0 | 0 | 0 | — |
| softplus-b30 | 4 | 1 | 0.3046–0.3046 | 3 | 0 (0) | 0 | 0 | 0 | — |
| softplus-b30 | 8 | 1 | 0.2624–0.2624 | 1 | 0 (0) | 0 | 0 | 0 | — |
| softplus-b30 | 16 | 1 | 0.0466–0.0466 | 2 | 0 (0) | 0 | 0 | 0 | — |
| relu | 4 | 1 | 0.3046–0.3046 | 3 | 0 (0) | 0 | 0 | 0 | — |
| relu | 8 | 1 | 0.2620–0.2620 | 1 | 0 (0) | 0 | 0 | 0 | — |
| relu | 16 | 1 | 0.0439–0.0439 | 3 | 0 (0) | 0 | 0 | 0 | — |

## Mortes: born-dead (t_morte=0) vs durante-treino (t_morte>0)

_Só a morte DURANTE-treino é o objeto de T2' (neurônio vivo que morre). z_min raso/fundo = a mais rasa e a mais funda das mortes durante-treino da célula._

| ativ | m | Σborn-dead | Σdurante-treino | z_min raso | z_min fundo | Σrev_real |
|---|--:|--:|--:|--:|--:|--:|
| softplus-b300 | 4 | 2 | 1 | -0.0206 | -0.0206 | 0 |
| softplus-b300 | 8 | 1 | 0 | — | — | 0 |
| softplus-b300 | 16 | 2 | 1 | -0.0168 | -0.0168 | 0 |
| softplus-b30 | 4 | 2 | 1 | -0.1644 | -0.1644 | 0 |
| softplus-b30 | 8 | 1 | 0 | — | — | 0 |
| softplus-b30 | 16 | 2 | 0 | — | — | 0 |
| relu | 4 | 2 | 1 | -0.0000 | -0.0000 | 0 |
| relu | 8 | 1 | 0 | — | — | 0 |
| relu | 16 | 2 | 1 | -0.0000 | -0.0000 | 0 |

## Veredito (i)-(iv) — sem amaciar

**(i) loss platôia em valor > 0 (pressão persiste):**
- loss_plato mínimo sobre todos os runs = 4.3942e-02 (piso de 'pressão' arbitrado a 1e-03). CONFIRMA.
  - softplus-b300: m4=0.3046, m8=0.2621, m16=0.0440  (esperado: cresce quando m cai)
  - softplus-b30: m4=0.3046, m8=0.2624, m16=0.0466  (esperado: cresce quando m cai)
  - relu: m4=0.3046, m8=0.2620, m16=0.0439  (esperado: cresce quando m cai)

**(ii) softplus tem ressurreições genuínas (>0, dur física):**
- softplus-b300: Σrev_real=0 (por m: {4: 0, 8: 0, 16: 0}), Σv3_completos=0.
- softplus-b30: Σrev_real=0 (por m: {4: 0, 8: 0, 16: 0}), Σv3_completos=0.
- softplus tem 3 mortes DURANTE-treino (t_morte>0) no total (alvo real de T2'); a mais rasa mergulhou a z_min=-0.016772763803601265 (margem-líder negativa em TODOS os dados).
- **FALHA**: softplus NÃO ressuscita (morto-real->vivo = 0) mesmo com pressão persistente (loss-platô>0) e mortes ocorrendo. Mecanismo: a T=0 (determinístico) a margem-líder mergulha FUNDO (z_min≪0), onde σ(βz) é desprezível ⟹ sem gradiente de retorno e sem ruído p/ escapar ⟹ morte TERMINAL. É a condição-que-MATA a hipótese descrita no pré-desenho.

**(iii) relu: EXATAMENTE 0 ressurreições de morto-real (grad≡0):**
- ground-truth Σrev_real(relu) = 0 (morto-real zmax<0 -> vivo zmax>0).
- régua v3 Σzmin<0 completos(relu) = 0 (episódios v3 que ressuscitaram tendo mergulhado <0).
- régua v3 Σcompletos(relu) = 0 (inclui rasos: 0<margem<lim_morte, neurônio NUNCA morto-real).
- CONFIRMA: relu não ressuscita morto-real. Episódios v3 de relu (se houver) são de margem-líder rasa (nunca <0), i.e. o limiar da régua > 0 pegou quase-mortos.

**(iv) β menor ⟹ mais ressurreições (β30 vs β300):**
- m=4: β300 rev_real=0 v3=0  |  β30 rev_real=0 v3=0
- m=8: β300 rev_real=0 v3=0  |  β30 rev_real=0 v3=0
- m=16: β300 rev_real=0 v3=0  |  β30 rev_real=0 v3=0
- total: β300 Σrev_real=0  vs  β30 Σrev_real=0. direção NÃO confirma (exploratório, sem estatística).
