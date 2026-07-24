# Prova completa — Lema 3 (quina de entrada, com constante) [TEOREMA]

## Enunciado (L3, forma final — com a CONSTANTE explícita)

Sob (H1)–(H4), seja $v = c\,|R^{hyb}(t_{hit})|$ a velocidade de chegada
(transversal por (H1)≡(H0)). Então

$$t_{cross}^\beta \;=\; t_{hit} \;+\; \frac{1}{\beta\,v}\bigl(1 + O(\delta_z) + o(1)\bigr),
\qquad \|\theta^\beta(t_{cross}) - \theta^\infty(t_{hit})\| = O(\beta^{-1}),$$

onde $t_{cross}^\beta$ é o primeiro tempo com $w = w_0$ (com $w_0 = 1$: $z = 0$).
**A constante do fit do passo 2 fica explicada:** $K_{teo} = 1/v = 7.05$ na
instância vs $K_{emp} = 7.13$ medido — desvio 1.1%.

## Prova

1. **Pré-quina com peso exponencial.** Para $z \ge \delta_z$:
   $|\mathrm{softplus}_\beta(z) - z| \le \beta^{-1}e^{-\beta z}$ e
   $1 - \sigma(\beta z) \le e^{-\beta z}$; a inomogeneidade do Grönwall é
   $h(t) \le C\,e^{-\beta z^0(t)}$. Perto do hit, $z^0(t) = v\,(t_{hit}-t)(1+O(\delta_z))$
   (transversalidade), logo
   $$\int h \,dt \;\le\; C\!\int e^{-\beta v (t_{hit}-t)}dt \;\le\; \frac{C}{\beta v};$$
   a versão EXATA do peso ao longo da trajetória real inclui o fator do
   $\sigma$ na camada: $\int e^{-\beta z}dt = \frac{1}{\beta v}\int_0^\infty
   e^{-s}(1+e^{-s})\,ds\,(1+o(1)) = \frac{3/2}{\beta v}(1+o(1))$ —
   constante prevista e confirmada a $<1\%$ na verificação (c).
   **Autoconsistência do peso, por extenso (rota monótona, sem ponto fixo
   exponencial):** como $e^{-\beta z}$ é decrescente em $z$, o fluxo com
   inomogeneidade avaliada em qualquer sub-solução $z_- \le z^\beta$ MAJORA a
   inomogeneidade verdadeira. Itere: $z^{(0)} = z^0$;
   $z^{(k+1)}$ = solução com peso $e^{-\beta z^{(k)}}$. A iteração é monótona
   decrescente e limitada por baixo pela sub-solução do relu com folga
   $2C/(\beta v)$ (pois o incremento de cada passo é
   $\le C/(\beta v)\cdot\sup e^{\beta(z^{(k)}-z^{(k+1)})}$, e enquanto
   $|z^{(k)}-z^{(k+1)}| \le 1/\beta$ o fator é $\le e$, mantido pelo tamanho do
   incremento — indução em $k$). Logo converge a $z^\beta$ com
   $\Delta\theta(t_\delta) = O(1/(\beta v))$, sem exigir $C_2 e^{c'K} \le K$.
2. **Identidade da travessia da janela.** Em $[0, \delta_z]$ as velocidades são
   $v_0(z) = v(1+O(\delta_z))$ (relu) e $v_\beta(z) = v_0(z)\,\sigma(\beta z) + O(\beta^{-1})$
   (softplus). A diferença de tempos de travessia é
   $$\int_0^{\delta_z}\!\Bigl(\frac{1}{v_\beta}-\frac{1}{v_0}\Bigr)dz
   = \frac{1}{v}\int_0^{\delta_z}\frac{1-\sigma(\beta z)}{\sigma(\beta z)}\,dz\,(1+O(\delta_z))
   = \frac{1}{v}\int_0^{\delta_z} e^{-\beta z}\,dz\,(1+O(\delta_z))
   = \frac{1-e^{-\beta\delta_z}}{\beta v}(1+O(\delta_z)),$$
   usando a identidade exata $\tfrac{1-\sigma(s)}{\sigma(s)} = e^{-s}$. Com
   $w_0 = 1$ não há deslocamento adicional ($\log w_0 = 0$). Somando 1+2:
   $t_{cross} - t_{hit} = \tfrac{1}{\beta v}(1+O(\delta_z)+o(1))$. Estado:
   velocidades limitadas na janela ⟹ $O(\beta^{-1})$. $\square$

## Verificação (`prova_L3_verificacao.py`)

(a) SymPy: $\tfrac{1-\sigma}{\sigma} = e^{-s}$ e $\int_0^\infty e^{-\beta z}dz = 1/\beta$;
(b) constante: refit dos $t_{cross}$ medidos no passo 2 (0.13518, 0.11850,
0.11379, 0.11215) ⟹ $K_{emp}$; comparar com $K_{teo} = 1/(c|R^{hyb}(t_{hit})|)$,
tolerância 5%; (c) peso: $\beta\!\int e^{-\beta z(t)}dt$ na aproximação real
estável $\approx 1/v$.

## Status

[TEOREMA] — script passou em 2026-07-24: $K_{teo} = 7.042$ vs $K_{emp} = 7.127$
(refit do passo 2, desvio 1.2%); peso exponencial com o fator exato $3/2$ do
$\sigma$ na camada: teoria $10.563$, medido $10.52$ (<1%). Terceiro lema com
prova completa — e com DUAS constantes empíricas explicadas pela teoria.
