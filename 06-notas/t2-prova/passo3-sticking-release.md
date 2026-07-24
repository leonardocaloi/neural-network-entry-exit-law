# T2' — Passo 3: sticking e release com R prescrito [TEOREMA-esboço]

**Setup:** margem escalar com $R(t)$ prescrito ($C^1$, satisfazendo (H1)/(H2)), a
partir do dado uniforme do passo 2 ($w(t_{cross}) = w_0$, $t_{cross} = t_{hit}+O(\beta^{-1})$).

## L3.0 (quadratura em tempo lento — correção de unidades do I4)

De I3, em tempo LENTO $t$: $\frac{d\Phi(u)}{dt} = -c\,\beta\,R(t)\,(1+E_\beta)$, logo

$$\Phi\bigl(u(t)\bigr) = \Phi(u_0) - c\,\beta\,A(t)\,(1+O(\beta^{-1})), \qquad A(t)=\int_{t_{hit}}^{t} R\,ds.$$

O fator $\beta$ é essencial (o I4 do passo 0 estava correto em $\tau$; aqui fica
explícito em $t$).

## L3.1 (profundidade do sticking — a "cola")

Em subintervalos compactos de $(t_{hit}, t_{exit})$, como $A(t) \le -a_0 < 0$ (H2):
$\Phi(u) \sim c\beta|A(t)|$ e, como $\Phi(u) \approx u$ para $u$ grande:

$$u(t) = c\,\beta\,|A(t)|\,(1+o(1)), \qquad z(t) = -\frac{\log\beta + \log(c|A(t)|)}{\beta}(1+o(1)).$$

**Corolário (feedback):** $w = 1/u = O(\beta^{-1})$ ⟹
$\mathrm{softplus}_\beta(z) = \beta^{-1}\log(1+w) = O(\beta^{-2})$ no miolo do
sticking. [Isto CORRIGE a explicação mecanística do passo 1 — lá se afirmou
$u=O(1)$; na verdade $u = O(\beta)$ e o feedback é ainda menor que o alegado.
A cota $O(\beta^{-1})$ de L1.2 permanece válida (dominada pelas janelas de quina);
a superconvergência empírica do passo 1 fica explicada.]

## L3.2 (release pela lei de área, com taxa)

$u$ retorna a $O(1)$ quando $c\beta|A(t)| = O(1)$, i.e. $|A(t)| = O(\beta^{-1})$;
pela transversalidade (H2) ($A'(t_{exit}) = c\,R^0(t_{exit}) > 0$):

$$t_{release}^\beta = t_{exit} + O\!\left(\frac{1}{\beta\; c\,R^0(t_{exit})}\right)$$

— **sem fator $\ln\beta$** (mais nítido que o esqueleto; consistente com o ataque
de 2026-07-24: $\mathrm{err}\cdot\beta$ constante em 3 décadas).

## L3.3 (sanduíche — o dispositivo de prova)

O fluxo de I3 é monótono em $R$: envelopes $R^0 \pm \delta$ dão sub/super-soluções
exatamente integráveis via $\Phi$; combinando com L1.3 (deslocamento de $t_{exit}$
sob perturbação $\delta$): $|t_{release} - t_{exit}| \le \frac{C_1}{\beta\,|R^0(t_{exit})|} + \frac{c\,\delta\,(t_{exit}-t_{hit})}{|A'(t_{exit})|}$.
Com $\delta = O(\beta^{-1})$ (feedback, L1.2), tudo fecha em $O(\beta^{-1})$. $\square$ (esboço)

## Verificação (`passo3_verificacao.py`)

1. SymPy: L3.0 (fator $\beta$) e o corolário $O(\beta^{-2})$ do feedback.
2. Numérico (R do B-sozinho da instância, prescrito): (a) $u_{\max}/(c\beta|A|_{\max}) \to$ const;
   (b) $\beta\,|z_{\min}|/\log\beta \to$ const; (c) $\beta\,(t_{release}-t_{exit})$
   limitado através de $\beta \in \{300,10^3,3\cdot10^3,10^4\}$; (d) feedback no
   miolo $\propto \beta^{-2}$.

## Status

[TEOREMA-esboço] — script passou em 2026-07-24: $u_{max}/(c\beta|A|_{max}) \to 0.98$
(lei de profundidade assintoticamente exata); $\beta(t_{rel}-t_{exit}) \to 0$ (release
$O(1/\beta)$ sem $\ln\beta$); feedback$\cdot\beta^2 \approx 43$ constante em 1.5 década
($O(\beta^{-2})$ no miolo). Draft no passo 7.
