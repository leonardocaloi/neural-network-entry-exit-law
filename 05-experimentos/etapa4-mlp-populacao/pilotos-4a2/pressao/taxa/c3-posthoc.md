# C3 (gap-de-subida) — AGREGACAO POST-HOC, EXPLORATORIA

> **[ESPECULACAO / POST-HOC]** C3 foi pre-registrado como *estimation-only*, sem condicao de MATA, e a analise congelada `analise_4a3.py` NAO contem agregado para ele. Este arquivo foi produzido por `agrega_c3_posthoc.py` DEPOIS de ver os dados confirmatorios, a pedido de revisor externo. Nada aqui e' confirmatorio; nenhum numero abaixo testa hipotese pre-registrada. N e' pequeno (16 voltas, poucas seeds) e as conclusoes devem ser lidas como geracao de hipotese para a etapa seguinte.

- runs lidos: 400; braco analisado: `sp-b3` (primario do 4a.3)
- voltas dinamicas (denominador do C2): **16**
- voltas COM area aberta (t_area>=0): **13**  (= as ordenadas do C2)
- seeds distintas com volta: 10

## 0. Honestidade estrutural (ler ANTES dos numeros)

O relogio GT-AREA acumula `A_hat` **apenas em passos em que o neuronio esta morto**, e o episodio **fecha na volta**. Logo `t_area < t_res` sempre que a area abre — a desigualdade e' **por construcao**, nao e' achado. O conteudo empirico de C2/C3 e':

1. a area ter de fato virado positiva em **13/16** voltas (o modo de falha real e' *volta sem area*, e ele ocorreu 3 vezes);
2. a **magnitude** do lead time `t_res - t_area` e a questao de se `t_area` carrega informacao sobre `t_res` **alem** da que a mera escala temporal do episodio ja' carrega (secao 2).

Alem disso: a **drive media do episodio nao esta gravada**. O motor (`motor_4a3.py`) acumula `A_hat` in-place e persiste apenas o trio de tempos, `z_park` e `sigma_park`; nao ha' serie de `R_hat` por episodio. Onde a tarefa pede 'drive', usamos o proxy disponivel: profundidade `|z_park|` e canal `sigma(beta*z_park)`. Isso e' um substituto, nao a drive.

## 1. Lead time do portao (t_res - t_area), n = 13

quantidade | min | q25 | mediana | q75 | max | media
---|---|---|---|---|---|---
lead = t_res - t_area | 347 | 6661 | 111373 | 180199 | 334436 | 119723.5
lag  = t_area - t_morte | 0 | 0 | 0 | 0 | 98973 | 12179.6
dur  = t_res - t_morte | 3944 | 65511 | 111373 | 180199 | 334436 | 131903.2

Fracao do episodio ja' decorrida quando a area abre (`lag/dur`): mediana 0.000, min 0.000, max 0.997.
- IC95 da mediana do lead: **[6661, 258014]** (bootstrap de seed) / [6661, 180199] (bootstrap de episodio, otimista)
- IC95 da mediana de lag/dur: **[0.000, 0.000]** (bootstrap de seed) / [0.000, 0.000] (bootstrap de episodio, otimista)

Episodios (um por linha):

seed | m | k | t_morte | t_area | t_res | dur | lead | lag/dur | z_park | sigma_park
---|---|---|---|---|---|---|---|---|---|---
1914 | 6 | 2 | 51493 | 51493 | 309507 | 258014 | 258014 | 0.000 | -0.1416 | 0.3954
1918 | 4 | 2 | 5803 | 5803 | 9747 | 3944 | 3944 | 0.000 | -0.0275 | 0.4794
1919 | 4 | 0 | 1969 | 100942 | 101289 | 99320 | 347 | 0.997 | -0.9335 | 0.0573
1919 | 6 | 5 | 70875 | 70875 | 224660 | 153785 | 153785 | 0.000 | -0.0139 | 0.4896
1923 | 6 | 0 | 277766 | 277766 | 401674 | 123908 | 123908 | 0.000 | -0.2360 | 0.33
1924 | 4 | 3 | 128087 | 128087 | 308286 | 180199 | 180199 | 0.000 | -0.1874 | 0.363
1924 | 6 | 2 | 1020 | 1532 | 18079 | 17059 | 16547 | 0.030 | -0.0970 | 0.4277
1934 | 6 | 3 | 151330 | 151330 | 485766 | 334436 | 334436 | 0.000 | -0.0834 | 0.4378
1937 | 6 | 3 | 111796 | 111796 | 196987 | 85191 | 85191 | 0.000 | -0.0598 | 0.4552
1937 | 6 | 4 | 1968 | 1968 | 7674 | 5706 | 5706 | 0.000 | -0.0028 | 0.4979
1939 | 6 | 5 | 4336 | 63186 | 69847 | 65511 | 6661 | 0.898 | -0.1275 | 0.4055
1948 | 4 | 2 | 24290 | 24290 | 300585 | 276295 | 276295 | 0.000 | -0.0387 | 0.471
1949 | 6 | 1 | 347591 | 347591 | 458964 | 111373 | 111373 | 0.000 | -0.1746 | 0.3719

Voltas SEM area (contadas CONTRA a lei no C2):

seed | m | k | t_morte | t_res | dur | z_park
---|---|---|---|---|---|---
1923 | 6 | 1 | 35568 | 289371 | 253803 | -0.3552
1924 | 4 | 2 | 143 | 380852 | 380709 | -0.7762
1934 | 6 | 4 | 1113 | 59971 | 58858 | -0.2034

## 2. t_area prediz t_res alem da escala temporal do episodio?

Spearman bruto e PARCIAL (residuos de postos, OLS com intercepto). ICs = percentil de bootstrap **de seed** (clusteriza a dependencia dentro de uma mesma seed); em parenteses o bootstrap de episodio, que ignora essa dependencia e portanto e' otimista.

estatistica | valor | IC95 (boot de seed) | IC95 (boot de episodio)
---|---|---|---
rho(t_area, t_res) bruto | +0.791 | [+0.333, +0.944] | [+0.364, +0.950]
rho(t_area, t_res | t_morte) | +0.100 | [-0.432, +0.779] | [-0.529, +0.805]
rho(t_area, t_res | dur)  [ver ressalva] | +0.818 | [+0.268, +0.947] | [+0.256, +0.946]
rho(lag, lead)  [decomposicao livre de t_morte] | -0.561 | [-0.876, -0.136] | [-0.879, -0.156]
rho(lead, dur) | +0.912 | [+0.719, +1.000] | [+0.563, +1.000]

**Ressalva sobre `| dur`:** `dur = t_res - t_morte` e' funcao de `t_res`, entao condicionar em `dur` NAO e' um controle limpo — parte da variancia de `t_res` esta' no proprio condicionante. O controle interpretavel e' `| t_morte` (quando o episodio comecou) e, equivalentemente, a decomposicao `rho(lag, lead)`: se a espera ate a area abrir nao informar nada sobre o tempo restante ate a volta, `rho(lag, lead)` fica indistinguivel de zero.

## 3. Lead time contra profundidade / canal (proxy de drive)

estatistica | valor | IC95 (boot de seed) | IC95 (boot de episodio)
---|---|---|---
rho(lead, |z_park|) | +0.005 | [-0.636, +0.747] | [-0.605, +0.628]
rho(lead, sigma_park) | -0.005 | [-0.747, +0.636] | [-0.628, +0.605]
rho(dur, |z_park|) | +0.236 | [-0.533, +0.761] | [-0.390, +0.727]
rho(lag, |z_park|) | +0.383 | [-0.301, +0.745] | [-0.156, +0.759]

A razao `|z_park|/drive` pedida na tarefa **nao e' computavel** com o que foi gravado: nao ha' drive por episodio (secao 0). `sigma_park = sigma(beta*z_park)` e' monotona decrescente em `|z_park|`, entao as duas primeiras linhas sao a mesma informacao com sinal trocado — nao sao evidencias independentes.

## 4. O fato que domina tudo: lag = 0 em 10/13 episodios

Em **10 das 13** voltas com area, `t_area == t_morte` EXATAMENTE: a area do lider ja' e' positiva no primeiro passo morto. So' 3 episodios tem espera nao-nula (lag = 512, 58850 e 98973).

Consequencias, que devem ser ditas antes de qualquer numero da secao 2:

- o lead time NAO e' uma antecedencia informativa do portao: nesses 10 casos ele e' identicamente a duracao do episodio, e a estatistica 'lead' vira um sinonimo de 'quanto tempo o neuronio ficou morto';
- `rho(t_area, t_res)` bruto = +0.79 e' quase inteiramente `rho(t_morte, t_res)`, i.e. episodios que comecam tarde terminam tarde. Controlando por `t_morte` a correlacao desaba para +0.10 com IC95 cruzando zero: **nao ha' evidencia de que o zero da area preveja o tempo de retorno alem de quando o episodio comecou**;
- `rho(lead, dur) = +0.91` e' tautologico pela mesma razao;
- `rho(lag, lead) = -0.56` e' carregada pelos 2-3 episodios de lag grande (os que abrem a area tarde voltam logo em seguida) e nao suporta peso.

Interpretacao mecanicista honesta: sob GT-AREA o portao nao 'aquece' — na maioria dos episodios ele ja' esta' aberto na morte, e o que varia e' quanto tempo a subida leva. Isso e' consistente com a lei de area (a area e' condicao, nao gatilho temporal) e ao mesmo tempo esvazia a leitura de C3 como 'lead time preditivo'.

## 5. Leitura conservadora

Com n = 13 episodios em 10 seeds, os ICs de posto sao largos por construcao; qualquer IC que contenha 0 deve ser lido como **sem evidencia**, nao como ausencia de efeito. Nenhuma correlacao aqui sobrevive a uma correcao para as ~9 estatisticas calculadas, e nenhuma foi pre-especificada. Uso legitimo: dimensionar n e escolher UM estimando pre-registrado para a proxima etapa.

Gerado por `agrega_c3_posthoc.py` (B=20000 reamostragens, seed=20260726).
