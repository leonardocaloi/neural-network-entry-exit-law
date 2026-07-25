"""Etapa 4a.2 — RÉGUA v3 anti-chatter (trilho 1a; ARQUIVO NOVO, não toca no 4a).

Contexto [RESULTADO 2026-07-24]: a run confirmatória do 4a foi INVALIDADA pelo
controle pré-registrado P-NULL. Diagnóstico (06-notas/2026-07-24-...): em T>0 a
`Regua` v2 mede CHATTER de fronteira — a margem do líder faz um passeio aleatório
em torno do limiar único e cruza-o de volta em ~4 passos (59% dos episódios ≤5;
mediana 4). Prova de que é ruído: ReLU-dura "ressuscita" na MESMA taxa que
softplus em T>0, impossível por gradiente (grad≡0 no morto). A física real do
teorema T2' vive em T→0 (mediana ~1110 passos, 99.6% censurada em 1e5).

Esta régua adiciona um DEBOUNCE com base física (nada de número mágico) sobre a
MESMA lógica de v2 (limiar calibrado, R̂_k, Â_k, flags de v_k). Três mecanismos,
todos derivados de UMA escala física — a difusão por passo da margem:

    σ_z = √(2ηT) · ‖(x_{j*},1)‖         (desvio-padrão do kick de ruído na margem
                                          do líder num passo de SGLD)

pois um passo p ← p + √(2ηT)·ξ − η∇L move a margem z=w·x+b por δz = δw·x + δb,
com Var(δz_ruído) = σ_passo²·(x_{j*}²+1) = (2ηT)·‖(x_{j*},1)‖².

  (1) HISTERESE de dois limiares
        lim_morte = 2√(nT)/‖(x_{j*},1)‖           (o limiar CALIBRADO de v2)
        lim_vida  = lim_morte + n_σ·σ_z           (c_h = 1 + n_σ·σ_z/lim_morte)
      Morte quando margem-líder < lim_morte; ressurreição só quando > lim_vida.
      A banda Δ = lim_vida − lim_morte = n_σ·σ_z é "n_σ larguras-de-ruído" da
      margem: uma flutuação de um passo não atravessa a banda (n_σ>1); só uma
      DERIVA (o mecanismo de área do teorema) a vence. c_h é DERIVADO por célula
      — vale ~1.04 (n=32, líder x≈1) a ~1.11 (líder x≈2), não é constante.

  (2) PERMANÊNCIA / DWELL de D passos
        D = ⌈(Δ/σ_z)²⌉ = ⌈n_σ²⌉
      = tempo de difusão pura (passo σ_z) para atravessar a banda de histerese,
      i.e. a escala de tempo do próprio chatter. Um cruzamento (morte OU
      ressurreição) só é COMETIDO se sustentado por D passos consecutivos; um
      candidato que reverte antes é descartado (chatter rejeitado). t_morte e
      t_res marcam o INÍCIO da corrida sustentada (não o passo de confirmação),
      então tempos físicos são preservados. Note: n_σ fixa a banda E o dwell de
      uma vez só — um único botão físico.

  (3) PROFUNDIDADE (opcional, ligada por default)
        episódio "profundo" ⟺ min da margem-líder no episódio ≤ −c_p·lim_morte
      Chatter fica raso (margem oscila logo abaixo de lim_morte>0, nunca fica
      negativa); a morte física manda a margem fundo no negativo (ativação
      desligada). Não APAGA episódios (regra da casa: negativo é dado) — grava
      o flag `profundo` e o `z_min`; a contagem "que conta" filtra por ele.

Botões físicos expostos: n_σ (banda em larguras-de-ruído ⟹ dwell=n_σ²) e c_p
(profundidade em unidades de lim_morte). A CALIBRAÇÃO fina de n_σ/c_p é o
trilho 1b (toy-COM-ruído); aqui o default n_σ=10 (⟹ banda ~10σ_z, dwell 100,
c_h~1.04–1.11) é a proposta a validar, e os pilotos abaixo reportam a
dependência em n_σ sem ajustá-lo aos dados invalidados do 4a.

interface observa(t, z, a, res, v, x) IDÊNTICA a v2 ⟹ a régua entra no harness
por rebind (ver piloto_regua_v3.py); NÃO modifica harness_mlp.py.
"""
import argparse

import numpy as np

from harness_mlp import Regua, episodio_limpo  # reuso; NÃO modifica o harness

N_SIGMA_DEFAULT = 10.0   # banda em larguras-de-ruído σ_z  (⟹ dwell D = n_σ²)
C_P_DEFAULT = 1.0        # profundidade exigida em unidades de lim_morte


class ReguaV3(Regua):
    """Régua v2 + debounce físico (histerese, dwell, profundidade).

    Herda de `Regua` só para reusar `_fecha`/assinatura e deixar explícito o
    parentesco; TODA a máquina de estados é reescrita aqui (v2 não tinha
    candidatos nem histerese). O(m)/passo, vetorizada.
    """

    def __init__(self, m, eta, T, n_sigma=N_SIGMA_DEFAULT, c_p=C_P_DEFAULT,
                 usar_profundidade=True):
        self.m, self.eta = m, eta
        self.lim_base = 2.0 * np.sqrt(max(T, 1e-7))     # ×√n/‖x_j*‖ = 2√(nT)/‖x_j*‖
        self.sig_passo = float(np.sqrt(2.0 * eta * max(T, 0.0)))  # √(2ηT)
        self.n_sigma = float(n_sigma)
        self.D = max(1, int(np.ceil(self.n_sigma ** 2)))  # dwell = ⌈n_σ²⌉
        self.c_p = float(c_p)
        self.usar_profundidade = bool(usar_profundidade)
        # estado por neurônio -----------------------------------------------
        self.aberto = np.zeros(m, bool)      # episódio-candidato aberto
        self.confirmado = np.zeros(m, bool)  # morte confirmada (dwell atingido)
        self.ini = np.full(m, -1)            # passo de início (t_morte provisório)
        self.run_abaixo = np.zeros(m, int)   # passos consecutivos < lim_morte
        self.run_acima = np.zeros(m, int)    # passos consecutivos > lim_vida (morto)
        self.revive_ini = np.full(m, -1)     # início da corrida de ressurreição
        self.lim_ini = np.zeros(m)           # lim_morte no início (p/ profundidade)
        # acumuladores do episódio (mesma semântica de v2) -------------------
        self.A_hat = np.zeros(m)
        self.t_morte = np.full(m, -1); self.t_area = np.full(m, -1)
        self.t_sinal = np.full(m, -1)
        self.v_morte = np.zeros(m); self.v_min_abs = np.full(m, np.inf)
        self.v_cruzou = np.zeros(m, bool); self.j_prev = np.full(m, -1)
        self.j_trocas = np.zeros(m, int)
        self.z_min = np.full(m, np.inf)      # menor margem-líder no episódio
        self.switches = []                   # (t, k, +1 ressurreição | -1 morte)
        self.episodios = []

    # a régua morta (v2) marcava/fechava dentro de observa; aqui `morto` é
    # exposto como os episódios CONFIRMADOS em aberto (p/ compat. com harness).
    @property
    def morto(self):
        return self.aberto & self.confirmado

    def _fecha(self, k, t_res):
        prof = bool(self.z_min[k] <= -self.c_p * self.lim_ini[k])
        self.episodios.append(dict(
            k=int(k), t_morte=int(self.t_morte[k]), t_area=int(self.t_area[k]),
            t_sinal=int(self.t_sinal[k]), t_res=int(t_res),
            v_morte=float(self.v_morte[k]), v_min_abs=float(self.v_min_abs[k]),
            v_cruzou=bool(self.v_cruzou[k]), j_trocas=int(self.j_trocas[k]),
            z_min=float(self.z_min[k]), profundo=prof, dwell=int(self.D)))

    def observa(self, t, z, a, res, v, x):
        """Estado CONSISTENTE pré-update. Mesmos argumentos de Regua.observa:
        z:(n,m) margens, a:(n,m)=act(z) da própria ativação, res:(n,)=f-y,
        v:(m,), x:(n,1)."""
        n = z.shape[0]
        zmax, jstar = z.max(0), z.argmax(0)
        normas = np.sqrt(x[:, 0] ** 2 + 1.0)
        norma_j = normas[jstar]                              # ‖(x_j*,1)‖ por neurônio
        lim_morte = self.lim_base * np.sqrt(n) / norma_j     # 2√(nT)/‖x_j*‖  (v2)
        sigma_z = self.sig_passo * norma_j                   # √(2ηT)·‖x_j*‖
        lim_vida = lim_morte + self.n_sigma * sigma_z        # histerese
        below = zmax < lim_morte
        above_vida = zmax > lim_vida
        idx = np.arange(self.m)
        R_hat = v * (-res[jstar] + v * a[jstar, idx])        # res=f-y ⟹ (y-f)=-res

        # (A) abre candidatos novos: neurônio vivo que caiu abaixo de lim_morte
        novos = (~self.aberto) & below
        if novos.any():
            self.aberto[novos] = True; self.confirmado[novos] = False
            self.ini[novos] = t; self.run_abaixo[novos] = 0
            self.run_acima[novos] = 0; self.revive_ini[novos] = -1
            self.lim_ini[novos] = lim_morte[novos]
            self.A_hat[novos] = 0.0
            self.t_morte[novos] = -1; self.t_area[novos] = -1
            self.t_sinal[novos] = -1
            self.v_morte[novos] = v[novos]
            self.v_min_abs[novos] = np.abs(v[novos])
            self.v_cruzou[novos] = False
            self.j_prev[novos] = jstar[novos]; self.j_trocas[novos] = 0
            self.z_min[novos] = zmax[novos]

        # (B) acumula para TODO episódio aberto (marca-depois-acumula, como v2)
        ab = self.aberto
        if ab.any():
            self.z_min[ab] = np.minimum(self.z_min[ab], zmax[ab])
            self.A_hat[ab] += R_hat[ab] * self.eta
            cruza = ab & (self.t_area < 0) & (self.A_hat > 0)
            self.t_area[cruza] = t
            sin = ab & (self.t_sinal < 0) & (R_hat > 0)
            self.t_sinal[sin] = t
            self.v_min_abs[ab] = np.minimum(self.v_min_abs[ab], np.abs(v[ab]))
            self.v_cruzou[ab] |= ((np.sign(v[ab]) != np.sign(self.v_morte[ab]))
                                  & (v[ab] != 0))
            troca_j = ab & (jstar != self.j_prev)
            self.j_trocas[troca_j] += 1
            self.j_prev[ab] = jstar[ab]

        # (C) dwell de MORTE: candidatos não confirmados
        cand = self.aberto & (~self.confirmado)
        aborta = cand & (~below)                 # subiu antes de confirmar ⟹ chatter
        if aborta.any():
            self.aberto[aborta] = False           # descarta (nenhum episódio)
        segue = cand & below
        self.run_abaixo[segue] += 1
        commit = segue & (self.run_abaixo >= self.D)
        if commit.any():
            self.confirmado[commit] = True
            self.t_morte[commit] = self.ini[commit]
            self.run_acima[commit] = 0
            for k in np.where(commit)[0]:
                self.switches.append((int(self.ini[k]), int(k), -1))

        # (D) dwell de RESSURREIÇÃO: morte confirmada, margem acima de lim_vida
        conf = self.aberto & self.confirmado
        subiu = conf & above_vida
        caiu = conf & (~above_vida)
        self.run_acima[caiu] = 0; self.revive_ini[caiu] = -1
        prim = subiu & (self.run_acima == 0)      # 1º passo da corrida de subida
        self.revive_ini[prim] = t
        self.run_acima[subiu] += 1
        rev = subiu & (self.run_acima >= self.D)
        if rev.any():
            for k in np.where(rev)[0]:
                self._fecha(k, int(self.revive_ini[k]))
                self.switches.append((int(self.revive_ini[k]), int(k), +1))
            self.aberto[rev] = False; self.confirmado[rev] = False

    def censurados(self, t_fim):
        """Episódios CONFIRMADOS ainda abertos no horizonte (censura à direita).
        Candidatos não confirmados (< D passos abaixo) NÃO são episódios — são
        indistinguíveis de chatter em curso; ficam de fora (documentado)."""
        out = []
        for k in np.where(self.aberto & self.confirmado)[0]:
            prof = bool(self.z_min[k] <= -self.c_p * self.lim_ini[k])
            out.append(dict(
                k=int(k), t_morte=int(self.t_morte[k]), t_area=int(self.t_area[k]),
                t_sinal=int(self.t_sinal[k]), t_res=-1,
                v_morte=float(self.v_morte[k]), v_min_abs=float(self.v_min_abs[k]),
                v_cruzou=bool(self.v_cruzou[k]), j_trocas=int(self.j_trocas[k]),
                z_min=float(self.z_min[k]), profundo=prof, dwell=int(self.D),
                censurado_em=int(t_fim)))
        return out

    def escalas(self, n, norma_j, T):
        """Escalas derivadas (documentação/teste): sem depender de dados."""
        lim_morte = 2.0 * np.sqrt(n * max(T, 1e-7)) / norma_j
        sigma_z = np.sqrt(2.0 * self.eta * max(T, 0.0)) * norma_j
        Delta = self.n_sigma * sigma_z
        return dict(lim_morte=lim_morte, sigma_z=sigma_z, banda=Delta,
                    lim_vida=lim_morte + Delta, c_h=1.0 + Delta / lim_morte,
                    D=self.D, prof_thr=-self.c_p * lim_morte)


def conta(ep, exigir_profundo=True):
    """Predicado de contagem: episódio COMPLETO (pós-debounce) e, se pedido,
    profundo. NÃO substitui episodio_limpo (que checa alheios/v/j) — compõe."""
    if ep.get("t_res", -1) < 0:
        return False
    return (not exigir_profundo) or ep.get("profundo", False)


# ===================== UNIT-TESTS SINTÉTICOS (margens fabricadas) ============
def _obs_seq(reg, margens, res=0.0, a=0.0, v=1.0, x=0.0):
    """Alimenta a régua com uma sequência de margens-líder (n=1 ponto). res/a/v
    podem ser escalares (constantes) ou listas por passo."""
    for t, zt in enumerate(margens):
        zz = np.array([[float(zt)]])                       # (n=1, m=1)
        rr = np.array([float(res[t] if isinstance(res, (list, tuple)) else res)])
        aa = np.array([[float(a[t] if isinstance(a, (list, tuple)) else a)]])
        vv = np.array([float(v[t] if isinstance(v, (list, tuple)) else v)])
        xx = np.array([[float(x)]])
        reg.observa(t, zz, aa, rr, vv, xx)


def _testes():
    # célula física limpa: n=1, x=0 ⟹ ‖(x,1)‖=1; T=0.25, η=0.02
    #   lim_morte = 2√(0.25) = 1.0 ; σ_z = √(2·0.02·0.25) = 0.1
    #   n_σ=3 ⟹ banda=0.3, lim_vida=1.3, D=9, c_p=1 ⟹ prof_thr=−1.0
    T, eta = 0.25, 0.02
    esc = ReguaV3(1, eta, T, n_sigma=3.0).escalas(1, 1.0, T)
    assert abs(esc["lim_morte"] - 1.0) < 1e-9, esc
    assert abs(esc["sigma_z"] - 0.1) < 1e-9, esc
    assert abs(esc["lim_vida"] - 1.3) < 1e-9, esc
    assert esc["D"] == 9, esc
    assert abs(esc["prof_thr"] + 1.0) < 1e-9, esc
    print(f"[OK] escalas derivadas: lim_morte=1.0 σ_z=0.1 banda=0.3 "
          f"lim_vida=1.3 c_h={esc['c_h']:.3f} D=9 prof_thr=−1.0")

    # T1) CHATTER PURO deve ser REJEITADO (0 episódios) --------------------
    # a margem cruza lim_morte=1.0 mas nunca sustenta D=9 passos abaixo, nunca
    # sobe além de 1.3 sustentado, nunca fica funda: passeio raso 0.8↔1.1.
    r = ReguaV3(1, eta, T, n_sigma=3.0)
    _obs_seq(r, [0.8, 1.1] * 60)
    assert len(r.episodios) == 0 and len(r.censurados(120)) == 0, \
        (len(r.episodios), r.switches)
    # variante: mergulho de 5 passos (<D=9) e volta — ainda chatter
    r2 = ReguaV3(1, eta, T, n_sigma=3.0)
    _obs_seq(r2, [2.0] * 3 + [-0.5] * 5 + [2.0] * 20)
    assert len(r2.episodios) == 0 and len(r2.censurados(28)) == 0, r2.switches
    print(f"[OK] T1 chatter puro rejeitado: 0 episódios em ambos "
          f"(passeio raso e mergulho de 5<D=9 passos)")

    # T2) MORTE PROFUNDA + RETORNO detectada com t_morte/t_res CORRETOS -----
    # vivo(z=2)×3 | fundo(z=−5)×50 | vivo(z=2)×15 ; z=−5 ≤ prof_thr=−1 ⟹ profundo
    r = ReguaV3(1, eta, T, n_sigma=3.0)
    _obs_seq(r, [2.0] * 3 + [-5.0] * 50 + [2.0] * 15)
    assert len(r.episodios) == 1, len(r.episodios)
    ep = r.episodios[0]
    assert ep["t_morte"] == 3, ep          # 1º passo abaixo (0-indexado)
    assert ep["t_res"] == 53, ep           # 1º passo da subida sustentada
    assert ep["profundo"] is True and ep["z_min"] <= -1.0, ep
    assert ep["dwell"] == 9, ep
    # sem falso-positivo de censura e switches confirmados (1 morte, 1 vida)
    assert len(r.censurados(68)) == 0
    assert sorted(s[2] for s in r.switches) == [-1, 1], r.switches
    print(f"[OK] T2 morte profunda+retorno: 1 episódio, t_morte=3 t_res=53 "
          f"profundo=True z_min={ep['z_min']:.1f}")

    # T2b) morte RASA sustentada (passa dwell/histerese mas NÃO profundidade)
    #   z=0.5 fica ABAIXO de lim_morte=1.0 mas ACIMA de prof_thr=−1.0 ⟹ raso.
    r = ReguaV3(1, eta, T, n_sigma=3.0)
    _obs_seq(r, [2.0] * 3 + [0.5] * 50 + [2.0] * 15)
    assert len(r.episodios) == 1, len(r.episodios)
    ep = r.episodios[0]
    assert ep["profundo"] is False, ep     # gravado, mas conta() o exclui
    assert conta(ep, exigir_profundo=True) is False
    assert conta(ep, exigir_profundo=False) is True
    print(f"[OK] T2b morte rasa sustentada: episódio GRAVADO com profundo=False "
          f"(z_min={ep['z_min']:.2f}); conta()=False sob profundidade")

    # T3) ÁREA Â_k e FLAGS v_k PRESERVADOS de v2 --------------------------
    # com n_σ=1 (D=1, banda=σ_z) e saltos limpos (fundo −5 ↔ alto +5, ambos
    # cruzam lim_morte E lim_vida no MESMO passo), v3 deve reproduzir v2 EXATO.
    margens = [3.0] * 4 + [-5.0] * 30 + [5.0] * 10
    # residual que faz a área cruzar 0 (R̂=v·(−res)=+0.5>0) e v que TROCA de
    # sinal no meio do episódio (testa v_cruzou e v_min_abs)
    passos = len(margens)
    res_seq = [-0.5] * passos                               # R̂ = v·(0.5)
    v_seq = [1.0] * 4 + [1.0] * 15 + [-0.3] * 15 + [1.0] * 10  # troca no meio
    r2 = Regua(1, eta, T)                                   # v2 de referência
    r3 = ReguaV3(1, eta, T, n_sigma=1.0)
    _obs_seq(r2, margens, res=res_seq, a=0.0, v=v_seq)
    _obs_seq(r3, margens, res=res_seq, a=0.0, v=v_seq)
    assert len(r2.episodios) == 1 and len(r3.episodios) == 1, \
        (len(r2.episodios), len(r3.episodios))
    e2, e3 = r2.episodios[0], r3.episodios[0]
    for chave in ("t_morte", "t_area", "t_sinal", "t_res", "v_morte",
                  "v_min_abs", "v_cruzou", "j_trocas"):
        assert e2[chave] == e3[chave], (chave, e2[chave], e3[chave])
    assert abs(r2.A_hat[0] - r3.A_hat[0]) < 1e-12, (r2.A_hat[0], r3.A_hat[0])
    assert e3["v_cruzou"] is True and e3["v_min_abs"] <= 0.3 + 1e-12, e3
    assert e3["t_area"] == e3["t_morte"], e3   # área cruza 0 já no 1º passo morto
    print(f"[OK] T3 paridade v2≡v3 (n_σ=1): t_morte={e3['t_morte']} "
          f"t_area={e3['t_area']} t_res={e3['t_res']} v_cruzou={e3['v_cruzou']} "
          f"Â={r3.A_hat[0]:.4f} (Δ vs v2 = {abs(r2.A_hat[0]-r3.A_hat[0]):.1e})")

    # T4) episodio_limpo (importado de v2) opera no dict de v3 sem adaptação
    assert episodio_limpo(e3, r3.switches, q=0, phi_v=0.1) in (True, False)
    print("[OK] T4 episodio_limpo de v2 aceita o episódio de v3 (dict compatível)")

    # T5) DWELL cresce com n_σ² e banda/c_h com n_σ (monotonicidade física)
    for ns in (3.0, 5.0, 10.0):
        e = ReguaV3(1, 3e-4, 1e-3, n_sigma=ns).escalas(32, np.sqrt(5.0), 1e-3)
        print(f"   n_σ={ns:4.1f}: D={e['D']:3d}  banda={e['banda']:.4f}  "
              f"lim_morte={e['lim_morte']:.4f}  c_h={e['c_h']:.3f}")
    assert ReguaV3(1, 3e-4, 1e-3, n_sigma=10.0).D == 100
    print("[OK] T5 D=⌈n_σ²⌉ e banda=n_σ·σ_z (célula MLP n=32,T=1e-3,η=3e-4)")

    print("\n== TODOS OS UNIT-TESTS DA RÉGUA v3 PASSARAM ==")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        _testes()
    else:
        _testes()
