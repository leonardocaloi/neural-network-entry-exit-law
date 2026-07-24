"""Congela config_v2.json a partir do preditor-output.json + parametros de analise.
Rodar UMA vez no fechamento do pre-registro; o commit do resultado e o pre-registro.
"""
import json

with open("preditor-output.json") as fh:
    pred = json.load(fh)

config = {
    "constantes": {
        **pred["constantes"],
        "ETA_B": 1e-4,
        "N_SEEDS_CELULA": 150,             # subido de 50 pelo parecer (potencia P-2)
        "W_P3": 50_000,
        "W_LOCAL": 70_000,                 # horizonte de busca de captura, ambos os lados (M1/M4)
        # P-1 (pre-registrado): >=3 rejeicoes em celulas VALIDAS (binomial bicaudal,
        # alpha=0.05, contra p_prevista) OU >=8 desvios de mesmo sinal; <7 validas = inconclusivo
        "P1_MIN_REJEICOES": 3,
        "P1_MIN_MESMO_SINAL": 8,
        # P-2 (redesenho pos-parecer): calibracao da curva inteira pooled (bins por
        # q predito) + falsificador exato de captura tardia (modelo preve ZERO)
        "P2_Q_BINS": [0.0, 0.1, 0.5, 0.9, 1.0],
        "P2_MIN_BINS_REJEITADOS": 2,
        "P2_MIN_CAPTURAS_TARDIAS": 2,      # >=2 p/ robustez a edge-case de implementacao
        "MAX_FRAC_CENSURA": 0.30,
    },
    "celulas": pred["celulas"],
    "pooled_por_T": pred["pooled_por_T"],
}
with open("config_v2.json", "w") as fh:
    json.dump(config, fh, indent=1)
print(f"config_v2.json congelada: {len(config['celulas'])} celulas, "
      f"{len(config['pooled_por_T'])} curvas pooled")
for c in config["celulas"]:
    print(f"  T={c['T']} d0={c['d0']:.3f} -> p_prevista={c['p_captura']:.3f} "
      f"(censura {c['censura']:.2f})")
