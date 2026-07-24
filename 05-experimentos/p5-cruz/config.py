# Configuração do experimento p5-cruz. Definida ANTES de rodar (2026-07-23).
# Qualquer mudança pós-primeira-execução deve ser registrada no README (seção "desvios").

SEEDS = list(range(20))          # 20 réplicas independentes
ETA = 1e-3                        # passo
TEMPERATURAS = [1e-4, 1e-3, 1e-2] # T do SGLD (P-c prevê deriva crescente com T)
N_PASSOS = 200_000
INIT_BOX = 2.0                    # w0 ~ U([-2,2]²), rejeitando |wi|<0.3 (longe da cruz)
INIT_MIN = 0.3
DELTA_CONTATO = 0.05              # d < δ conta como "em contato"
# Janelas de medição (fixadas a priori):
JANELA_POUSO = (0, 50_000)        # onde se mede tempo-até-contato (P-a)
JANELA_DERIVA = (50_000, 200_000) # onde se mede d<w1²>/dt em contato (P-c)
BINS_OCUPACAO = 60                # heatmap (d, r_N) em escala log (P-b)
