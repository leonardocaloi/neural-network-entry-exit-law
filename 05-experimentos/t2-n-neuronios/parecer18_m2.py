import sys,os,numpy as np,time
sys.path.insert(0,'/home/caloi/Projects/j-space/05-experimentos/t2-n-neuronios')
os.chdir('/home/caloi/Projects/j-space/05-experimentos/t2-n-neuronios')
from multiprocessing import Pool
from verifica_forcing import sorteia
from degrau2_readout import hibrido
SEED0=31410001
def one(a):
    m,seed=a
    try: Xs,Ys,P0=sorteia(seed,m)
    except RuntimeError: return None
    P=np.zeros((m+1,3)); P[:,:2]=P0; P[0,2]=1.0; P[1:,2]=1.0/np.sqrt(m)
    h=hibrido(Xs,Ys,P,2e-4,medir=True)
    if not h["ok"]: return None
    return (m,h["Lambda_ck"],h["L_B"],h["T_ep"])
def main():
    t0=time.time()
    with Pool(2) as p:
        for m in (4,8,16):
            res=[r for r in p.map(one,[(m,SEED0+i) for i in range(30)],chunksize=2) if r][:10]
            if not res: print("m=%d nada"%m,flush=True); continue
            L=np.array([r[1] for r in res]); T=np.array([r[3] for r in res])
            print("m=%2d N=%2d  Lambda med %.2f range [%.2f,%.2f] | T_ep med %.4f | e^{LamTep} med %.3f (%.1f min)"
                  %(m,len(res),np.median(L),L.min(),L.max(),np.median(T),np.median(np.exp(L*T)),(time.time()-t0)/60),flush=True)
if __name__=='__main__': main()
