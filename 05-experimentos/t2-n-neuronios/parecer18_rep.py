import sys, os, json, time
import numpy as np
sys.path.insert(0,'/home/caloi/Projects/j-space/05-experimentos/t2-n-neuronios')
os.chdir('/home/caloi/Projects/j-space/05-experimentos/t2-n-neuronios')
from multiprocessing import Pool
import degrau2_v0 as D
D.SEED0 = 77770101          # SEEDS FRESCAS, independentes do run oficial
D.N_MAX_FULL = 8
n_inst = 220
def main():
    t0=time.time(); todos=[]
    with Pool(2) as pool:
        for v0 in D.V0_NIVEIS:
            cand = pool.map(D.peneira, [(D.SEED0+i, v0) for i in range(n_inst)], chunksize=8)
            tar = [(s,v0,D.DT0) for s in cand if s is not None][:D.N_MAX_FULL]
            print(f"[v0={v0}] {len(tar)} candidatas ({(time.time()-t0)/60:.1f} min)",flush=True)
            todos += pool.map(D.uma, tar, chunksize=1)
    ok=[r for r in todos if r.get('passou')]
    print(f"\n=== REPLICA INDEPENDENTE  ok={len(ok)}/{len(todos)}  ({(time.time()-t0)/60:.1f} min) ===")
    for v0 in D.V0_NIVEIS:
        o=[r for r in ok if r['v0_init']==v0]
        if not o: continue
        med=np.array([r['medido'] for r in o]); pn=np.array([r['pred_naive'] for r in o]); pc=np.array([r['pred_corr'] for r in o])
        en=np.array([r['err_naive'] for r in o]); ec=np.array([r['err_corr'] for r in o])
        vh=np.array([r['v0_hit'] for r in o]); LB=np.array([r['L_B'] for r in o]); La=np.array([r['Lambda_ck'] for r in o])
        print(f"v0={v0} N={len(o)}: mu/naive med {np.median(med/pn):.4f} | mu/corr med {np.median(med/pc):.4f}")
        print(f"   err_naive med {np.median(en)*100:.2f}%  err_corr med {np.median(ec)*100:.2f}%  wins {(ec<en).sum()}/{len(o)}")
        print(f"   |v0_hit-v0|max {np.abs(vh-v0).max():.2e}   max|L_B-Lambda| {np.abs(LB-La).max():.2e}")
    

if __name__=='__main__':
    main()
