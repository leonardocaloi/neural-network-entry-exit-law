from pathlib import Path
import json,sys,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent / '.'))
import verify_forcing as V
from multiprocessing import Pool
rs=[json.loads(l) for l in open(str(Path(__file__).resolve().parent / 'forcing_m.jsonl'))]
rs=[r for r in rs if r['passou'] and r['m']==8]
rs=sorted(rs,key=lambda r:-r['erro_rel'])[:6]
def job(a):
    r,b=a
    X=np.array(r['X']);Y=np.array(r['Y']);P0=np.array(r['P0'])
    tr,nm=V.roda_acoplado(X,Y,P0,b)
    if tr is None or nm!=1: return None
    return b*(tr-r['t_exit'])
if __name__=='__main__':
    with Pool(6) as p:
        for b in (1e4,3e4):
            out=p.map(job,[(r,b) for r in rs])
            for r,med in zip(rs,out):
                if med is None: print(f"  beta={b:.0e} seed {r['seed']}: sem release"); continue
                pr=r['predito']
                print(f"  beta={b:.0e} seed {r['seed']} R_exit={r['R_exit']:.4f} pred={pr:.2f} med={med:.2f} erro={abs(med-pr)/abs(pr)*100:.2f}%")
            print()
