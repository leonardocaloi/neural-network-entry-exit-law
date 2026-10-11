import json,numpy as np
R=[json.loads(l) for l in open('readout_scale.jsonl')]
ok=[r for r in R if r.get('passou')]
print("total",len(R),"ok",len(ok))
mot={}
for r in R:
    if not r.get('passou'): mot[r['motivo']]=mot.get(r['motivo'],0)+1
print(mot)
for v0 in (0.5,2.0):
    o=[r for r in ok if r['v0_init']==v0]
    med=np.array([r['medido'] for r in o]); pn=np.array([r['pred_naive'] for r in o]); pc=np.array([r['pred_corr'] for r in o])
    vh=np.array([r['v0_hit'] for r in o])
    print(f"\nv0={v0} N={len(o)}  |v0_hit-v0|max={np.abs(vh-v0).max():.2e}")
    print("  mu/naive med %.4f range [%.3f,%.3f]"%(np.median(med/pn),(med/pn).min(),(med/pn).max()))
    print("  mu/corr  med %.4f range [%.3f,%.3f]"%(np.median(med/pc),(med/pc).min(),(med/pc).max()))
    en=np.array([r['err_naive'] for r in o]); ec=np.array([r['err_corr'] for r in o])
    print("  err_corr med %.2f%% p90 %.2f%% max %.2f%%"%(np.median(ec)*100,np.percentile(ec,90)*100,ec.max()*100))
    print("  err_naive med %.2f%%"%(np.median(en)*100))
    print("  paired wins %d/%d"%(int((ec<en).sum()),len(o)))
en=np.array([r['err_naive'] for r in ok]); ec=np.array([r['err_corr'] for r in ok])
print("\npooled N=%d  err_naive med %.2f%%  err_corr med %.2f%% p90 %.2f%% max %.2f%%"%(len(ok),np.median(en)*100,np.median(ec)*100,np.percentile(ec,90)*100,ec.max()*100))
print("paired wins %d/%d  median ratio %.4f"%(int((ec<en).sum()),len(ok),np.median(ec/en)))
# sign test
from math import comb
k=int((ec<en).sum()); n=len(ok)
p=sum(comb(n,j) for j in range(k,n+1))/2**n
print("one-sided sign test p = %.3g"%p)
LB=np.array([r['L_B'] for r in ok]); Lam=np.array([r['Lambda_ck'] for r in ok]); Tep=np.array([r['T_ep'] for r in ok])
print("max|L_B-Lambda| = %.2e  rel %.2e"%(np.abs(LB-Lam).max(),(np.abs(LB-Lam)/Lam).max()))
print("Lambda [%.2f,%.2f] med %.2f ; Tep [%.4f,%.3f] med %.4f"%(Lam.min(),Lam.max(),np.median(Lam),Tep.min(),Tep.max(),np.median(Tep)))
g=np.exp(Lam*Tep); print("e^{Lam Tep} [%.3f,%.3f] med %.3f"%(g.min(),g.max(),np.median(g)))
print("corr(log err_corr, log gronwall) = %.3f"%np.corrcoef(np.log(ec),np.log(g))[0,1])
Re=np.array([r['R_exit'] for r in ok])
print("corr(log err_corr, log 1/|R_exit|) = %.3f"%np.corrcoef(np.log(ec),np.log(1/np.abs(Re)))[0,1])
vm=np.array([r['v_min_live'] for r in ok]); print("phi_v min %.3e median %.3f"%(vm.min(),np.median(vm)))
