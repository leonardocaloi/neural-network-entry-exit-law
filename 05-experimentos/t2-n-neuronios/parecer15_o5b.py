import sys, json, numpy as np
sys.path.insert(0,"/tmp/claude-1000/-home-caloi-Projects-j-space/552bb7fa-f3f0-42f8-8086-cb79faf68c7a/scratchpad")
sys.path.insert(0,"/home/caloi/Projects/j-space/05-experimentos/t2-n-neuronios")
from parecer15_o5 import run
from verifica_forcing import sorteia, FACE
recs=[json.loads(l) for l in open("/home/caloi/Projects/j-space/05-experimentos/t2-n-neuronios/forcing_m.jsonl")]
ok=[r for r in recs if r["passou"]]
print(" m  seed        v       t_hit    b2*rhoB   b2*rhoB*v^2  b*rhoA*v")
rows=[]
for m in (4,8,16):
    for r in [x for x in ok if x["m"]==m][:4]:
        o=run(r["seed"],m,[8e3])
        if o is None: continue
        c=r["c"]; v=c*abs(r["R_hit"])
        b,rB,rA=o["rows"][0]
        print(f"{m:3d} {r['seed']} {v:8.4f} {o['t_hit']:.4f}  {rB*b*b:8.3f}  {rB*b*b*v*v:10.4f}  {rA*b*v:8.4f}")
