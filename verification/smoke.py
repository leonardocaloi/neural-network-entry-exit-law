"""Check archived evidence without rerunning either training campaign.

Run from any directory. --engines adds short CPU training checks, not a
replication of the confirmatory experiments. No research outputs are written.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[1]
MLP = ROOT / "05-experimentos/etapa4-mlp-populacao"
TAXA = MLP / "pilotos-4a2/pressao/taxa"
TRANSFORMER = ROOT / "05-experimentos/etapa5-transformer"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_manifest(directory, manifest_name, count, config=None):
    manifest = json.loads((directory / manifest_name).read_text())
    records = {k: v for k, v in manifest.items() if not k.startswith("_")}
    assert len(records) == count, (directory.name, len(records))
    for name, expected in records.items():
        payload = (directory / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected, name
        json.loads(gzip.decompress(payload))
    if config is not None:
        assert hashlib.sha256(config.read_bytes()).hexdigest() == manifest["_config_sha256"]
    print(f"PASS: {directory.name}: {count} original files match SHA256 manifests")


def check_statistics():
    a4 = load_module("analysis_mlp", TAXA / "analise_4a3.py")
    a5 = load_module("analysis_transformer", TRANSFORMER / "analise_e5.py")
    cfg4 = json.loads((TAXA / "config_4a3.json").read_text())
    cfg5 = json.loads((TRANSFORMER / "config_e5.json").read_text())
    runs4 = a4.carrega(str(TAXA / "results-4a3"))
    runs5 = a5.carrega(str(TRANSFORMER / "results-e5"))
    grid4 = {(r["rot"], r["m"], r["seed"]) for r in runs4}
    expected4 = {(b["nome"], m, s) for b in cfg4["grade"]["bracos"]
                 for m in cfg4["grade"]["m"] for s in cfg4["grade"]["seeds"]}
    grid5 = {(r["arm"], r["fac"], r["depth"], r["seed"]) for r in runs5}
    g5 = cfg5["grade"]
    expected5 = {(b["nome"], f, d, s) for b in g5["bracos"] for f in g5["factors"]
                 for d in g5["inj_depths"] for s in g5["seeds"]}
    assert len(runs4) == len(grid4) == 400 and grid4 == expected4
    assert len(runs5) == len(grid5) == 800 and grid5 == expected5
    _, av4 = a4.relatorio(runs4, cfg4)
    _, av5 = a5.relatorio(runs5, cfg5)
    assert av4["validade"]["valido"] and av5["E_A"]["valido"]
    assert (av4["C2"]["n_ordenados"], av4["C2"]["n_voltas"]) == (13, 16)
    assert (av4["C1"]["voltas_elig"], av4["C1"]["N_elig"]) == (15, 30)
    assert (av5["E_C"]["sust"], av5["E_C"]["N_inj"]) == (22, 50)
    assert (av5["E_B"]["relu"]["n_sust_train"], av5["E_B"]["relu"]["n_sust_frozen"]) == (20, 0)
    cell = av5["tab_cell_depth"][("sp-b3", "frozen", -0.3)]
    assert (cell["n_area_ok"], cell["n_sust_trios"]) == (21, 22)
    assert av5["E_D"]["status"].startswith("CONSISTENTE")
    for mod in (a4, a5):
        for k, n in ((0, 100), (13, 16), (22, 50), (20, 100), (6, 6)):
            _, lo, hi = mod.clopper_pearson(k, n)
            ref_lo = 0.0 if k == 0 else beta_dist.ppf(0.025, k, n-k+1)
            ref_hi = 1.0 if k == n else beta_dist.ppf(0.975, k+1, n-k)
            assert np.allclose([lo, hi], [ref_lo, ref_hi], rtol=0, atol=1e-11)
    print("PASS: complete grids, main denominators, negative controls, and CP intervals")
    rows = [json.loads(x) for x in (ROOT / "05-experimentos/t2-sweep-instancias/sweep_instancias.jsonl").read_text().splitlines()]
    retained = [r for r in rows if r["passou_H1H2"]]
    assert len(rows) == 600 and len(retained) == 57
    assert all(r.get("erro_rel") is not None for r in retained)
    errors = np.array([r["erro_rel"] for r in retained])
    assert np.isclose(np.median(errors), 0.026764885048814316)
    print(f"PASS: sweep: 600 sampled, 57 retained; median error {100*np.median(errors):.4f}%")


def check_engines():
    import torch
    torch.set_num_threads(1)
    for directory in (MLP, TAXA.parent, TAXA, TRANSFORMER):
        sys.path.insert(0, str(directory))
    import motor_4a3
    import motor_e5_conf
    motor_4a3._paridade()  # Existing five-case, 400-step instrumentation check.
    kwargs = dict(passos=200, seed=42, ativ="softplus", beta=3., freeze_attn=True,
                  kill_depth=0.3, rec_cada=50)
    first = motor_e5_conf.treina_e5_conf(**kwargs)
    second = motor_e5_conf.treina_e5_conf(**kwargs)
    assert first == second
    null = motor_e5_conf.treina_e5_conf(**{**kwargs, "ativ": "relu"})
    injected = [u for u in null["unidades"] if u["injetada"]]
    assert len(injected) == 1 and null["n_inj_sustained"] == 0
    assert injected[0]["own_move"] == 0
    assert np.isfinite(first["loss_hist"]).all()
    print(f"PASS: short CPU engines, determinism and frozen-ReLU null (torch {torch.__version__})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", action="store_true")
    args = parser.parse_args()
    check_manifest(TAXA / "results-4a3", "manifest-4a3.json", 400, TAXA / "config_4a3.json")
    check_manifest(TRANSFORMER / "results-e5", "manifest-e5.json", 800, TRANSFORMER / "config_e5.json")
    check_manifest(MLP / "results", "manifest.json", 600)
    check_statistics()
    if args.engines:
        check_engines()
    print("All requested smoke checks passed. Full training campaigns were not rerun.")


if __name__ == "__main__":
    main()
