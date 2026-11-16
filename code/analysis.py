# analysis.py: H1 gap, probes, non-neural control, router, baselines. Input: out/ dir from the Kaggle run.
import json, sys, os, numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
D = sys.argv[1] if len(sys.argv) > 1 else "out"
HEAD = os.environ.get("HEAD", "rom_a")
rng = np.random.default_rng(1234)

pairs = [json.loads(l) for l in open(f"{D}/pairs.jsonl")]
P = {p["qid"]: p for p in pairs}
G = {}
for v in ["en", "dev", "rom_a", "rom_b", "noise3", "noise6", "selfrr"]:
    fp = f"{D}/gen_{v}.jsonl"
    if os.path.exists(fp): G[v] = {r["qid"]: r for r in map(json.loads, open(fp))}
RW = {}
if os.path.exists(f"{D}/gen_rewrite.jsonl"): RW = {r["qid"]: r for r in map(json.loads, open(f"{D}/gen_rewrite.jsonl"))}
_aq = set(json.load(open(f"{D}/acts_{HEAD}_qids.json"))) & set(json.load(open(f"{D}/acts_dev_qids.json")))
qids = [q for q in P if all(q in G[v] for v in G) and q in _aq]
grp = np.array([P[q]["group_id"] for q in qids]); ug = np.unique(grp)
lab = {v: np.array([G[v][q]["correct"] for q in qids], int) for v in G}
n = len(qids); res = {"n": n, "acc": {v: float(lab[v].mean()) for v in lab}}

def boot(fn, B=2000):
    idx = {g: np.where(grp == g)[0] for g in ug}; out = []
    for _ in range(B):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))]); out.append(fn(s))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
def mcnemar(a, b):
    bb = int(((a == 1) & (b == 0)).sum()); cc = int(((a == 0) & (b == 1)).sum())
    p = binomtest(bb, bb + cc, 0.5).pvalue if bb + cc else 1.0
    return {"b_dev_only": bb, "c_other_only": cc, "p": float(p)}
for v in lab: res.setdefault("ci", {})[v] = boot(lambda s, v=v: lab[v][s].mean())
res["mcnemar_dev_vs"] = {v: mcnemar(lab["dev"], lab[v]) for v in lab if v != "dev"}
ds = lambda s: lab["dev"][s].mean() - lab[HEAD][s].mean()
dn = lambda s: lab["dev"][s].mean() - (lab["noise3"][s].mean() + lab["noise6"][s].mean()) / 2
res["delta_script"] = float(ds(np.arange(n))); res["delta_noise"] = float(dn(np.arange(n)))
res["excess"] = float(ds(np.arange(n)) - dn(np.arange(n))); res["excess_ci"] = boot(lambda s: ds(s) - dn(s))
res["recoverable"] = int(((lab[HEAD] == 0) & (lab["dev"] == 1)).sum()); res["reverse"] = int(((lab[HEAD] == 1) & (lab["dev"] == 0)).sum())
res["known_set"] = int(lab["dev"].sum())
sig = res["mcnemar_dev_vs"][HEAD]["p"] < 0.05 and res["excess_ci"][0] > 0
res["H1_verdict"] = "ACCEPT" if sig else "NOT ACCEPTED"
# per type
res["by_qtype"] = {}
qt = np.array([P[q]["qtype"] for q in qids])
for t in np.unique(qt):
    m = qt == t; res["by_qtype"][t] = {"n": int(m.sum()), "dev": float(lab["dev"][m].mean()), HEAD: float(lab[HEAD][m].mean())}

# probes
meta = json.load(open(f"{D}/run_meta.json")); LAYERS = meta["layers_saved"]
def acts(view, L):
    q = json.load(open(f"{D}/acts_{view}_qids.json")); X = np.load(f"{D}/acts_{view}_L{L}.npy").astype(np.float32)
    ix = {x: i for i, x in enumerate(q)}; return X[[ix[x] for x in qids]]
# splits by group
ugs = rng.permutation(ug); sp = {g: ("train" if i < .6 * len(ugs) else "val" if i < .8 * len(ugs) else "test") for i, g in enumerate(ugs)}
split = np.array([sp[g] for g in grp]); tr, va, te = [split == s for s in ("train", "val", "test")]
def fit(X, y, m):
    sc = StandardScaler().fit(X[m]); best = None
    for C in (0.001, 0.01, 0.1):
        cv = []
        gk = GroupKFold(3)
        for a, b in gk.split(X[m], y[m], grp[m]):
            if len(np.unique(y[m][a])) < 2 or len(np.unique(y[m][b])) < 2: continue
            c = LogisticRegression(C=C, max_iter=300).fit(sc.transform(X[m][a]), y[m][a]); cv.append(roc_auc_score(y[m][b], c.predict_proba(sc.transform(X[m][b]))[:, 1]))
        s = np.mean(cv) if cv else 0
        if best is None or s > best[0]: best = (s, C)
    return sc, LogisticRegression(C=best[1], max_iter=500).fit(sc.transform(X[m]), y[m])
def auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
res["probe_layers"] = {}
yr, yd = lab[HEAD], lab["dev"]; yrr = lab["selfrr"] if "selfrr" in lab else yd; bestL = {}
for L in LAYERS:
    X = acts(HEAD, L); r = {}
    for nm, y in (("C_rom", yr), ("C_dev", yd)):
        sc, c = fit(X, y, tr); r[nm] = {"val": auc(y[va], c.predict_proba(sc.transform(X[va]))[:, 1])}
    res["probe_layers"][L] = r
for nm in ("C_rom", "C_dev"): bestL[nm] = max(LAYERS, key=lambda L: np.nan_to_num(res["probe_layers"][L][nm]["val"]))
res["best_layer"] = bestL
S = {}
for nm, y in (("C_rom", yr), ("C_dev", yd)):
    X = acts(HEAD, bestL[nm]); sc, c = fit(X, y, tr); s = c.predict_proba(sc.transform(X))[:, 1]; S[nm] = s
    res.setdefault("probe_test", {})[nm] = {"auc": auc(y[te], s[te]),
        "auc_ci": [float(x) for x in np.percentile([auc(y[te][i], s[te][i]) for i in [rng.integers(0, te.sum(), te.sum()) for _ in range(300)]], [2.5, 97.5])]}
# H2c: among rom-wrong, can s_dev predict dev-correct
w = (yr == 0) & te
res["H2c_auc"] = auc(yd[w], S["C_dev"][w]); res["H2c_n"] = {"pos": int(yd[w].sum()), "neg": int((1 - yd[w]).sum())}
# non-neural control
from sklearn.ensemble import GradientBoostingClassifier
def feats(q):
    p = P[q]; r = G[HEAD][q]
    return [["generic","intersection","count","comparative","yesno","ordinal","multihop","difference","superlative"].index(p["qtype"]) if p["qtype"] in ["generic","intersection","count","comparative","yesno","ordinal","multihop","difference","superlative"] else 9,
            ["entity","numerical","boolean","date","string"].index(p["answer_type"]), len(p["en"]), len(p[HEAD]), len(p[HEAD].split()), r["prompt_tokens"] / max(1, len(p[HEAD].split())), p["n_ents"]]
F = np.array([feats(q) for q in qids])
ctrl = {}
for nm, y in (("C_rom", yr), ("C_dev", yd)):
    g = GradientBoostingClassifier(random_state=1234, n_estimators=100, max_depth=3).fit(F[tr], y[tr])
    ctrl[nm] = g.predict_proba(F)[:, 1]; res["probe_test"][nm]["control_auc"] = auc(y[te], ctrl[nm][te])
# output-side signals
ent = np.array([G[HEAD][q]["first_token_entropy"] for q in qids]); mlp = np.array([G[HEAD][q]["mean_logprob"] for q in qids])
res["probe_test"]["C_rom"]["entropy_auc"] = auc(yr[te], -ent[te]); res["probe_test"]["C_rom"]["mean_logprob_auc"] = auc(yr[te], mlp[te])

# router and baselines on validation->test. delivered answer correctness:
lens = lambda v: np.array([G[v][q]["prompt_tokens"] + G[v][q]["gen_tokens"] for q in qids])
cost_rom = lens(HEAD); cost_dev = lens("dev")
cost_rr = (lens("selfrr") + np.array([RW[q]["prompt_tokens"] + RW[q]["gen_tokens"] if q in RW else 80 for q in qids])) if "selfrr" in G else cost_dev + 40
def run(policy_scores, t, action):  # generic
    pass
def eval_policy(act, ans_correct, cost, m):
    deliv = act != 0
    return {"coverage": float(deliv[m].mean()), "right": float((ans_correct * deliv)[m].mean()), "wrong": float(((1 - ans_correct) * deliv)[m].mean()),
            "sel_acc": float(ans_correct[deliv & m].mean()) if (deliv & m).any() else float("nan"), "tokens": float(cost[m].mean())}
def grid_router(m, lam):
    best = None
    for t1 in np.quantile(S["C_rom"][m], np.linspace(0, 1, 21)):
        for t2 in np.quantile(S["C_dev"][m], np.linspace(0, 1, 21)):
            a = np.where(S["C_rom"] >= t1, 1, np.where(S["C_dev"] >= t2, 2, 0))
            c = np.where(a == 1, yr, np.where(a == 2, yrr, 0))  # RE_RENDER uses Tiny Aya self-rewrite answer
            u = ((c * (a != 0)) - lam * ((1 - c) * (a != 0)))[m].mean()
            if best is None or u > best[0]: best = (u, t1, t2)
    return best
res["policy_test"] = {}
for lam in (0.5, 1, 2, 4):
    u, t1, t2 = grid_router(va, lam)
    a = np.where(S["C_rom"] >= t1, 1, np.where(S["C_dev"] >= t2, 2, 0)); c = np.where(a == 1, yr, np.where(a == 2, yrr, 0))
    cost = np.where(a == 2, cost_rom + cost_rr, cost_rom)  # self-rewrite approx: extra ~ prompt+rewrite tokens
    r = eval_policy(a, c, cost, te); cov = r["coverage"]
    r.update({"lambda": lam, "t1": float(t1), "t2": float(t2), "rerender_frac": float((a == 2)[te].mean())})
    r["utility"] = r["right"] - lam * r["wrong"]; res["policy_test"][f"router_lam{lam}"] = r
    # baselines at matched coverage on test (threshold set on val quantile)
    def abst(score, cov_target):
        thr = np.quantile(score[va], 1 - cov_target); return (score >= thr).astype(int)
    for nm, sc_ in (("always_answer+probe", S["C_rom"]), ("always_answer+entropy", -ent), ("always_answer+meanlogprob", mlp), ("always_answer+nonneural", ctrl["C_rom"])):
        aa = abst(sc_, cov); e = eval_policy(aa, yr, cost_rom, te); e["utility"] = e["right"] - lam * e["wrong"]; res["policy_test"][f"{nm}_lam{lam}"] = e
    for nm, sc_ in (("always_normalise+probe", S["C_dev"]), ("always_normalise+nonneural", ctrl["C_dev"])):
        aa = abst(sc_, cov); e = eval_policy(aa, yrr, cost_rom + cost_rr, te); e["utility"] = e["right"] - lam * e["wrong"]; res["policy_test"][f"{nm}_lam{lam}"] = e
    # random routing at same rerender budget
    rr = np.random.default_rng(1).random(n) < (a == 2)[te].mean()
    ar = np.where(abst(S["C_rom"], cov) == 1, np.where(rr, 2, 1), 0); cr = np.where(ar == 1, yr, np.where(ar == 2, yrr, 0))
    e = eval_policy(ar, cr, cost, te); e["utility"] = e["right"] - lam * e["wrong"]; res["policy_test"][f"random_route_lam{lam}"] = e
    o = np.where(yr == 1, 1, np.where(yrr == 1, 2, 0)); co = np.where(o == 1, yr, np.where(o == 2, yrr, 0))
    e = eval_policy(o, co, cost, te); e["utility"] = e["right"] - lam * e["wrong"]; res["policy_test"][f"oracle_lam{lam}"] = e
a0 = np.ones(n, int); e = eval_policy(a0, yr, cost_rom, te); res["policy_test"]["always_answer_nogate"] = e
a1 = np.where(np.ones(n) > 0, 2, 0); e = eval_policy(a1, yrr, cost_rom + cost_rr, te); res["policy_test"]["always_normalise_nogate"] = e
res["splits"] = {s: int(m.sum()) for s, m in (("train", tr), ("val", va), ("test", te))}
os.makedirs("results", exist_ok=True); json.dump(res, open(f"results/{os.path.basename(D.rstrip('/'))}_analysis.json", "w"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ["n", "acc", "delta_script", "delta_noise", "excess", "excess_ci", "recoverable", "reverse", "known_set", "H1_verdict", "mcnemar_dev_vs", "best_layer", "probe_test", "H2c_auc", "H2c_n", "splits"]}, indent=1, default=float))
