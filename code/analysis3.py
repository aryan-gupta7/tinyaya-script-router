# analysis3.py: gap robustness, retention, grouped out-of-fold probes vs entropy, risk-coverage. HEAD view via env (rom_a or rom_a2).
import json, sys, os, numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
D = sys.argv[1]; OUTJ = sys.argv[2]; H = os.environ.get("HEAD", "rom_a2"); LAY = [int(x) for x in os.environ.get("LAYERS", "19,31").split(",")]
rng = np.random.default_rng(1234)
P = {p["qid"]: p for p in map(json.loads, open(f"{D}/pairs.jsonl"))}
G = {v: {r["qid"]: r for r in map(json.loads, open(f"{D}/gen_{v}.jsonl"))} for v in ("dev", H, "noise3", "noise6")}
qids = [q for q in P if all(q in G[v] for v in G)]
n = len(qids); grp = np.array([str(P[q]["group_id"]) for q in qids]); ug = np.unique(grp)
lab = {v: np.array([G[v][q]["correct"] for q in qids], int) for v in G}
isb = np.array([P[q]["answer_type"] == "boolean" for q in qids]); yn = np.array([P[q]["qtype"] == "yesno" for q in qids])
def cont(v):  # looser containment scorer
    o = []
    for q in qids:
        out = C.norm(G[v][q]["output"].split("\n")[0]); ok = False
        for a in P[q]["gold_aliases"]:
            g = C.norm(a)
            if g and (out == g or (len(g) >= 3 and (" " + g + " ") in (" " + out + " ")) or C.score(G[v][q]["output"], [a], P[q]["answer_type"])[0]): ok = True; break
        o.append(ok)
    return np.array(o, int)
lc = {v: cont(v) for v in ("dev", H)}
no_dev = np.array([not any("ऀ" <= ch <= "ॿ" for ch in P[q].get(H, "")) and "{" not in P[q].get(H, "") for q in qids])
idx_g = {g: np.where(grp == g)[0] for g in ug}
def gboot(f, B=2000):
    v = []
    for _ in range(B):
        s = np.concatenate([idx_g[g] for g in rng.choice(ug, len(ug))]); v.append(f(s))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
def mc(a, b, m):
    bb = int(((a == 1) & (b == 0) & m).sum()); cc = int(((a == 0) & (b == 1) & m).sum())
    return bb, cc, float(binomtest(bb, bb + cc, .5).pvalue) if bb + cc else 1.0
out = {"head_view": H, "n": n, "groups": int(len(ug)), "gap": {}}
subs = {"all": np.ones(n, bool), "non_boolean": ~isb, "non_yesno_qtype": ~yn, "no_corruption": no_dev, "no_corruption_non_boolean": no_dev & ~isb}
for nm, m in subs.items():
    a, b = lab["dev"], lab[H]; bb, cc, p = mc(a, b, m)
    out["gap"][nm] = {"n": int(m.sum()), "acc_dev": float(a[m].mean()), "acc_head": float(b[m].mean()), "dev_only": bb, "head_only": cc, "p": p,
                      "gap_ci_group_boot": gboot(lambda s, m=m: (a[s][m[s]].mean() - b[s][m[s]].mean()) if m[s].any() else np.nan)}
    # retention (non-boolean emphasised)
    keep = ((a == 1) & (b == 1) & m).sum() / max(1, ((a == 1) & m).sum()); back = ((a == 1) & (b == 1) & m).sum() / max(1, ((b == 1) & m).sum())
    out["gap"][nm].update({"romanized_keeps_of_dev_correct": float(keep), "dev_keeps_of_roman_correct": float(back)})
a, b = lc["dev"], lc[H]; bb, cc, p = mc(a, b, subs["all"]); out["gap"]["containment_scorer_all"] = {"acc_dev": float(a.mean()), "acc_head": float(b.mean()), "dev_only": bb, "head_only": cc, "p": p}
# excess over noise with group-clustered bootstrap
d_s = lambda s: lab["dev"][s].mean() - lab[H][s].mean(); d_n = lambda s: lab["dev"][s].mean() - (lab["noise3"][s].mean() + lab["noise6"][s].mean()) / 2
out["excess_over_noise"] = {"value": float(d_s(np.arange(n)) - d_n(np.arange(n))), "ci_group_boot": gboot(lambda s: d_s(s) - d_n(s)),
    "acc_noise3": float(lab["noise3"].mean()), "acc_noise6": float(lab["noise6"].mean()), "note": "noise3/noise6 edits may be nested, treat as one control family"}
# out-of-fold grouped probes vs entropy
y = lab[H]; ent = np.array([G[H][q]["first_token_entropy"] for q in qids]); mlp = np.array([G[H][q]["mean_logprob"] for q in qids])
qa = json.load(open(f"{D}/acts_{H}_qids.json")); ix = {q: i for i, q in enumerate(qa)}; sel = [ix[q] for q in qids]
def auc(y_, s): return float(roc_auc_score(y_, s)) if len(np.unique(y_)) > 1 else float("nan")
def aurc(y_, s):
    o = np.argsort(-s, kind="stable"); e = 1 - y_[o]; return float(np.mean(np.cumsum(e) / np.arange(1, len(e) + 1)))
oof = {}; gkf = GroupKFold(5)
for L in LAY:
    X = np.load(f"{D}/acts_{H}_L{L}.npy").astype(np.float32)[sel]; s = np.zeros(n)
    for tr, te in gkf.split(X, y, grp):
        sc = StandardScaler().fit(X[tr]); s[te] = LogisticRegression(C=0.001, max_iter=500).fit(sc.transform(X[tr]), y[tr]).decision_function(sc.transform(X[te]))
    oof[L] = s
sig = {f"probe_L{L}": oof[L] for L in LAY}; sig.update({"entropy": -ent, "mean_logprob": mlp})
# stacked probe + entropy (OOF stack by a second grouped CV)
L0 = LAY[-1]; Z = np.c_[oof[L0], ent, mlp]; st = np.zeros(n)
for tr, te in gkf.split(Z, y, grp):
    sc = StandardScaler().fit(Z[tr]); st[te] = LogisticRegression(C=1.0).fit(sc.transform(Z[tr]), y[tr]).decision_function(sc.transform(Z[te]))
sig["probe+entropy+logprob"] = st
Fn = np.c_[[G[H][q]["prompt_tokens"] for q in qids], [len(P[q][H].split()) for q in qids], [P[q]["n_ents"] for q in qids], isb, [len(P[q]["en"]) for q in qids]]
sn = np.zeros(n)
for tr, te in gkf.split(Fn, y, grp):
    sc = StandardScaler().fit(Fn[tr]); sn[te] = LogisticRegression(C=1.0).fit(sc.transform(Fn[tr]), y[tr]).decision_function(sc.transform(Fn[te]))
sig["non-neural control (length, entities, boolean)"] = sn
out["oof_selective"] = {}
for sub_name, m in (("all", subs["all"]), ("non_boolean", ~isb)):
    t = {}
    for nm, s in sig.items():
        t[nm] = {"auroc": auc(y[m], s[m]), "aurc": aurc(y[m], s[m]),
                 "acc_at_cov": {str(c): float(y[m][np.argsort(-s[m], kind="stable")[:max(1, int(c * m.sum()))]].mean()) for c in (0.1, 0.2, 0.3, 0.5, 0.7)}}
    t["no abstention"] = {"aurc": float(1 - y[m].mean())}
    out["oof_selective"][sub_name] = t
    bp = lambda s_, f, m_=m: f(y[s_][m_[s_]], sig[f"probe_L{L0}"][s_][m_[s_]]) - f(y[s_][m_[s_]], sig["entropy"][s_][m_[s_]]) if m_[s_].any() and len(np.unique(y[s_][m_[s_]])) > 1 else np.nan
    out["oof_selective"][sub_name]["probe_minus_entropy_auroc_ci"] = gboot(lambda s_: bp(s_, auc), 500)
    out["oof_selective"][sub_name]["probe_minus_entropy_auroc"] = float(auc(y[m], sig[f"probe_L{L0}"][m]) - auc(y[m], sig["entropy"][m]))
# recoverability (H2c) out-of-fold: among head-wrong, predict dev-correct from rom prompt
yd = lab["dev"]; w = lab[H] == 0
X = np.load(f"{D}/acts_{H}_L{L0}.npy").astype(np.float32)[sel]; sd = np.full(n, np.nan)
for tr, te in gkf.split(X, yd, grp):
    sc = StandardScaler().fit(X[tr][w[tr]]); sd[te] = LogisticRegression(C=0.001, max_iter=500).fit(sc.transform(X[tr][w[tr]]), yd[tr][w[tr]]).decision_function(sc.transform(X[te]))
out["h2c_oof"] = {"n_wrong_head": int(w.sum()), "positives": int(yd[w].sum()), "auroc_probe": auc(yd[w], sd[w]), "auroc_entropy": auc(yd[w], -ent[w]),
                  "auroc_ci_probe": gboot(lambda s_: auc(yd[s_][w[s_]], sd[s_][w[s_]]) if w[s_].any() and len(np.unique(yd[s_][w[s_]])) > 1 else np.nan, 300)}
os.makedirs(os.path.dirname(OUTJ) or ".", exist_ok=True); json.dump(out, open(OUTJ, "w"), indent=1, default=float)
for k, v in out["gap"].items(): print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "gap_ci_group_boot"}, [round(x, 3) for x in v.get("gap_ci_group_boot", [])])
print("excess", out["excess_over_noise"])
for sn_, t in out["oof_selective"].items():
    print("==", sn_); [print(f"  {k:48s}", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a in ("auroc", "aurc", "probe_minus_entropy_auroc", "probe_minus_entropy_auroc_ci")}) for k, v in t.items() if isinstance(v, dict)]
print("h2c", out["h2c_oof"])
