# analysis4.py <dir> <out.json>: gap by gold-answer type, group-level sign test, noise-control range, rewrite significance.
import json, sys, os, numpy as np
from scipy.stats import binomtest
D = sys.argv[1]; OUT = sys.argv[2]; H = "rom_a2"
P = {p["qid"]: p for p in map(json.loads, open(f"{D}/pairs.jsonl"))}
G = {v: {r["qid"]: r for r in map(json.loads, open(f"{D}/gen_{v}.jsonl"))} for v in ("dev", H, "noise3", "noise6", "selfrr")}
qids = [q for q in P if all(q in G[v] for v in G)]
L = {v: np.array([G[v][q]["correct"] for q in qids], int) for v in G}
at = np.array([P[q]["answer_type"] for q in qids]); grp = np.array([str(P[q]["group_id"]) for q in qids])
def mc(a, b, m):
    bb = int(((a == 1) & (b == 0) & m).sum()); cc = int(((a == 0) & (b == 1) & m).sum())
    return bb, cc, (float(binomtest(bb, bb + cc, .5).pvalue) if bb + cc else 1.0)
o = {"by_answer_type": {}, "group_level_sign_test": {}, "noise_control": {}, "selfrewrite_vs_roman": {}}
for t in ("entity", "numerical", "date", "boolean", "string"):
    m = at == t
    if m.sum() < 10: continue
    bb, cc, p = mc(L["dev"], L[H], m)
    o["by_answer_type"][t] = {"n": int(m.sum()), "acc_dev": float(L["dev"][m].mean()), "acc_roman": float(L[H][m].mean()), "dev_only": bb, "roman_only": cc, "p": p,
                              "roman_keeps_of_dev_correct": float(((L["dev"] == 1) & (L[H] == 1) & m).sum() / max(1, ((L["dev"] == 1) & m).sum()))}
for nm, m in (("all", np.ones(len(qids), bool)), ("non_boolean", at != "boolean")):
    pos = neg = 0
    for g in np.unique(grp):
        i = (grp == g) & m
        if not i.any(): continue
        b = ((L["dev"] == 1) & (L[H] == 0) & i).sum(); c = ((L["dev"] == 0) & (L[H] == 1) & i).sum()
        if b > c: pos += 1
        elif c > b: neg += 1
    o["group_level_sign_test"][nm] = {"groups_dev_better": pos, "groups_roman_better": neg, "p": float(binomtest(pos, pos + neg, .5).pvalue) if pos + neg else 1.0}
gap = L["dev"].mean() - L[H].mean()
for n_ in ("noise3", "noise6"):
    o["noise_control"][n_] = {"acc": float(L[n_].mean()), "dev_minus_noise": float(L["dev"].mean() - L[n_].mean()), "excess_of_script_gap": float(gap - (L["dev"].mean() - L[n_].mean()))}
bb, cc, p = mc(L[H], L["selfrr"], np.ones(len(qids), bool))
o["selfrewrite_vs_roman"] = {"acc_roman": float(L[H].mean()), "acc_after_selfrewrite": float(L["selfrr"].mean()), "roman_only_right": bb, "rewrite_only_right": cc, "mcnemar_p": p}
json.dump(o, open(OUT, "w"), indent=1); print(json.dumps(o, indent=1))
