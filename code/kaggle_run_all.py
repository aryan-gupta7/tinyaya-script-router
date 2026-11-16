# common.py: scoring, Mintaka loading, views. Used by the Kaggle run and local analysis.
import json, re, random, string, urllib.request, os

SEED = 1234
ART = {"the", "a", "an"}
NUMW = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9","ten":"10"}
DEVNUM = str.maketrans("०१२३४५६७८९", "0123456789")

def norm(s):
    s = str(s).translate(DEVNUM).lower().strip()
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"[%s]" % re.escape(string.punctuation.replace("'", "")), " ", s).replace("'", "")
    toks = [NUMW.get(t, t) for t in s.split() if t not in ART]
    return " ".join(toks)

def f1(p, g):
    p, g = p.split(), g.split()
    if not p or not g: return 0.0
    from collections import Counter
    c = sum((Counter(p) & Counter(g)).values())
    if c == 0: return 0.0
    pr, rc = c / len(p), c / len(g)
    return 2 * pr * rc / (pr + rc)

def score(output, aliases, atype):
    """returns (correct, f1)"""
    out = output.strip().split("\n")[0]
    o = norm(out)
    if atype in ("boolean",):
        w = o.split()[0] if o else ""
        yes = {"yes", "हाँ", "हां", "ha", "haan", "true"}
        no = {"no", "नहीं", "nahi", "nahin", "false"}
        gold = norm(aliases[0])
        pred = "yes" if w in yes else ("no" if w in no else "")
        return (pred == gold, 1.0 if pred == gold else 0.0)
    best = 0.0; ok = False
    for a in aliases:
        g = norm(a)
        if not g: continue
        if o == g: return (True, 1.0)
        ff = f1(o, g); best = max(best, ff)
        if ff >= 0.8 and min(len(o), len(g)) >= 4: ok = True
    return (ok, best)

def load_mintaka(cache="/tmp/mintaka"):
    os.makedirs(cache, exist_ok=True)
    rows = []
    for sp in ("train", "dev", "test"):
        fp = f"{cache}/{sp}.json"
        if not os.path.exists(fp):
            urllib.request.urlretrieve(f"https://raw.githubusercontent.com/amazon-science/mintaka/main/data/mintaka_{sp}.json", fp)
        for e in json.load(open(fp)):
            a = e["answer"]
            if a.get("answer") is None or "hi" not in e.get("translations", {}): continue
            t = a["answerType"]
            if t == "boolean":
                al = ["yes" if a["answer"][0] else "no"]
            elif t == "entity":
                al = [x["label"]["en"] for x in a["answer"] if x.get("label", {}).get("en")] + [a.get("mention", "")]
            else:
                al = [str(x) for x in a["answer"]] + [a.get("mention", "")]
            al = [x for x in dict.fromkeys(al) if x]
            qe = e.get("questionEntity") or []
            ents = [q["name"] for q in qe if q.get("entityType") == "entity"]
            rows.append({"qid": "mk_" + e["id"], "split0": sp, "en": e["question"], "dev": e["translations"]["hi"],
                         "gold_aliases": al, "answer_type": t, "qtype": e["complexityType"], "category": e.get("category"),
                         "group_id": ents[0] if ents else "none_" + e["id"], "n_ents": len(ents)})
    return rows

# ---- views
def to_itrans(s):
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    return transliterate(s, sanscript.DEVANAGARI, sanscript.ITRANS)

def casual(itrans):
    s = itrans
    for a, b in [("R^i", "ri"), ("kSh", "ksh"), ("GY", "gy"), ("~n", "n"), (".N", "n"), (".n", "n"), ("M", "n"), ("H", "h"),
                 ("A", "a"), ("I", "i"), ("U", "u"), ("T", "t"), ("D", "d"), ("N", "n"), ("Sh", "sh"), ("shh", "sh"), ("Ch", "chh")]:
        s = s.replace(a, b)
    s = s.lower().replace("aa", "a").replace("ii", "i").replace("uu", "u")
    def fix(w):
        if len(w) >= 4 and w.endswith("a") and w[-2] not in "aeiou": return w[:-1]
        return w
    return re.sub(r"[a-z]+", lambda m: fix(m.group(0)), s)

def rom_a(dev): return casual(to_itrans(dev))
def rom_b(dev): return to_itrans(dev)

def noise(dev, rate, seed):
    rnd = random.Random(seed); out = []
    chars = list(dev); i = 0
    while i < len(chars):
        c = chars[i]
        if rnd.random() < rate:
            op = rnd.choice(["matra", "dup", "swap", "space"])
            if op == "matra" and "ा" <= c <= "ौ": i += 1; continue
            if op == "dup": out.append(c); out.append(c); i += 1; continue
            if op == "swap" and i + 1 < len(chars): out.append(chars[i + 1]); out.append(c); i += 2; continue
            if op == "space" and c == " ": i += 1; continue
        out.append(c); i += 1
    return "".join(out)

def build_views(r):
    r = dict(r)
    d = r["dev"]
    r["rom_a"] = rom_a(d); r["rom_b"] = rom_b(d)
    s = int(r["qid"][3:], 16) % (2**31) if re.fullmatch(r"[0-9a-f]+", r["qid"][3:]) else SEED
    r["noise3"] = noise(d, 0.03, SEED + s); r["noise6"] = noise(d, 0.06, SEED + s + 1)
    return r

PROMPT = ("Answer the question with a short answer only: an entity name, a number, a date, or yes or no. "
          "Answer in English. Do not explain.\nQuestion: {q}\nAnswer:")
REWRITE = ("Rewrite the following question in Hindi using Devanagari script. Keep the meaning and every name exactly. "
           "Output only the rewritten question.\nQuestion: {q}")


# gen_main.py: build views, run Tiny Aya greedy on each view, capture last-prompt-token activations.
import os, sys, json, time, random, subprocess
N = int(os.environ.get("N", "2400")); BS = int(os.environ.get("BS", "24"))
VIEWS = os.environ.get("VIEWS", "rom_a,dev,noise3,noise6").split(",")
ACT_VIEWS = set(os.environ.get("ACT_VIEWS", "dev,rom_a").split(","))
OUT = "/kaggle/working/out"; os.makedirs(OUT, exist_ok=True)
import glob
_c = glob.glob("/kaggle/input/**/config.json", recursive=True)
print("FOUND CONFIGS:", _c, flush=True)
MP = os.environ.get("MODEL_PATH") or os.path.dirname([p for p in _c if "aya" in p.lower()][0])
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "indic-transliteration"], check=False)
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

rows = [r for r in load_mintaka("/tmp/mintaka") if r["answer_type"] != "string"]
random.Random(SEED).shuffle(rows)
rows = [build_views(r) for r in rows[:N]]
with open(f"{OUT}/pairs.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

tok = AutoTokenizer.from_pretrained(MP); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float16, device_map={"": 0}); model.eval()
cfg = model.config; L = cfg.num_hidden_layers
LAYERS = list(range(3, L, 4)) + ([L - 1] if (L - 1) % 4 != 3 else [])
meta = {"n_layers": L, "hidden": cfg.hidden_size, "layers_saved": LAYERS, "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "N": N, "model_path": MP}
import transformers; meta["transformers"] = transformers.__version__
json.dump(meta, open(f"{OUT}/run_meta.json", "w"))
print(meta, flush=True)

cap = {}
def mk(i):
    def hook(m, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1: cap[i] = h[:, -1, :].detach().to("cpu", torch.float16)
    return hook
for i in LAYERS: model.model.layers[i].register_forward_hook(mk(i))

def prompt_ids(q):
    msg = [{"role": "user", "content": PROMPT.format(q=q)}]
    return tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)

ex = prompt_ids("TEST"); print("TEMPLATE:", repr(ex[:300]), "...", len(tok(ex).input_ids), "tokens", flush=True)


T_START = time.time(); MAXH = float(os.environ.get("MAXH", "1.75"))
def run_view(view, textfn, max_new=24, save_acts=False):
    fp = f"{OUT}/gen_{view}.jsonl"
    done = set()
    if os.path.exists(fp): done = {json.loads(l)["qid"] for l in open(fp)}
    todo = [r for r in rows if r["qid"] not in done]
    if not todo: return
    texts = {r["qid"]: textfn(r) for r in todo}
    todo.sort(key=lambda r: len(texts[r["qid"]]))
    acts_buf = {i: [] for i in LAYERS}; qid_buf = []
    t0 = time.time(); fo = open(fp, "a"); nbad = 0
    for s in range(0, len(todo), BS):
        if (time.time() - T_START) / 3600 > MAXH: print("TIME GUARD HIT", flush=True); break
        b = todo[s:s + BS]
        enc = tok([texts[r["qid"]] for r in b], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        cap.clear(); tb = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, output_scores=True, return_dict_in_generate=True,
                                 pad_token_id=tok.pad_token_id)
        lat = time.time() - tb
        seq = out.sequences[:, enc.input_ids.shape[1]:]
        sc = torch.stack(out.scores, 1).float()
        if not torch.isfinite(sc[:, 0]).all(): nbad += 1
        lp = torch.log_softmax(sc, -1)
        for j, r in enumerate(b):
            ids = seq[j].tolist(); n = len(ids)
            for k, t in enumerate(ids):
                if t in (tok.eos_token_id, tok.pad_token_id): n = k; break
            n_eff = max(n, 1)
            p0 = lp[j, 0].exp(); e0 = float(-(p0 * lp[j, 0]).nan_to_num().sum())
            mlp = float(np.mean([lp[j, k, ids[k]].item() for k in range(n_eff)]))
            o = tok.decode(ids[:n], skip_special_tokens=True).strip()
            c, ff = score(o, r["gold_aliases"], r["answer_type"])
            fo.write(json.dumps({"qid": r["qid"], "view": view, "prompt_tokens": int(enc.attention_mask[j].sum()), "gen_tokens": n,
                                 "output": o, "correct": bool(c), "f1": ff, "first_token_entropy": e0, "mean_logprob": mlp,
                                 "latency_s": lat / len(b)}, ensure_ascii=False) + "\n")
        fo.flush()
        if save_acts:
            for i in LAYERS: acts_buf[i].append(cap[i].numpy())
            qid_buf += [r["qid"] for r in b]
        if (s // BS) % 10 == 0: print(view, s + len(b), "/", len(todo), f"{time.time()-t0:.0f}s", flush=True)
    fo.close()
    if save_acts and qid_buf:
        for i in LAYERS: np.save(f"{OUT}/acts_{view}_L{i}.npy", np.concatenate(acts_buf[i]))
        json.dump(qid_buf, open(f"{OUT}/acts_{view}_qids.json", "w"))
    print("DONE", view, f"{time.time()-t0:.0f}s nonfinite_batches={nbad}", flush=True)

for view in VIEWS:  # order matters: rom_a and dev first (core), then controls
    run_view(view, lambda r, v=view: prompt_ids(r[v]), 24, view in ACT_VIEWS)

# self re-render: Tiny Aya rewrites ROM_A question into Devanagari, then answers the rewrite
if os.environ.get("RERENDER", "1") == "1":
    def rw_text(r):
        return tok.apply_chat_template([{"role": "user", "content": REWRITE.format(q=r["rom_a"])}], add_generation_prompt=True, tokenize=False)
    run_view("rewrite", rw_text, 64, False)
    rwo = {}
    fp = f"{OUT}/gen_rewrite.jsonl"
    if os.path.exists(fp):
        for l in open(fp):
            d = json.loads(l); rwo[d["qid"]] = d["output"].split("\n")[0]
    for r in rows: r["selfrr"] = rwo.get(r["qid"], r["rom_a"])
    run_view("selfrr", lambda r: prompt_ids(r["selfrr"]), 24, False)
    with open(f"{OUT}/pairs.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("ALL DONE", flush=True)


# ---- analysis on the same machine so only small json has to be downloaded
ANALYSIS = r'''# analysis.py: H1 gap, probes, non-neural control, router, baselines. Input: out/ dir from the Kaggle run.
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
'''
open("/kaggle/working/analysis.py", "w").write(ANALYSIS)
import subprocess as _sp
os.makedirs("/kaggle/working/results", exist_ok=True)
for _head in ("rom_a",):
    _r = _sp.run([sys.executable, "/kaggle/working/analysis.py", OUT], capture_output=True, text=True, cwd="/kaggle/working", env={**os.environ, "HEAD": _head})
    print(_r.stdout[-6000:], _r.stderr[-3000:], flush=True)
