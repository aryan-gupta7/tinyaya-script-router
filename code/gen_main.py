# gen_main.py: build views, run Tiny Aya greedy on each view, capture last-prompt-token activations.
import os, sys, json, time, random, subprocess
N = int(os.environ.get("N", "64")); BS = int(os.environ.get("BS", "24"))
VIEWS = os.environ.get("VIEWS", "rom_a,dev,rom_b,noise3,noise6,en").split(",")
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


T_START = time.time(); MAXH = float(os.environ.get("MAXH", "3.0"))
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
