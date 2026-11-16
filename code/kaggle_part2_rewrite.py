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
TYPED = json.loads(r'''{"mk_49d267c0": "eragon kb paida hua?", "mk_2806c339": "napoleon ne kab haar maan li", "mk_115c6477": "maryl streep ka janm kb hua", "mk_13cdde92": "kaunse sheher ko beantown bolte hai", "mk_1f07df2b": "america ke dusre rashtrapati ki maut kitni umra pe hui", "mk_6a84b2e5": "ernest hemingway kaise mara", "mk_299a4c54": "kya democratic party sau saal se zyada purani h?", "mk_8e9166f6": "crash bandicoot kis type ka jaanwar hai", "mk_1e728684": "kya 2020 me donald trump election jeeta?", "mk_9bdf8ef3": "meursault ko kis baat ki sazaa hui?", "mk_4e93ad5e": "call of duty series me kitne games hai", "mk_51291cd7": "kya diablo series blizzard ne banai?", "mk_9edd2779": "kitne desho ki abadi 50 crore se zada hai", "mk_353c4540": "kaunsi actress avenger me thi aur lost in translation me bhi thi", "mk_94139615": "chris evans kitni filme kar chuka hai?", "mk_bc0d71c7": "red hot chilli peppers ka mukhya gaayak kaun hai", "mk_0f60219b": "kis khel me, jo supergiant games ne banaya hai, usme khiladi zagreus ka kirdaar lete hai jaise hi wo underworld se bhagne ki koshish krta hai", "mk_45a87547": "kaun bada hai? cate blanchett ya tilda swinton", "mk_36c5df28": "kis final fantasy game me mukhya kirdaar bijli hai", "mk_ce4dd5b7": "rashtapati donald trump ki kitni patniya reh chuki hai", "mk_2560abd0": "thomas jefferson ki pehli biwi ka kya naam tha", "mk_a764337e": "stephenie meyer ki pehli book series kab nikli thi?", "mk_831757fd": "kaunse sims game ki sbse zada bikri hui", "mk_081b9f5c": "nolan ki batman trilogy me kitni filme hai", "mk_25c6bcf6": "kya felt ne kabhi cherry red records ke liye record kiya tha?", "mk_5999b944": "metroid me mukhya kirdaar kaun hai?", "mk_49b8206b": "america ke sbse zada abadi wale state ki rajdhani kya hai?", "mk_6a45f064": "senses fail ka kaunsa album 2004 me aaya tha?", "mk_66832edc": "dusre vishwa yudh ke natije me kaunsi international organization bani thi?", "mk_d2faac77": "pehle black senator kaun the?", "mk_455e31fc": "kis president ki umra inauguration ke time sbse zada thi?", "mk_805eba17": "kya call of duty activision ne develop kiya tha?", "mk_b21a5edc": "golden state warriors ka wo kaunsa NBA player hai jo 2019-20 ka pura season nahi khel paya tha kyuki 2019 NBA finals ke game 6 me uska ACL injury ho gaya tha?", "mk_1117fbf9": "pokemon games ki kaunsi generation me pokemon breed nahi kar sakte?", "mk_ad37eec7": "america ke 44th president ka janm kis state me hua tha?", "mk_343700c6": "curse of the black pearl me will elizabeth ko court karne ke layak kyu nahi hai?", "mk_54330901": "ancient greece ka kaunsa sheher zada populated tha, sparta ya athens?", "mk_0fbb68de": "south america ka elevation ke hisaab se chautha sbse uncha pahad kaunsa hai?", "mk_8e5b5966": "jurassic park me park ke owner john hammond ka role kisne kiya tha?", "mk_a976dbd9": "twice ke kaunse members south korea se nahi hai?", "mk_bf3a1127": "america ke saath lage hue kaunse desh uske south me nahi hai?", "mk_c430f831": "revolutionary war ki sbse lambi ladai kaunsi thi?", "mk_fd2c3558": "office me marne wale aakhri president kaun the?", "mk_2b2c1c25": "J K Rowling ki abhi shaadi kisse hui hai?", "mk_7ff16cbd": "kya assassin's creed 2008 se pehle release hua tha?", "mk_9219012a": "jimi hendrix rock musician the ya heavy metal musician?", "mk_59902bf9": "kya president trump ne ek se zada term serve kiya tha?", "mk_423a4b34": "a song of ice and fire ki kaunsi book sbse lambi hai?", "mk_2bcdb5f6": "kaunsi movie direct karne ke liye josh trank maan gaye the lekin filming shuru hone se pehle hi hat gaye?", "mk_84a8bdc4": "new york yankees kitni world series me khele hai lekin jeete nahi?", "mk_2c18021c": "1980 ke baad ab tak kitne democrat presidents hue hai?", "mk_fa21408c": "NFL ki sbse nayi team kaunsi hai?", "mk_a1bb56c0": "halo combat evolved kb release hua tha?", "mk_585d5933": "pehle roman emperor ki teesri biwi kaun thi?", "mk_e2953651": "kis desh ki abadi zada hai, bangladesh ya brazil?", "mk_d6dc0dbb": "america me freshwater lakes kitni hai?", "mk_c7c9612a": "breaking benjamin ka kaunsa member band se kabhi nahi gaya?", "mk_23521e69": "red dead redemption ki kaunsi game me arthur morgan mukhya kirdaar hai?", "mk_d067dccf": "ab tak ki sbse kam kamai karne wali movie kaunsi hai?", "mk_59c6fe4b": "king tut ka kaunsa rishtedaar KV35 me dafn hai?", "mk_4cf1e359": "kitne presidents ka janm texas me hua tha?", "mk_74d115a2": "per capita sabse zada metal bands kis desh me hai?", "mk_904ee94d": "england ke pehle prime minister kaun the?", "mk_4671b0c6": "kya daniel radcliffe american hai?", "mk_7272df70": "telegraph kisne banaya tha?", "mk_cd76bb1a": "little women book me sbse chhoti kaun hai?", "mk_746f48bc": "japan ki imperial capital kya thi?", "mk_944c6d00": "judas priest ka kaunsa singer american tha?", "mk_ef2888ec": "kya J.K. Rowling stephen king se chhoti hai?", "mk_31639a3d": "the winery dogs ka wo kaunsa member hai jiska janm pennsylvania me hua tha?", "mk_9dbf6ed4": "mario andretti ne indy 500 kitni baar jeeta hai?", "mk_04fff9ea": "kaunsa sheher bada hai, new york ya los angeles?", "mk_6b3bd4e5": "final fantasy ki kaunsi game me noctis main character hai?", "mk_745e9aea": "squid game ka kaunsa episode sbse lamba hai?", "mk_e2f206f0": "UK ke pehle tory prime minister kaun the?", "mk_b4bacb88": "hunger games series ki kaunsi book 2010 me release hui thi?", "mk_7ff1e3fa": "roosevelt ne america ke president ke roop me kitne saal ka term serve kiya tha?", "mk_74adcb0b": "betty white ka janm kis saal hua tha?", "mk_2601c6b9": "second world war ki aakhri badi ladai kaunsi thi?", "mk_d32ec3b2": "animal crossing ki kaunsi game me villagers apne aap ghar chhod ke nahi ja sakte?", "mk_f7c30a3a": "america ke kis president ki teesre president ke saath kaafi dushmani thi?", "mk_90d8f2cd": "america ke kitne presidents ko congress ne formally impeach kiya hai?", "mk_464f0550": "kis series me zada prequel books hai, harry potter ya the maze runner?", "mk_db52e0f5": "kya russia duniya ka sbse bada desh hai?", "mk_88116c97": "terry pratchett ka kaunsa novel unki maut ke baad publish hua tha?", "mk_1851a6eb": "kya president trump ko kabhi impeach kiya gaya tha?", "mk_d2978af4": "1891 me office me marne wale california ke senator kaun the?", "mk_342c6fe7": "kaunse author howard university ke alumni the aur unhone nobel prize in literature jeeta?", "mk_8400c4db": "second world war me nazi germany kis desh par kabza nahi kar paya tha?", "mk_9a19831b": "super bowl me ab tak ka sabse lamba field goal kitne yards ka tha?", "mk_fa08ae35": "jacksonville jaguars ke pehle coach kaun the?", "mk_fbc0640a": "kya malaysia ki population taiwan se kam hai?", "mk_0053ba67": "kya vincent lombardo ki book china house 1980s me likhi gayi thi?", "mk_76027ffa": "kaunsi war zada lambi chali, iraq war ya korean war?", "mk_d4f5ad3a": "metallica ke sabse bade member ki shaadi kisse hui hai?", "mk_e5c7869a": "roger federer ke paas kitne olympic medals hai jo silver nahi hai?", "mk_f3c6f21b": "1950 ke baad kaunse incumbent US presidents second term ke liye lade lekin jeet nahi paye?", "mk_d6ef54e5": "duniya me sabse zada meteorites kahan mile hai?", "mk_181dc5f6": "harper lee ki aakhri book kaunsi thi?", "mk_b7dfdd1c": "kya twilight ki paanch books hai?", "mk_fdb461b3": "twilight saga me kitni filme hai?", "mk_7e7d2c9d": "legend of heroes ki crossbell arc ki pehli game kaunsi hai?", "mk_0dc05e5b": "confederate states me union me wapas judne wala aakhri state kaunsa tha?", "mk_c665beb3": "kis author ki books par sabse zada movies bani hai?", "mk_a4f6c3f0": "pevensie kitne bhai behen hai?", "mk_79c4937a": "career hits ke hisaab se paanchve number par kaun hai?", "mk_0e392685": "american civil war me union se sabse pehle kaunsa state alag hua tha?", "mk_48f44d0d": "kaunsa NFL quarterback ek season me sabse zada passing yards kar chuka hai?", "mk_695e3f04": "nobel peace prize jeetne wale aur soviet union ke president kaun the?", "mk_4f8c06b0": "kaunsi baseball team american league east me khelti hai aur jiska home stadium st petersburg florida me hai?", "mk_9f305943": "shazam ki kitni movies hai?", "mk_2ab3e7d0": "second world war me attack hua kaunsa US battleship ab pearl harbor me war memorial ke roop me rakha hai?", "mk_b0332ca2": "pacific northwest me kaunsa state hai jisme salem naam ka sheher hai?", "mk_46cf0001": "the old man and the sea ke author ki pehli biwi kaun thi?", "mk_3980db89": "king henry VIII ka kaunsa bachne wala bachcha height me sabse chhota tha?", "mk_24383c36": "america me ab tak kitni auratein secretary of state reh chuki hai?", "mk_ca621e91": "rocky movies ke character ka poora naam kya hai?", "mk_05fcad88": "kaunsa roman emperor sabse lambe time tak rule kiya tha?", "mk_7963b70d": "backstreet boys ka kaunsa member 2006 me band me nahi tha lekin 2012 me wapas aa gaya?", "mk_40f6dc9c": "kis female athlete ne sabse zada olympic medals jeete hai?", "mk_b415a817": "tolkien ka kaunsa novel lord of the rings universe me set hai lekin trilogy ka part nahi hai?", "mk_008b2254": "duniya me kitne deserts ka area 3 million square miles se zada hai?", "mk_f23c1a33": "kya to kill a mockingbird 1977 se pehle release hui thi?", "mk_7ca3fe57": "kya garden state me zach braff lead actor tha?", "mk_61a09c01": "witcher series me kitni games hai?", "mk_827fe5b9": "kya second world war me allies ne nuclear weapons use kiye the?", "mk_f2d7db77": "sweden ka pehla king kaun tha?", "mk_20870583": "chris butler ki 2019 ki kaunsi movie best animated feature film ke liye nominate hui thi lekin jeet nahi payi?", "mk_a06aa9a2": "harry potter ka role karne wala actor ab kitne saal ka hai?", "mk_2ed7d419": "kis game me sabse zada characters hai?", "mk_93855345": "kya twilight saga new moon 1995 se pehle release hui thi?", "mk_7317f406": "kya mario aur luigi bhai hai?", "mk_c870591d": "declaration of independence par sign karne wale aur france me pehle ambassador kaun the?", "mk_1733d72b": "area ke hisaab se kaunsa desh bada hai, brazil ya indonesia?", "mk_e36136d4": "kya cloud strife ki height tifa lockhart se zada hai?", "mk_84d6612d": "NASA kis saal bana tha?", "mk_3a8af787": "1967 me super bowl kaun haara tha?", "mk_de38aca4": "spaceballs ke director ka janm kb hua tha?", "mk_fc106384": "aaj ke time me iran ke naam se jaane jaane wale desh me ancient persian civilization thi?", "mk_a5706237": "2014 me album of the year ka award kis album ko mila tha?", "mk_63efd289": "rene zellweger ne kis country music star se shaadi ki thi?", "mk_c7cc2086": "kya zelda hi usually link ko bachati hai?", "mk_5a294cc0": "united kingdom ki parliament ka lower house kya hai?", "mk_c1c0541b": "chronological order me child's play ki doosri movie kaunsi thi?", "mk_36b02ea5": "hyrule kingdom me set video game pehli baar kis saal aaya tha?", "mk_209f8b5d": "sabse zada passes complete kisne kiye hai?", "mk_30880c95": "avengers ke director ka apni wife se divorce kb hua tha?", "mk_245cc55d": "jane fonda ne kitne academy awards jeete hai?", "mk_d518f8d9": "kya street fighter mortal kombat se pehle aaya tha?", "mk_e55448f2": "kis sheher ko space city ke naam se jaana jata hai?"}''')
rows_full = list(rows)
rows = [r for r in rows if r["qid"] in TYPED]
for r in rows: r["rom_h"] = TYPED[r["qid"]]
print("typed rows", len(rows), flush=True)
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
def run_view(view, textfn, max_new=24, save_acts=False, scores=True, bs=None):
    fp = f"{OUT}/gen_{view}.jsonl"
    done = set()
    if os.path.exists(fp): done = {json.loads(l)["qid"] for l in open(fp)}
    todo = [r for r in rows if r["qid"] not in done]
    if not todo: return
    texts = {r["qid"]: textfn(r) for r in todo}
    todo.sort(key=lambda r: len(texts[r["qid"]]))
    acts_buf = {i: [] for i in LAYERS}; qid_buf = []
    t0 = time.time(); fo = open(fp, "a"); nbad = 0
    BSZ = bs or BS
    for s in range(0, len(todo), BSZ):
        if (time.time() - T_START) / 3600 > MAXH: print("TIME GUARD HIT", flush=True); break
        b = todo[s:s + BSZ]
        enc = tok([texts[r["qid"]] for r in b], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        cap.clear(); tb = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, output_scores=scores, return_dict_in_generate=True,
                                 pad_token_id=tok.pad_token_id)
        lat = time.time() - tb
        seq = out.sequences[:, enc.input_ids.shape[1]:]
        if scores:
            sc = torch.stack(out.scores, 1).float()
            if not torch.isfinite(sc[:, 0]).all(): nbad += 1
            lp = torch.log_softmax(sc, -1)
        for j, r in enumerate(b):
            ids = seq[j].tolist(); n = len(ids)
            for k, t in enumerate(ids):
                if t in (tok.eos_token_id, tok.pad_token_id): n = k; break
            n_eff = max(n, 1)
            if scores:
                p0 = lp[j, 0].exp(); e0 = float(-(p0 * lp[j, 0]).nan_to_num().sum())
                mlp = float(np.mean([lp[j, k, ids[k]].item() for k in range(n_eff)]))
            else:
                e0 = 0.0; mlp = 0.0
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


torch.cuda.empty_cache()
rows = rows_full
def rw_text(r):
    return tok.apply_chat_template([{"role": "user", "content": REWRITE.format(q=r["rom_a"])}], add_generation_prompt=True, tokenize=False)
run_view("rewrite", rw_text, 64, False, scores=False, bs=16)
rwo = {}
for l in open(f"{OUT}/gen_rewrite.jsonl"):
    d = json.loads(l); rwo[d["qid"]] = d["output"].split("\n")[0]
for r in rows: r["selfrr"] = rwo.get(r["qid"], r["rom_a"])
run_view("selfrr", lambda r: prompt_ids(r["selfrr"]), 24, False, scores=True, bs=16)
rows = [r for r in rows_full if r["qid"] in TYPED]
torch.cuda.empty_cache()
print("GEN DONE", flush=True)


# ================= part 2: ROM_H analysis, SafeTune probe, AuditKIT, CuratorKIT =================
import glob, traceback
from scipy.stats import binomtest
RES = {}
def _find(name):
    c = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return c[0] if c else None
def _load(name):
    p = _find(name); return [json.loads(l) for l in open(p)] if p else None
main = {v: {r["qid"]: r for r in (_load(f"gen_{v}.jsonl") or [])} for v in ("rom_a", "dev", "noise3", "noise6")}
main["selfrr"] = {r["qid"]: r for r in map(json.loads, open(f"{OUT}/gen_selfrr.jsonl"))}
main_pairs = _load("pairs.jsonl") or []
print("main outputs found:", {k: len(v) for k, v in main.items()}, "pairs", len(main_pairs), flush=True)
romh = {r["qid"]: r for r in map(json.loads, open(f"{OUT}/gen_rom_h.jsonl"))}
try:
    q = [x for x in romh if x in main["rom_a"] and x in main["dev"]]
    lab = {"rom_h": np.array([romh[x]["correct"] for x in q], int), "rom_a": np.array([main["rom_a"][x]["correct"] for x in q], int), "dev": np.array([main["dev"][x]["correct"] for x in q], int)}
    def mc(a, b):
        bb = int(((a == 1) & (b == 0)).sum()); cc = int(((a == 0) & (b == 1)).sum())
        return {"a_only": bb, "b_only": cc, "p": float(binomtest(bb, bb + cc, .5).pvalue) if bb + cc else 1.0}
    RES["romh"] = {"n": len(q), "acc": {k: float(v.mean()) for k, v in lab.items()},
                   "dev_vs_romh": mc(lab["dev"], lab["rom_h"]), "roma_vs_romh": mc(lab["rom_a"], lab["rom_h"]), "dev_vs_roma": mc(lab["dev"], lab["rom_a"]),
                   "note": "typed Roman Hindi was written by teammates from the English question, so wording can drift from the Devanagari twin"}
    print(RES["romh"], flush=True)
except Exception: RES["romh_error"] = traceback.format_exc()[-800:]; print(RES["romh_error"])

# SafeTune: mean-difference linear probe (its own fit_linear_probe) on correct vs wrong ROM_A prompts
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "safetune"], check=False)
    import importlib; sp = importlib.import_module("safetune.interventions.steer.probe_guard")
    from sklearn.metrics import roc_auc_score
    prs = [p for p in main_pairs if p["qid"] in main["rom_a"]]
    rng2 = random.Random(7); gids = sorted({p["group_id"] for p in prs}); rng2.shuffle(gids); gtr = set(gids[:int(.6 * len(gids))])
    tr_ = [p for p in prs if p["group_id"] in gtr][:700]; te_ = [p for p in prs if p["group_id"] not in gtr][:500]
    pos = [prompt_ids(p["rom_a"]) for p in tr_ if main["rom_a"][p["qid"]]["correct"]]
    neg = [prompt_ids(p["rom_a"]) for p in tr_ if not main["rom_a"][p["qid"]]["correct"]]
    out_ = {}
    for lay in (19, 27):
        probe = sp.fit_linear_probe(model, tok, pos, neg, sp.LinearProbeConfig(pick_layer=lay, pool_method="last_token"))
        X = sp._collect_pooled(model, tok, [prompt_ids(p["rom_a"]) for p in te_], lay, "last_token")
        s = probe.score(X.float()).cpu().numpy(); y = np.array([main["rom_a"][p["qid"]]["correct"] for p in te_], int)
        out_[f"layer{lay}"] = {"auroc_test": float(roc_auc_score(y, s)), "n_train_pos": len(pos), "n_train_neg": len(neg), "n_test": len(te_)}
    RES["safetune_probe"] = out_; print("SAFETUNE", out_, flush=True)
except Exception: RES["safetune_error"] = traceback.format_exc()[-1500:]; print(RES["safetune_error"], flush=True)

# AuditKIT: precomputed verdicts, paired comparison between views
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "auditkit"], check=False)
    import auditkit as ak
    runs = {}
    for v in ("dev", "rom_a", "noise3", "noise6", "selfrr"):
        smp = [ak.Sample(input=q_, target="correct", id=q_, actual_output=("correct" if r["correct"] else "wrong")) for q_, r in main[v].items()]
        runs[v] = ak.evaluate(smp, "precomputed", "exact_match")
    txt = {}
    for v in ("rom_a", "noise3", "noise6", "selfrr"):
        txt[f"dev_vs_{v}"] = str(ak.compare(runs["dev"], runs[v]).summary())
    RES["auditkit"] = {"version": getattr(ak, "__version__", "?"), "summaries": {v: runs[v].summary() if not callable(getattr(runs[v], "summary", None)) else str(runs[v].summary()) for v in runs}, "comparisons": txt}
    print(json.dumps(RES["auditkit"], indent=1)[:3000], flush=True)
except Exception: RES["auditkit_error"] = traceback.format_exc()[-1500:]; print(RES["auditkit_error"], flush=True)

# CuratorKIT: clean and near-duplicate screen on the question pool, with provenance export
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "curatorkit"], check=False)
    from curatorkit import Curator, CuratorConfig
    with open("/kaggle/working/pool.jsonl", "w") as f:
        for p in main_pairs: f.write(json.dumps({"id": p["qid"], "instruction": p["en"], "output": p["gold_aliases"][0]}, ensure_ascii=False) + "\\n")
    cfg = CuratorConfig(dataset="/kaggle/working/pool.jsonl", format="auto", dedup="minhash", minhash_threshold=0.85, min_tokens=1, clean=True,
                        output_dir="/kaggle/working/curatorkit_out", export_formats=["alpaca"], name="mintaka_hi_pool")
    cres = Curator(cfg).run(); cres.print_summary()
    rej = sum(1 for _ in open("/kaggle/working/curatorkit_out/rejected.jsonl")) if os.path.exists("/kaggle/working/curatorkit_out/rejected.jsonl") else None
    RES["curatorkit"] = {"input": len(main_pairs), "rejected_lines": rej, "files": os.listdir("/kaggle/working/curatorkit_out")}
    print(RES["curatorkit"], flush=True)
except Exception: RES["curatorkit_error"] = traceback.format_exc()[-1500:]; print(RES["curatorkit_error"], flush=True)

json.dump(RES, open("/kaggle/working/part2_results.json", "w"), indent=1, default=str)
print("PART2 DONE", flush=True)
