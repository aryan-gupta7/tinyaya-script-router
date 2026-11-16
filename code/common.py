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


# ---- fixed casual romanizer (rom_a2): keeps long vowels, only drops the inherent schwa
def rom_a2(dev):
    d = dev.replace("ॉ", "ो").replace("ऑ", "ओ").replace("़", "").replace("‍", "").replace("‌", "")
    s = to_itrans(d).replace("{}", "")
    for a, b in (("A", "\x01"), ("I", "\x02"), ("U", "\x03")): s = s.replace(a, b)
    for a, b in (("R^i", "ri"), ("kSh", "ksh"), ("GY", "gy"), ("~n", "n"), (".N", "n"), (".n", "n"), ("M", "n"), ("H", "h"),
                 ("T", "t"), ("D", "d"), ("N", "n"), ("Sh", "sh"), ("shh", "sh"), ("Ch", "chh"), ("aa", "a"), ("ii", "i"), ("uu", "u")):
        s = s.replace(a, b)
    s = s.lower()
    def fix(w):
        if len(w) >= 4 and w.endswith("a") and w[-2] not in "aeiou\x01\x02\x03": return w[:-1]
        return w
    s = re.sub(r"[a-z\x01\x02\x03]+", lambda m: fix(m.group(0)), s)
    return s.replace("\x01", "a").replace("\x02", "i").replace("\x03", "u")

_bv_old = build_views
def build_views(r):
    r = _bv_old(r); r["rom_a2"] = rom_a2(r["dev"]); return r
