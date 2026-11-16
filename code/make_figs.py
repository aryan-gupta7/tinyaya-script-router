# make_figs.py <analysis3.json> <outdir>: accuracy-by-view bars and selective-accuracy curves from saved results.
import json, sys, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
a = json.load(open(sys.argv[1])); od = sys.argv[2]; os.makedirs(od, exist_ok=True)
g = a["gap"]; names = ["all", "non_boolean", "no_corruption_non_boolean"]
fig, ax = plt.subplots(figsize=(7, 3.6)); w = 0.38
for i, n in enumerate(names):
    ax.bar(i - w/2, g[n]["acc_dev"] * 100, w, color="#1f77b4", label="Devanagari" if i == 0 else None)
    ax.bar(i + w/2, g[n]["acc_head"] * 100, w, color="#d62728", label="Roman" if i == 0 else None)
    ax.text(i - w/2, g[n]["acc_dev"] * 100 + .3, f'{g[n]["acc_dev"]*100:.1f}', ha="center", fontsize=8); ax.text(i + w/2, g[n]["acc_head"] * 100 + .3, f'{g[n]["acc_head"]*100:.1f}', ha="center", fontsize=8)
ax.set_xticks(range(len(names))); ax.set_xticklabels([f'{n.replace("_"," ")}\n(n={g[n]["n"]})' for n in names], fontsize=8); ax.set_ylabel("accuracy (%)"); ax.legend(loc="upper right"); ax.set_title("Tiny Aya Global, Mintaka Hindi: Devanagari vs Roman")
plt.tight_layout(); plt.savefig(f"{od}/accuracy_by_script.png", dpi=150); plt.close()
for sub in ("all", "non_boolean"):
    t = a["oof_selective"][sub]; covs = ["0.1", "0.2", "0.3", "0.5", "0.7"]; fig, ax = plt.subplots(figsize=(6, 3.8))
    for k, v in t.items():
        if isinstance(v, dict) and "acc_at_cov" in v: ax.plot([float(c) for c in covs], [v["acc_at_cov"][c] * 100 for c in covs], marker="o", label=k.split(" (")[0])
    ax.set_xlabel("coverage (fraction answered)"); ax.set_ylabel("accuracy of answered (%)"); ax.set_title(f"Selective accuracy, {sub.replace('_',' ')} (out-of-fold)"); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{od}/selective_{sub}.png", dpi=150); plt.close()
print("figs ok")
