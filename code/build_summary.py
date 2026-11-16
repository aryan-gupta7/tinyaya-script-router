# build_summary.py <results_dir>: writes results/summary.md from the json files (no numbers typed by hand).
import json, sys, os
R = sys.argv[1]; L = lambda n: json.load(open(f"{R}/{n}")) if os.path.exists(f"{R}/{n}") else None
a3 = {"rom_a2": L("analysis3_rom_a2.json"), "rom_a": L("analysis3_rom_a.json")}; pol = L("policy_selfrewrite.json"); p3 = L("part3_results.json")
o = ["# Results summary (generated from the json files in this folder)\n"]
def pct(x): return f"{100*x:.1f}%"
for k, a in a3.items():
    if not a: continue
    o.append(f"\n## Script gap, Roman view `{k}` ({a['n']} questions, {a['groups']} question groups)\n")
    o.append("| subset | n | Devanagari | Roman | Dev-only right | Roman-only right | McNemar p | gap 95% CI (group bootstrap) | Roman keeps of Dev-correct |\n|---|---|---|---|---|---|---|---|---|")
    for s, g in a["gap"].items():
        if "gap_ci_group_boot" in g:
            ci = g["gap_ci_group_boot"]; o.append(f"| {s} | {g['n']} | {pct(g['acc_dev'])} | {pct(g['acc_head'])} | {g['dev_only']} | {g['head_only']} | {g['p']:.1e} | [{100*ci[0]:.1f}, {100*ci[1]:.1f}] pp | {pct(g['romanized_keeps_of_dev_correct'])} |")
        else: o.append(f"| {s} | - | {pct(g['acc_dev'])} | {pct(g['acc_head'])} | {g['dev_only']} | {g['head_only']} | {g['p']:.1e} | - | - |")
    e = a["excess_over_noise"]; o.append(f"\nExcess of the script gap over the typo-noise gap: {100*e['value']:.1f} pp, 95% CI [{100*e['ci_group_boot'][0]:.1f}, {100*e['ci_group_boot'][1]:.1f}]. Noise controls: {pct(e['acc_noise3'])} (3% edits), {pct(e['acc_noise6'])} (6% edits).\n")
    for sub, t in a["oof_selective"].items():
        o.append(f"\n**Predicting wrong answers, {sub.replace('_',' ')} questions, grouped 5-fold out-of-fold** (higher AUROC better, lower AURC better)\n\n| signal | AUROC | AURC | accuracy at 30% coverage |\n|---|---|---|---|")
        for n, v in t.items():
            if isinstance(v, dict) and "auroc" in v: o.append(f"| {n} | {v['auroc']:.3f} | {v['aurc']:.3f} | {pct(v['acc_at_cov']['0.3'])} |")
        o.append(f"\nProbe minus entropy AUROC: {t['probe_minus_entropy_auroc']:+.3f}, 95% CI [{t['probe_minus_entropy_auroc_ci'][0]:+.3f}, {t['probe_minus_entropy_auroc_ci'][1]:+.3f}].\n")
    h = a["h2c_oof"]; o.append(f"\nRecoverability probe (among Roman-wrong, predict Devanagari-right): AUROC {h['auroc_probe']:.3f} (95% CI [{h['auroc_ci_probe'][0]:.3f}, {h['auroc_ci_probe'][1]:.3f}]) versus entropy {h['auroc_entropy']:.3f}; {h['positives']} positives among {h['n_wrong_head']}.\n")
if pol:
    o.append("\n## Router and baselines (test split, self-rewrite as the re-render step)\n\n| policy | coverage | right delivered | wrong delivered | tokens/query |\n|---|---|---|---|---|")
    for k in ("always_answer_nogate", "always_normalise_nogate"):
        v = pol["policy_test"][k]; o.append(f"| {k} | {pct(v['coverage'])} | {pct(v['right'])} | {pct(v['wrong'])} | {v['tokens']:.0f} |")
    for lam in (0.5, 1):
        for k in ("router", "always_answer+entropy", "always_answer+probe", "always_normalise+probe", "random_route", "oracle"):
            v = pol["policy_test"].get(f"{k}_lam{lam}")
            if v: o.append(f"| {k} (lambda={lam}) | {pct(v['coverage'])} | {pct(v['right'])} | {pct(v['wrong'])} | {v['tokens']:.0f} |")
if p3:
    o.append("\n## Typed Roman Hindi and toolkit runs\n```json\n" + json.dumps({k: v for k, v in p3.items()}, indent=1)[:6000] + "\n```")
os.makedirs(R, exist_ok=True); open(f"{R}/summary.md", "w").write("\n".join(o)); print("\n".join(o)[:3000])
