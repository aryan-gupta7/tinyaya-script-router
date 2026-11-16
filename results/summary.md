# Results summary (generated from the json files in this folder)


## Script gap, Roman view `rom_a2` (2400 questions, 1281 question groups)

| subset | n | Devanagari | Roman | Dev-only right | Roman-only right | McNemar p | gap 95% CI (group bootstrap) | Roman keeps of Dev-correct |
|---|---|---|---|---|---|---|---|---|
| all | 2400 | 15.6% | 10.1% | 248 | 115 | 2.4e-12 | [3.8, 7.1] pp | 33.9% |
| non_boolean | 2048 | 14.9% | 6.3% | 222 | 45 | 2.7e-29 | [7.1, 10.2] pp | 27.5% |
| non_yesno_qtype | 2156 | 15.4% | 7.5% | 235 | 65 | 9.4e-24 | [6.3, 9.4] pp | 29.2% |
| no_corruption | 2395 | 15.6% | 10.1% | 248 | 115 | 2.4e-12 | [3.9, 7.2] pp | 33.7% |
| no_corruption_non_boolean | 2043 | 14.9% | 6.3% | 222 | 45 | 2.7e-29 | [7.1, 10.3] pp | 27.2% |
| containment_scorer_all | - | 17.1% | 10.7% | 270 | 116 | 2.9e-15 | - | - |

Excess of the script gap over the typo-noise gap: 3.8 pp, 95% CI [2.1, 5.2]. Noise controls: 14.6% (3% edits), 13.1% (6% edits).


**Predicting wrong answers, all questions, grouped 5-fold out-of-fold** (higher AUROC better, lower AURC better)

| signal | AUROC | AURC | accuracy at 30% coverage |
|---|---|---|---|
| probe_L19 | 0.835 | 0.787 | 26.7% |
| probe_L31 | 0.839 | 0.774 | 26.1% |
| entropy | 0.795 | 0.784 | 23.3% |
| mean_logprob | 0.758 | 0.809 | 23.1% |
| probe+entropy+logprob | 0.851 | 0.769 | 26.7% |
| non-neural control (length, entities, boolean) | 0.706 | 0.820 | 19.6% |

Probe minus entropy AUROC: +0.044, 95% CI [+0.011, +0.081].


**Predicting wrong answers, non boolean questions, grouped 5-fold out-of-fold** (higher AUROC better, lower AURC better)

| signal | AUROC | AURC | accuracy at 30% coverage |
|---|---|---|---|
| probe_L19 | 0.816 | 0.861 | 16.1% |
| probe_L31 | 0.817 | 0.858 | 15.8% |
| entropy | 0.872 | 0.835 | 17.4% |
| mean_logprob | 0.844 | 0.850 | 17.9% |
| probe+entropy+logprob | 0.870 | 0.837 | 18.1% |
| non-neural control (length, entities, boolean) | 0.558 | 0.930 | 8.0% |

Probe minus entropy AUROC: -0.055, 95% CI [-0.088, -0.022].


Recoverability probe (among Roman-wrong, predict Devanagari-right): AUROC 0.591 (95% CI [0.545, 0.626]) versus entropy 0.547; 248 positives among 2158.


## Script gap, Roman view `rom_a` (2400 questions, 1281 question groups)

| subset | n | Devanagari | Roman | Dev-only right | Roman-only right | McNemar p | gap 95% CI (group bootstrap) | Roman keeps of Dev-correct |
|---|---|---|---|---|---|---|---|---|
| all | 2400 | 15.6% | 9.3% | 261 | 110 | 2.8e-15 | [4.5, 7.9] pp | 30.4% |
| non_boolean | 2048 | 14.9% | 5.9% | 228 | 42 | 4.1e-32 | [7.5, 10.6] pp | 25.5% |
| non_yesno_qtype | 2156 | 15.4% | 7.0% | 242 | 60 | 4.6e-27 | [6.7, 10.0] pp | 27.1% |
| no_corruption | 1731 | 15.3% | 10.2% | 176 | 88 | 6.5e-08 | [2.9, 7.1] pp | 33.6% |
| no_corruption_non_boolean | 1488 | 15.2% | 6.4% | 163 | 32 | 2.4e-22 | [6.9, 10.7] pp | 27.9% |
| containment_scorer_all | - | 17.1% | 10.1% | 281 | 113 | 1.3e-17 | - | - |

Excess of the script gap over the typo-noise gap: 4.5 pp, 95% CI [2.9, 6.0]. Noise controls: 14.6% (3% edits), 13.1% (6% edits).


**Predicting wrong answers, all questions, grouped 5-fold out-of-fold** (higher AUROC better, lower AURC better)

| signal | AUROC | AURC | accuracy at 30% coverage |
|---|---|---|---|
| probe_L19 | 0.798 | 0.811 | 22.8% |
| probe_L31 | 0.819 | 0.800 | 23.2% |
| entropy | 0.765 | 0.810 | 20.1% |
| mean_logprob | 0.735 | 0.828 | 19.6% |
| probe+entropy+logprob | 0.831 | 0.794 | 24.0% |
| non-neural control (length, entities, boolean) | 0.688 | 0.843 | 17.6% |

Probe minus entropy AUROC: +0.054, 95% CI [+0.022, +0.086].


**Predicting wrong answers, non boolean questions, grouped 5-fold out-of-fold** (higher AUROC better, lower AURC better)

| signal | AUROC | AURC | accuracy at 30% coverage |
|---|---|---|---|
| probe_L19 | 0.752 | 0.881 | 12.1% |
| probe_L31 | 0.787 | 0.884 | 14.3% |
| entropy | 0.837 | 0.855 | 15.8% |
| mean_logprob | 0.817 | 0.865 | 15.1% |
| probe+entropy+logprob | 0.843 | 0.860 | 15.8% |
| non-neural control (length, entities, boolean) | 0.533 | 0.934 | 7.3% |

Probe minus entropy AUROC: -0.049, 95% CI [-0.081, -0.018].


Recoverability probe (among Roman-wrong, predict Devanagari-right): AUROC 0.586 (95% CI [0.545, 0.617]) versus entropy 0.553; 261 positives among 2176.


## Router and baselines (test split, self-rewrite as the re-render step)

| policy | coverage | right delivered | wrong delivered | tokens/query |
|---|---|---|---|---|
| always_answer_nogate | 100.0% | 7.9% | 92.1% | 446 |
| always_normalise_nogate | 100.0% | 9.2% | 90.8% | 1368 |
| router (lambda=0.5) | 9.7% | 2.5% | 7.2% | 470 |
| always_answer+entropy (lambda=0.5) | 9.0% | 2.3% | 6.8% | 446 |
| always_answer+probe (lambda=0.5) | 7.2% | 2.0% | 5.2% | 446 |
| always_normalise+probe (lambda=0.5) | 6.8% | 1.4% | 5.4% | 1368 |
| random_route (lambda=0.5) | 7.2% | 2.0% | 5.2% | 470 |
| oracle (lambda=0.5) | 14.4% | 14.4% | 0.0% | 470 |
| router (lambda=1) | 7.2% | 2.0% | 5.2% | 446 |
| always_answer+entropy (lambda=1) | 6.5% | 1.6% | 5.0% | 446 |
| always_answer+probe (lambda=1) | 5.4% | 1.4% | 4.1% | 446 |
| always_normalise+probe (lambda=1) | 4.1% | 0.7% | 3.4% | 1368 |
| random_route (lambda=1) | 5.4% | 1.4% | 4.1% | 446 |
| oracle (lambda=1) | 14.4% | 14.4% | 0.0% | 446 |

## Typed Roman Hindi and toolkit runs
```json
{
 "typed_vs_machine": {
  "n": 150,
  "acc": {
   "rom_h": 0.22,
   "rom_a2": 0.12,
   "rom_a": 0.10666666666666667,
   "dev": 0.12666666666666668
  },
  "dev_vs_typed": {
   "first_only": 6,
   "second_only": 20,
   "p": 0.009355306625366211
  },
  "typed_vs_machine_a2": {
   "first_only": 21,
   "second_only": 6,
   "p": 0.005924612283706665
  },
  "dev_vs_machine_a2": {
   "first_only": 13,
   "second_only": 12,
   "p": 1.0
  },
  "note": "typed by teammates from the English question, wording can differ from the Devanagari twin"
 },
 "safetune_probe": {
  "layer19": {
   "auroc_test": 0.6588229768446762,
   "n_train_pos": 70,
   "n_train_neg": 630,
   "n_test": 500
  },
  "layer31": {
   "auroc_test": 0.650791414212578,
   "n_train_pos": 70,
   "n_train_neg": 630,
   "n_test": 500
  }
 },
 "auditkit": {
  "version": "1.1.12",
  "run_summaries": {
   "dev": "run 5d3408c15c7399c8 (fingerprint 5d3408c15c7399c8)\n  exact_match: 0.1562 \u00b1 0.0074  (n=2400)\n  model: precomputed (not a local model -- no measurable size)",
   "rom_a2": "run 8b84724a9b118f88 (fingerprint 8b84724a9b118f88)\n  exact_match: 0.1008 \u00b1 0.0061  (n=2400)\n  model: precomputed (not a local model -- no measurable size)",
   "rom_a": "run e9b5febb08e95958 (fingerprint e9b5febb08e95958)\n  exact_match: 0.0933 \u00b1 0.0059  (n=2400)\n  model: precomputed (not a local model -- no measurable size)",
   "noise3": "run 237aa7e08be9786c (fingerprint 237aa7e08be9786c)\n  exact_match: 0.1458 \u00b1 0.0072  (n=2400)\n  model: precomputed (not a local model -- no measurable size)",
   "noise6": "run d3f47143fae65bbc (fingerprint d3f47143fae65bbc)\n  exact_match: 0.1308 \u00b1 0.0069  (n=2400)\n  model: precomputed (not a local model -- no measurable size)"
  },
  "comparisons": {
   "dev_vs_rom_a2": "Comparison: 5d3408c15c7399c8 (baseline) -> 8b84724a9b118f88 (candidate)\n  per-metric (baseline -> candidate | absolute delta | relative %):\n    exact_match: 0.1562 -> 0.1008  delta=-0.0554 (-35.5% rel)  (n=2400->2400, std=0.3631->0.3011)\n  samples: 248 went correct->wrong, 115 went wrong->correct\n  model: precomputed -> precomputed (no measurable size)",
   "dev_vs_rom_a": "Comparison: 5d3408c15c7399c8 (baseline) -> e9b5febb08e95958 (candidate)\n  per-metric (baseline -> candidate | absolute delta | relative %):\n    exact_match: 0.1562 -> 0.0933  delta=-0.0629 (-40.3% rel)  (n=2400->2400, std=0.3631->0.2909)\n  samples: 261 went correct->wrong, 110 went wrong->correct\n  model: precomputed -> precomputed (no measurable size)",
   "dev_vs_noise3": "Comparison: 5d3408c15c7399c8 (baseline) -> 237aa7e08be9786c (candidate)\n  per-metric (baseline -> candidate | absolute delta | relative %):\n    exact_match: 0.1562 -> 0.1458  delta=-0.0104 (-6.7% rel)  (n=2400->2400, std=0.3631->0.3529)\n  samples: 50 went correct->wrong, 25 went wrong->correct\n  model: precomputed -> precomputed (no measurable size)",
   "dev_vs_noise6": "Comparison: 5d3408c15c7399c8 (baseline) -> d3f47143fae65bbc (candidate)\n  per-metric (baseline -> candidate | absolute delta | relative %):\n    exact_match: 0.1562 -> 0.1308  delta=-0.0254 (-16.3% rel)  (n=2400->2400, std=0.3631->0.3372)\n  samples: 102 went correct->wrong, 41 went wrong->correct\n  model: precomputed -> precomputed (no measurable size)"
  }
 },
 "curatorkit": {
  "input": 2400,
  "rejected_lines": 0,
  "files": [
   "README.md",
   "lexsi_provenance.json",
   "manifest.json",
   "sft_alpaca.jsonl",
   "rejected.jsonl",
   "checksums.txt",
   "dataset_card.md"
  ]
 }
}
```