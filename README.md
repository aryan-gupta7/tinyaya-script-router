# Typed in Roman, known in Devanagari

How much factual accuracy does Cohere Tiny Aya Global lose when a Hindi question is written in Roman script, and can its hidden state tell us when to be careful?
Entry for the Lexsi AI Safety and Alignment Hackathon, Track 2b, problem statement "Hallucination detection and abstention".
No fine-tuning. Model: `CohereLabs/tiny-aya-global` (3.35B). Data: Mintaka (Hindi and English, CC-BY-4.0), 2,400 questions in 1,281 question groups. All tables below are generated from files in `results/`.

## Headline
1. **Roman Hindi costs accuracy, beyond typo noise.** Devanagari 15.6%, Roman 10.1% (exact McNemar p = 2e-12, 248 questions right only in Devanagari against 115 right only in Roman). The gap exceeds what 3% and 6% typo noise explains by 3.8 points (95% interval 2.1 to 5.2, resampling question groups). On questions that are not yes/no: Devanagari 14.9%, Roman 6.3%. Roman keeps only 27.5% of the answers that Devanagari gets right.
2. **A hidden-state probe predicts wrong answers, but not better than entropy everywhere.** Out-of-fold on all questions the linear probe has AUROC 0.84 against 0.80 for the model's first-token entropy (difference +0.04, interval 0.01 to 0.08). On non-yes/no questions entropy is better, 0.87 against 0.82. Adding the probe to entropy gives no gain there. We do not claim the probe beats entropy.
3. **A probe-gated router did not help.** Asking Tiny Aya to rewrite the question into Devanagari raises accuracy from 10.1% to 11.0% for about three times the tokens. At matched coverage the probe router is no better than an entropy gate or random routing (`results/policy_selfrewrite.json`). We report this as a negative result.
4. **Human-typed Roman Hindi behaves differently from machine romanization.** 150 questions typed by three teammates from the English question scored 22.0%, higher than Devanagari (12.7%) and machine Roman (12.0%) on the same questions. Teammates kept English names in Latin letters and reworded freely, so this is not comparable with transliterating the Hindi text. It does show that the size of any script gap depends on how Roman text is produced, and that our machine-romanization gap should not be read as the loss real users suffer.

Plots: `results/figs/`. Full tables: `results/summary.md`.

## What was measured
- **Views** per question: Devanagari (Mintaka's Hindi), Roman (rule-based transliteration with casual-spelling cleanup, `rom_a2`), the same Devanagari with 3% and 6% character edits as noise controls, and for 150 questions Roman text typed by people. An earlier Roman version (`rom_a`) had a vowel-dropping bug that we found during review. It is kept in `results/analysis3_rom_a.json` and the gap holds with it (15.6% against 9.3%).
- **Prompt and scoring:** one template, greedy decoding, answer requested in English and scored against English gold aliases (normalised exact match, or token F1 of at least 0.8). The scorer is strict. A looser containment scorer keeps the gap (17.1% against 10.7%). A 90-answer check by the AI assistant found no wrong answer scored right (`docs/scorer_check.md`). It was not a human audit.
- **Probes:** logistic regression on the hidden state at the last prompt token (layers 19 and 31 reported), grouped 5-fold cross-validation by question group so related questions never straddle train and test.
- **Router:** answer if the first probe score is high, else re-render and answer if the second is high, else abstain. Baselines use the same coverage: always answer, always rewrite, entropy, mean log-probability, a non-neural gate, random routing and an oracle.

## Lexsi toolkits used (and what they did)
| Repo | Use | Evidence |
|---|---|---|
| AuditKIT | Paired comparison of per-question verdicts between views: Devanagari to Roman, 248 right-to-wrong and 115 wrong-to-right | `results/part3_results.json` |
| SafeTune | Its mean-difference linear probe, fitted on correct versus wrong Roman prompts (held-out AUROC 0.66). Weaker than our logistic-regression probe | `results/part3_results.json` |
| CuratorKIT | Clean and near-duplicate screen of the question pool with provenance export (2,400 in, 2,390 kept) | `dataset/curatorkit_out/` |

## Limitations
- Accuracy is low for every view because this is closed-book trivia on a 3B model, so the absolute numbers are small and many questions are unanswerable in any script.
- Mintaka may be in Tiny Aya's training data.
- Roman text is machine generated except the 150 typed questions, which were written from English and are not paired word for word with the Devanagari twins.
- One language, one model, one dataset. The probe is correlational. No mechanism is claimed.
- The scorer was checked by the AI assistant, not by a human. Noise controls at 3% and 6% may share some edits.
- Built in one day. Not done: human scorer audit, `tiny-aya-fire`, a patching experiment, CircuitKIT, a demo Space.

## Reproduce
Kaggle with a T4 GPU, model `cohere-labs/tiny-aya` (Transformers, `global`), internet on.
1. `code/kaggle_part1_main_run.py`: views and answers for Devanagari, Roman and the noise controls, with hidden states (about 40 minutes for 2,400 questions).
2. `code/kaggle_part2_rewrite.py`: self-rewrite step.
3. The final fixed-romanizer run, typed questions and toolkit use are in `code/kaggle_part4_final.py`.
4. Analysis on CPU: `HEAD=rom_a2 python code/analysis.py <dir>`, `HEAD=rom_a2 python code/analysis3.py <dir> out.json`, `python code/build_summary.py results`, `python code/make_figs.py results/analysis3_rom_a2.json results/figs`.
Seed 1234 everywhere. Decisions and deviations: `DECISIONS.md`.

## Artifacts
Hugging Face: dataset https://huggingface.co/datasets/s1nn3rx69/script-router-pairs and probes https://huggingface.co/s1nn3rx69/script-router-probes. Collections: https://huggingface.co/collections/s1nn3rx69/script-router-6afab6e7120d5e7cae179c2d and https://huggingface.co/collections/s1nn3rx69/script-router-data-6afabb4a51c779a9e3afedb5.

## Licences
Code MIT. Tiny Aya is CC-BY-NC with an acceptable use policy, so outputs stay non-commercial. Mintaka is CC-BY-4.0. CuratorKIT is under a non-commercial research licence.
