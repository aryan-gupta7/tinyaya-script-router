# Typed in Roman, known in Devanagari

*A noise-controlled measurement of the Roman-script accuracy gap in Cohere Tiny Aya, with a negative result for probe-gated routing. The gap is measured on machine-romanized Hindi; text typed by people did not show the loss.*

This entry is about **wrong answers to benign factual questions**, not harmful-request refusals. Other entries study romanized-Hindi safety refusals. We do not.

Entry for the Lexsi AI Safety and Alignment Hackathon, Track 2b, problem statement "Hallucination detection and abstention". No fine-tuning. Model: `CohereLabs/tiny-aya-global` (3.35B). Data: Mintaka (Hindi and English, CC-BY-4.0), 2,400 questions in 1,281 question groups. All tables below are generated from files in `results/`.

## Headline (lead with questions that are not yes/no)
1. **Machine-romanized Hindi costs accuracy, beyond typo noise.** On the 2,048 questions that are not yes/no, Devanagari scores 14.9% and Roman 6.3% (222 right only in Devanagari, 45 right only in Roman). Roman keeps only 27.5% of the answers Devanagari gets right. The loss appears for entity answers (17.3% to 7.6%), numbers (5.9% to 1.4%) and dates (16.9% to 7.0%) (`results/analysis4.json`), and holds when each question group gets one vote (164 groups favour Devanagari, 33 favour Roman, p = 4e-22).
2. **The yes/no questions go the other way, and that is a format artefact.** On the 352 yes/no questions Devanagari scores 19.6% and Roman 32.1%, because the model often answers a yes/no question with an entity in Devanagari and says "no" more often in Roman. Pooling them gives the all-question figures (Devanagari 15.6%, Roman 10.1%, McNemar p = 2e-12), which mix this format effect with the script effect, so we do not lead with them.
3. **How much of the gap is typo noise?** The script gap exceeds the typo-noise gap by 3.0 points against 6% noise and by 4.5 points against 3% noise (3.8 on average, 95% interval 2.1 to 5.2 with group resampling). The two noise controls are not independent (the 6% edits include the 3% ones), and character typos in Devanagari are only a rough stand-in for transliteration error.
4. **Hidden-state probes did not show that recoverability is readable before generation.** A probe for "would the Devanagari answer be right" reaches AUROC 0.59 (95% interval 0.54 to 0.63) among Roman-wrong answers, with 248 positives. That is below the 0.65 bar and below the 400 positives we planned for, so we do not claim it. A probe for "will this Roman answer be wrong" reaches out-of-fold AUROC 0.84 against 0.80 for first-token entropy overall, but entropy is better on non-yes/no questions (0.87 against 0.82), so we claim comparable, not better.
5. **The router did not help.** At matched coverage the probe router is no better than an entropy gate or random routing (`results/policy_selfrewrite.json`), covering about 10% of questions at 26% selective accuracy. Tiny Aya's own rewrite to Devanagari moves accuracy from 10.1% to 11.0% (129 questions lost, 151 gained, McNemar p = 0.21, not significant) for about three times the tokens.
6. **Human-typed Roman Hindi did not show the loss.** 150 questions typed by three teammates from the English question scored 22.0%, higher than Devanagari (12.7%) and machine Roman (12.0%) on the same questions. A plausible reason, which we did not test, is that machine romanization garbles names that people keep in Latin letters. So the machine-romanization gap above should not be read as the loss real users suffer.

Plots: `results/figs/`. Full tables: `results/summary.md`.

## What was measured
- **Views** per question: Devanagari (Mintaka's Hindi), Roman (rule-based transliteration with casual-spelling cleanup, `rom_a2`), the same Devanagari with 3% and 6% character edits as noise controls, and for 150 questions Roman text typed by people. An earlier Roman version (`rom_a`) had a vowel-dropping bug that we found during review. It is kept in `results/analysis3_rom_a.json` and the gap holds with it (15.6% against 9.3%).
- **Prompt and scoring:** one template, greedy decoding, answer requested in English and scored against English gold aliases (normalised exact match, or token F1 of at least 0.8). The scorer is strict. A looser containment scorer keeps the gap (17.1% against 10.7%). A read-through of 90 answers by the AI assistant found no wrong answer scored right but 3 correct sentence-style answers scored wrong, slightly more in the Roman view (`docs/scorer_check.md`). It was not a human audit, and this could inflate the gap a little.
- **Probes:** logistic regression on the hidden state at the last prompt token (layers 19 and 31 reported), grouped 5-fold cross-validation by question group so related questions never straddle train and test.
- **Router:** answer if the first probe score is high, else re-render and answer if the second is high, else abstain. Baselines use the same coverage: always answer, always rewrite, entropy, mean log-probability, a non-neural gate, random routing and an oracle.

## Lexsi toolkits used (and what they did)
| Repo | Use | Evidence |
|---|---|---|
| AuditKIT | Paired comparison of per-question verdicts between views, giving 248 right-to-wrong and 115 wrong-to-right from Devanagari to Roman. This reproduces counts we also compute ourselves, so its role is light | `results/part3_results.json` |
| SafeTune | Its mean-difference linear probe, fitted on correct versus wrong Roman prompts (held-out AUROC 0.66, weaker than our logistic-regression probe at 0.84) | `results/part3_results.json` |
| CuratorKIT | Clean and near-duplicate screen of the question pool with provenance export. 2,400 questions in, 2,390 passed, 10 dropped as exact duplicates | `data/curatorkit/` |

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
