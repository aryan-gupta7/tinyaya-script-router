# Typed in Roman, known in Devanagari

A probe-gated script router for Cohere Tiny Aya (`CohereLabs/tiny-aya-global`).
Entry for the Lexsi AI Safety and Alignment Hackathon, Track 2b-13 (hallucination detection and abstention).

**Status: results are being generated. This README will be updated with the numbers from `results/` once the run finishes. Nothing below reports a result.**

## Question
Many Indian users type Hindi in Roman script ("bharat ki rajdhani kya hai"). We ask:
1. Does Tiny Aya answer benign factual questions less accurately when they are typed in Roman script than in Devanagari, after controlling for plain typo noise?
2. Does the model's hidden state at the last prompt token, before it generates, reveal which Roman-script questions it would answer correctly in Devanagari?
3. Does a router that uses that signal (answer as typed, rewrite to Devanagari and re-ask, or abstain) deliver fewer wrong and more right answers than fair baselines at matched coverage?

No fine-tuning is used.

## Method
- Data: Mintaka (Hindi and English, CC-BY-4.0). Each question has views: Devanagari (DEV), Roman (ROM_A, rule-based ITRANS plus casual-spelling cleanup), and two noisy Devanagari controls (3 and 6 percent character edits). A human-typed Roman subset is collected separately.
- Model prompt: one fixed template, greedy decoding, the answer is always requested in English and scored against English gold aliases (`code/common.py`).
- Gap test: exact McNemar test over all question pairs, with the noise control and grouped bootstrap intervals.
- Probes: logistic regression on last-prompt-token hidden states, group-aware train, validation and test splits. One probe predicts whether the Roman answer is correct, another whether the Devanagari answer would be correct. A gradient-boosted control sees only question type, length, tokenization fragmentation and entity count.
- Router: answer if the first score passes a threshold, else re-render if the second passes its threshold, else abstain. Re-rendering is done by Tiny Aya itself. Baselines (always answer, always rewrite, entropy, mean log-probability, non-neural gate, random routing, oracle) use the same abstention treatment at matched coverage.
- Details: `docs/mechanism.md`, hypotheses and decision rules: `docs/preregistration.md`, decisions made along the way: `DECISIONS.md`.

## Repository layout
```
code/common.py          scoring, Mintaka loading, view construction
code/gen_main.py        generation and hidden-state capture on Kaggle
code/analysis.py        gap test, probes, control, router and baselines
code/kaggle_run_all.py  the single script run on Kaggle (common + gen_main + analysis)
docs/                   mechanism and preregistration
results/                outputs (added after the run)
```

## Reproduce
1. On Kaggle, attach the model `cohere-labs/tiny-aya` (Transformers, variation `global`), enable a T4 GPU and internet, and run `code/kaggle_run_all.py` as a script. Defaults: 2400 questions, seed 1234, about one hour.
2. Outputs (generations, hidden states, `results/*_analysis.json`) are written to `/kaggle/working`.
3. `python code/analysis.py <out_dir>` reruns the analysis from the saved files.

## Limitations
Mintaka may appear in Tiny Aya's training data. Romanization is machine generated apart from a small human-typed subset. One language and one model are tested. Answers are requested in English, so the test measures comprehension and recall, not generation in Roman script. The probe is correlational, so no mechanism is claimed. This was built in a few hours: CircuitKIT, `tiny-aya-fire`, a 4-bit test and a demo Space were dropped.

## Licences
Code: MIT. Tiny Aya is CC-BY-NC with an acceptable use policy, so use of the model outputs stays non-commercial. Mintaka is CC-BY-4.0.
