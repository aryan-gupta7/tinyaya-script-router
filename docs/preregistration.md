# 03 Pre-registration (freeze before touching the test split)

Instructions:
1. Fill the blanks marked `[FILL]` from the pilot (T2 to T6) only. The pilot uses a separate set of 300 questions that is excluded from every later split.
2. Commit this file with the message `prereg frozen` and record the commit hash in `DECISIONS.md`.
3. After that, any change to this file is a deviation and must be listed under "Deviations" at the bottom with the reason.

## 1. Data and splits
- Source: Mintaka Hindi parallel questions (`AmazonScience/mintaka`, config for Hindi plus English). Load and count the real number of rows in T3. The review reports 20,000 questions per language with 14k train, 2k dev, 4k test per the card, and warns the dataset page counts are inconsistent.
- Pilot: 300 questions, random, excluded afterwards.
- Main pool size N_main = `[FILL]`. Rule: choose N_main so the expected number of recoverable questions (wrong in ROM, correct in DEV) is at least 400 in the whole pool. Estimate the rate from the pilot. Cap at 10000 because of compute. If the cap does not reach 400 recoverable, state it and downgrade H2c to exploratory.
- Split by question group (group = first question entity Q-id; fall back to question id): train 60 percent, validation 20 percent, test 20 percent, by hash of group id with seed 1234. All views of one question are in the same split. Record split membership in `data/splits.json`.
- Robustness: 5-fold grouped cross-validation over the whole pool for AUROC CIs. Headline numbers use the single test split.
- Question types: Mintaka complexity type. If a type has native (DEV) accuracy below 2 percent in the pilot, report it but exclude it from the policy comparison (state which).

## 2. Headline romanization (fixed by the day-one check A7)
- ROM_H: at least 30 team-typed questions (target 150) matched to machine versions of the same questions.
- Rule: ROM_headline = the machine method (ROM_A or ROM_B) whose accuracy on those matched questions is closest in absolute value to the accuracy on the human-typed versions. Ties go to ROM_A. This choice is made before looking at any gap on the main pool.
- If at least 150 human-typed questions exist at the end, report a ROM_H row in every table.

## 3. Hypotheses and decision rules

**H1 (a script effect exists, net of noise).**
- Statistic: exact McNemar test on discordant pairs b vs c for DEV vs ROM_headline, over ALL questions in the main pool (not only the known set).
- Noise control: the same test and accuracy difference for DEV vs NOISE (levels N3 and N6). Excess = Delta_script - Delta_noise with 95 percent grouped bootstrap CI.
- Accept H1 only if: McNemar p < 0.05 with b > c AND the CI of the excess excludes 0.
- The known-set recovery numbers (what fraction of DEV-correct questions fail in ROM) are descriptive only, with the caveat that selection on DEV correctness inflates them.

**H2a (separability survives).** C_rom AUROC on the test split for ROM_headline prompts is at least 0.70 and within 0.05 of C_rom trained and tested in DEV. Reported with CI. Not a gate.

**H2b (score shift).** Mean score shift between DEV and ROM prompts, and ECE of C_rom on ROM before and after calibration on a small validation set. Descriptive.

**H2c (recoverability is readable before generation).**
- Population: questions wrong in ROM_headline. Label: correct in DEV. Signal: s_dev from C_dev on ROM prompts.
- Accept only if test AUROC at least 0.65 with the lower CI bound above 0.55, AND it beats the non-neural control by at least 0.03 AUROC within question type (paired bootstrap), AND the test subset has at least 150 positives and 150 negatives. If counts are lower, label H2c exploratory.
- Interpretation allowed: "recoverability is (or is not) readable before generation in this model". No mechanism wording.

**H3 (policy wins).** On the test split, at matched coverage (compare at coverage 60, 75, 90 percent):
- Router wrong_delivered is lower than the best baseline among {always answer + head, always normalise + head, always pivot + head, verbalised trigger, agreement gate, non-neural gate}, and
- Router right_delivered is not lower than always answer + head, and
- Router token cost is at most 0.6 times that of always normalise (or state the actual ratio).
- Accept H3 strictly only if the first two conditions hold with the paired bootstrap CI of the difference excluding 0. Otherwise report "no significant advantage" and the numbers.

**H4 (extension, only if time).** The fp16-trained probe keeps AUROC within 0.03 on NF4 activations (Kaggle T4, bitsandbytes). Exploratory: a mean native-minus-romanized activation-difference steering vector at one layer recovers part of the gap without a second pass.

**H5 (mechanism, only if the patching experiment is done).** Patching the DEV residual stream at layer L into the ROM run raises ROM accuracy toward DEV accuracy. Without this experiment no comprehension-vs-action claim is made.

## 4. Switch rules (decided in advance)

| Observation | Action |
|---|---|
| DEV accuracy (native Hindi, English answers) below 5 percent on the pilot | Enlarge pool to 1000; restrict to question types with at least 10 percent DEV accuracy; if still below 5 percent, switch to passage-grounded QA with a Roman question over a Devanagari passage (XQuAD-hi or IndicQA-hi, licences UNVERIFIED) and say so in `DECISIONS.md` |
| H1 not accepted on the pilot (b < 30 or p >= 0.05) | Enlarge once to 1000 questions. If still not accepted, the headline becomes "Tiny Aya's factual knowledge is script-robust"; report the dissociation with prior refusal results as a contrast; the router runs on the English vs Hindi gap using the EN view already collected |
| Probe AUROC at or below the non-neural control | Report as negative result; the router uses the best of entropy, mean log-probability or agreement gate as signal |
| Self-rewrite character error rate against the gold Devanagari twin above 30 percent | Use the dedicated transliterator if it installs; always report oracle as the bound |
| Scorer agreement below 95 percent in any view | Fix the scorer for all views and rerun scoring (no model rerun needed) |
| fp16 produces non-finite logits | Run in fp32 on two T4 (device_map auto) or in 4-bit NF4; record it and rerun the pilot |

## 5. Statistical rules
- Seeds: 1234 for splits, 1 to 5 for bootstrap and cross-validation.
- No test-split use for: layer choice, regulariser, calibration, thresholds, scorer edits.
- Multiple comparisons: the confirmatory hypotheses are H1, H2c, H3 (three). Everything else is labelled exploratory.
- Report every pre-registered test even when null.

## 6. Pilot values to fill after T2 to T6
| Quantity | Value |
|---|---|
| Kaggle Python version | [FILL] |
| Number of transformer layers, hidden size, from config.json | [FILL] |
| Template used (default or minimal) and prompt length in tokens | [FILL] |
| Pilot DEV accuracy / ROM_A / ROM_B / EN / NOISE N3 / N6 | [FILL] |
| Pilot b and c for DEV vs ROM_headline | [FILL] |
| Recoverable rate (fraction of all questions) | [FILL] |
| N_main | [FILL] |
| Headline romanization method | [FILL] |
| Throughput: generations per minute on one T4 | [FILL] |

## 7. Deviations
(append: date, what changed, why, who decided)
