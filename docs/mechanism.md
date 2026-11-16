# 02 Mechanism: how everything works

Read this fully before writing code. Every term used in `03` and `04` is defined here.

## 1. The idea in one picture

```
Mintaka question (English) + Hindi translation
        |
        v  build views (T4)
  EN   DEV   ROM_A   ROM_B   NOISE   ROM_H (150 typed by people)
        |
        v  Tiny Aya, greedy, answer in English (T5, T8)
  correct/incorrect label per question per view    <-- gold = English answer + aliases
        |
        v  capture hidden state at last prompt token, chosen layers (T9)
  activations X_view[layer]   (before any answer token is generated)
        |
        +--> H1: McNemar and noise control on the labels (no probe needed)
        |
        +--> probes on ROM activations (T11)
              C_rom: will the romanized answer be correct?   score s_rom
              C_dev: would the Devanagari answer be correct?  score s_dev
        |
        v  Router (T12)
   if s_rom >= t1        -> ANSWER as typed
   elif s_dev >= t2      -> RE-RENDER to Devanagari, re-ask, ANSWER
   else                  -> ABSTAIN
        |
        v  compare with baselines at matched coverage (T12, T13)
```

## 2. Terms

- **Question group**: Mintaka questions sharing the same main question entity or template. All splits are done by group so near-duplicates never straddle train and test.
- **View**: the same question rendered differently.
  - EN: the English Mintaka question.
  - DEV: the Hindi (Devanagari) translation from Mintaka.
  - ROM_A: DEV transliterated to Roman script by a model-based transliterator written in casual style (preferred tool: AI4Bharat IndicXlit, UNVERIFIED that it installs; fallback: few-shot prompt to a non-Tiny-Aya LLM, see T4).
  - ROM_B: DEV transliterated by a rule-based scheme (for example ITRANS via the `indic-transliteration` package). Produces spellings nobody types, so it is a stress case, not the headline.
  - NOISE: DEV with light character-level noise (control for "any perturbation hurts"). Two levels: N3 (about 3 percent of characters edited) and N6 (about 6 percent). Edits: drop a matra, duplicate a character, swap adjacent characters, drop a space. Seeded.
  - ROM_H: human-typed Roman Hindi for a subset (at least 150 questions, at least 2 writers, writers type from the English or Hindi text freely, in the way they type on a phone).
  - ROM_headline: whichever of ROM_A, ROM_B is chosen by the day-one typed-vs-machine check (see `03`). If ROM_H has at least 150 items it is also reported as its own row.
- **Known set**: questions answered correctly in DEV. Used only for descriptive recovery numbers (never as the H1 test).
- **Discordant pairs** (DEV vs ROM): b = correct in DEV but wrong in ROM; c = wrong in DEV but correct in ROM. The H1 statistic compares b and c.
- **Recoverable question**: wrong in ROM and correct in DEV. **Reverse case**: correct in ROM and wrong after re-render (what blanket normalisation breaks).
- **Probe**: a logistic regression on the 2048-dim hidden state at one layer (layer count and width must be read from the model config in T2; do not trust the numbers in the notes).
- **Abstention head**: any scalar score used to decide to abstain. Candidates: probe score, first-token entropy, mean log-probability of the generated answer, agreement between two answers.

## 3. Prompt template (fixed, one template for all views)

System and chat formatting: use the tokenizer's chat template with `add_generation_prompt=True`. The template adds a system preamble of about 366 tokens (per SafeTune README). Test a minimal template in T2 (check A15). The same template is used for every view and every model call. Record which template was used.

User message text (English instruction, question in the view's script):

```
Answer the question with a short answer only: an entity name, a number, a date, or yes or no. Answer in English. Do not explain.
Question: {question_in_view}
Answer:
```

Decoding: greedy (`do_sample=False`), `max_new_tokens=24`, no sampling, batch with left padding. The probe reads the activation at the last prompt token (the final token of the templated prompt, the position from which the first answer token is predicted). Use the same extraction rule for every view and every layer.

## 4. Scoring (answers are compared in English)

- Normalise both strings: lowercase, strip, remove leading articles, remove punctuation, normalise numbers (digits, commas, number words one to ten), normalise yes or no.
- A prediction is correct if it equals any gold alias after normalisation, or if token-F1 against any alias is at least 0.8 AND the shorter string has at least 4 characters. Aliases come from Mintaka's English answer labels; add Wikidata English aliases only if the T3 label fetch works.
- Yes or no questions: exact match on the normalised first word.
- Report token-F1 as a secondary number.
- Scorer audit (T6): a team member hand-labels 100 random outputs per view as correct or not. Report agreement with the scorer per view with a CI. If any view is below 95 percent, report it and fix the scorer for all views, not for one.
- Reason for English answers: Mintaka answer text is English even for Hindi questions, and per-script alias matching could fake a script gap. This protocol measures comprehension and recall across scripts, not generation in script. Say so in the write-up.

## 5. Probes

Two probes per layer, both trained on activations of the ROM_headline prompts (this is what the router sees at run time):
- **C_rom**: label = answer correct in ROM_headline. Output s_rom, a calibrated probability.
- **C_dev**: label = answer correct in DEV (the twin of the same question). Output s_dev. This is the recoverability signal: it asks "does the model know this in Devanagari?" from the Roman prompt.

Training recipe (all choices made inside the training folds only):
- Standardise features, L2-regularised logistic regression, C chosen from {0.001, 0.01, 0.1, 1} by grouped inner cross-validation.
- Layer sweep: save activations for every second layer plus the last layer. Choose the layer on validation AUROC for C_rom and separately for C_dev. Report AUROC by layer for all layers.
- Calibration: fit Platt scaling on the validation split.
- H2c is evaluated on the subset where the ROM answer was wrong: AUROC of s_dev for predicting "DEV correct", with a paired bootstrap CI, within each question type and pooled.

**Non-neural control (required)**: a gradient-boosted or logistic classifier that sees only: Mintaka complexity type, answer type, question length in characters and tokens, token fragmentation ratio (tokens per word) of the ROM prompt, and an entity-popularity proxy (Wikidata sitelink count if the T3 fetch works, else the answer's English Wikipedia page existence or skip and say so). Probes only count as informative if they beat this control within question type.

## 6. The router (explicit rules)

Inputs per question: the ROM_headline prompt. Run one prefill with hidden-state capture, compute s_rom and s_dev. Parameters t1 and t2 are chosen on the validation split.

```
if s_rom >= t1:                       action = ANSWER          # answer the typed prompt
elif s_dev >= t2:                     action = RE_RENDER       # rewrite to Devanagari, re-ask, answer
else:                                 action = ABSTAIN
```

Choosing t1, t2: grid search on validation to maximise expected utility U = right_delivered - lambda * wrong_delivered, for lambda in {0.5, 1, 2, 4}, and also record the full coverage sweep for the risk-coverage curve. Report results at matched coverage versus baselines. Thresholds are frozen before the test split is run.

Re-render step options, all reported:
- `self`: ask Tiny Aya to rewrite the Roman-script question in Devanagari with this fixed prompt (same chat template, greedy, max_new_tokens=64):
  ```
  Rewrite the following question in Hindi using Devanagari script. Keep the meaning and every name exactly. Output only the rewritten question.
  Question: {romanized_question}
  ```
  The rewritten text is then fed to the normal answer prompt. Count the tokens of both calls in the cost.
- `translit`: dedicated transliterator (IndicXlit if installed).
- `oracle`: the gold DEV twin (an upper bound, not a deployable method).

Optional extension: after re-render, compute s_native on the re-rendered prompt and abstain if below t3.

## 7. Baselines (all get the same abstention-head treatment)

Every baseline is: an action policy plus one abstention head, with the head's threshold set on validation to hit the same coverage as the router. For each action policy, choose the best head among {probe C_rom or C_dev as appropriate, first-token entropy, mean log-probability} on validation. Action policies:
1. **Always answer** the typed ROM prompt.
2. **Always normalise**: always re-render then answer (no abstention in the pure form, plus the abstention-head variant).
3. **Always English pivot**: translate the Roman question to English with Tiny Aya, answer in English, plus head. (Cost counted.)
4. **Random routing** at the same re-render budget as the router.
5. **Verbalised trigger**: prompt the model to say it is unsure; use that as the trigger for re-render (the output-triggered style of prior work).
6. **Agreement gate**: answer ROM and re-rendered prompts, abstain if the two answers disagree (the consensus style). Cost includes both passes.
7. **Non-neural gate**: the control classifier from section 5 as the router's signal.
8. **Oracle router**: knows the DEV and ROM labels; the upper bound.
Also report: always answer with abstention by length heuristic and by perplexity (the brief asked for these two heuristics).

## 8. Metrics (definitions)

Let each question end with action in {ANSWER, RE_RENDER, ABSTAIN} and, if not abstained, a delivered answer correct or wrong.
- **Accuracy by view**: fraction correct, per view, with grouped bootstrap CI.
- **Discordance table**: counts b and c for DEV vs ROM, DEV vs NOISE, DEV vs EN.
- **Script effect**: Delta_script = acc(DEV) - acc(ROM_headline). Delta_noise = acc(DEV) - acc(NOISE at matched edit rate). Excess = Delta_script - Delta_noise with paired bootstrap CI.
- **Recoverable and reverse slice**: counts and fractions of all questions and of the known set.
- **Coverage** = fraction not abstained.
- **Right delivered** = right answers delivered / all questions. **Wrong delivered** = wrong answers delivered / all questions.
- **Selective accuracy at coverage x**: accuracy among delivered answers at coverage 50, 70, 90 percent.
- **AURC**: area under the risk-coverage curve (risk = error rate among delivered answers), lower is better.
- **Probe**: AUROC with CI, expected calibration error, score shift between views.
- **Cost**: tokens processed = prompt tokens plus generated tokens per query, summed over every call the policy makes; also measured wall-clock latency per query on the same GPU. The probe reuses the prefill that answering needs anyway, so it adds no extra tokens. Pass counting is by tokens, stated in advance (this removes the ambiguity in "extra passes").
- **Utility**: U = right_delivered - lambda * wrong_delivered for lambda in {0.5, 1, 2, 4}.

All CIs: 2000 paired bootstrap resamples, resampling question groups. All pairwise significance for binary labels: exact McNemar.

## 9. What we will and will not claim

We may claim, if the numbers support it: a measured script effect net of noise on Tiny Aya Global for Hindi factual QA with English answers; whether pre-generation hidden state carries recoverability information beyond a non-neural control; the router's wrong and right delivered numbers at matched coverage with its token cost.

We will not claim: that the router fixes comprehension; a mechanism of why Roman script fails; results for other languages or models; clinical safety; that the dataset is uncontaminated (Tiny Aya's training data is not public; Mintaka and Global-MMLU are plausibly seen; this inflates the known set and the probe may learn memorisation strength, so state it as a limitation); novelty of selective script routing in general (prior work routes by output label on large models; ours is a pre-generation internal signal, factual QA, a 3.35B on-device model, with an abstain branch).
