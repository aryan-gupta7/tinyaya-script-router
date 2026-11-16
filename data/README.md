# Data files
- `pairs.jsonl`: 2,400 Mintaka questions with English, Devanagari (`dev`), the corrected Roman view (`rom_a2`), the first Roman view (`rom_a`, has a vowel bug), the noisy Devanagari controls (`noise3`, `noise6`), gold aliases, answer type, question type and question group. `rom_h` holds the typed Roman text for 150 questions.
- `generations/`: one file per view with the model's answer, a correct flag, token counts, first-token entropy and mean log-probability.
- `human_typed.json`: the 150 Roman Hindi questions typed by three teammates from the English question (anonymised).
- `curatorkit/`: provenance and manifest files written by CuratorKIT for the question pool (2,400 in, 2,390 passed, 10 dropped as exact duplicates).
- `../probes/`: logistic-regression correctness probe weights for layers 19 and 31 on the corrected Roman view (`coef`, `intercept`, feature `mean` and `scale`).
Hidden-state activations (about 35 MB at float16) are not stored here. They are uploaded to Hugging Face when possible.
