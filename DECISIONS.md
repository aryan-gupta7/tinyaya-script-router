# Decisions log
- 2026-11-16: model = CohereLabs/tiny-aya-global (Kaggle Models copy). Data = Mintaka (Hindi + English), loaded from the official GitHub files.
- Answers requested in English in every view and scored against English gold aliases.
- Romanization: ROM_A = rule-based ITRANS followed by a casual-spelling cleanup (IndicXlit not installed). ROM_B = plain ITRANS. Headline view = ROM_A. A human-typed subset (150 questions) is collected separately.
- Time-limited run: about 2000 Mintaka questions in one Kaggle session. No fine-tuning.
- Dropped for time and not run: CircuitKIT analysis, tiny-aya-fire, 4-bit probe test, Global-MMLU-Lite anchor, Space demo.
