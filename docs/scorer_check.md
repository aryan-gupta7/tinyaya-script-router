# Scorer check (done by the AI assistant, not a human audit)

Sample: 45 random answers from the Devanagari view and 45 from the first Roman view (seed 5), compared with gold aliases by reading.

- Scorer agreed with the reviewer on 86 of 90.
- 1 clear scorer miss: the answer "The film was titled "8 Mile" (2002)." contains the gold answer but is scored wrong because the scorer needs the answer to match as a short string.
- 3 debatable cases: year-only answers to questions whose gold answer is a full date (for example "1987" against "1987-04-17").
- No case where a wrong answer was scored right.

Because the scorer is strict, all views are slightly underscored. A looser containment scorer keeps the gap (Devanagari 17.1% against Roman 10.1%, 281 against 113 discordant pairs; see `results/`). A human audit of this scorer has not been done and is listed as a limitation.
