# Scorer check (done by the AI assistant, not a human audit)

Sample: 45 random answers from the Devanagari view and 45 from the first Roman view (seed 5). The assistant compared the scorer's verdict with a simple containment test (does the gold answer appear in the output) and read every disagreement.

Findings in this sample of 90:
- 3 answers were correct but scored wrong because the model answered in a full sentence: "The second game in the Star Fox series is Star Fox 64." (Devanagari view), and "The Sims 4: Get to Work" and "8 Mile" inside sentences (Roman view). The strict scorer misses these.
- 2 answers contained a gold string but were correctly scored wrong ("South Korea and Japan" for North and South Korea, "Democratic Republic of the Congo" for republic).
- 4 answers scored right without containing the gold string, all short ("13", "No."), which is intended.
- No wrong answer was scored right.

Consequence: the strict scorer slightly under-scores sentence-style answers, and in this small sample the misses were more frequent in the Roman view. This could inflate the script gap a little. A looser containment scorer was run on all 2,400 questions: Devanagari 17.1%, Roman 10.7%, 270 against 116 discordant pairs, so the gap remains. A human audit of the scorer has not been done and is listed as a limitation.
