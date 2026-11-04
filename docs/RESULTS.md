# Results: held-out run A

30 prompts written by Muse Spark without seeing our calibration prompts: 10 low-key night shots, 10 bright exteriors,
10 neutral. Seeds, prompts, pass marks and the judge's code were frozen before the GPU run (pre-registered 2 Nov 2026,
15:19 CET; run on 3 Nov). LTX-2.5, 8 steps, one H100 on Modal; up to 3 attempts per shot.

| | first drafts | all drafts |
|---|---|---|
| drafts | 30 | 44 |
| real failures | 12 | 18 |
| passing | 18 | 26 |
| false stops | 0 | 1 |
| missed | 2 | 4 |
| caught by step 1 | 1 | 2 |
| caught by step 3 | 3 | 4 |
| caught by step 5 | 8 | 11 |

- Without the gate, first drafts passing: 18 of 30. With gate and director, delivered and passing: **25 of 30**.
- Pass marks, frozen before the run: false stops ≤ 1: **met**; caught by step 3 ≥ 50 %: **not met** (25 %).
- Why two failures were missed: both were night shots that failed by a few points, inside the judge's margin of error
  at step 3. Narrowing that margin on night shots needs a new calibration and a fresh held-out run, not these 30.
- GPU time: 368 s spent, 19 s (5 %) saved by stopping drafts early. Director calls: 521 s waiting on Muse Spark.
  Replayed on run A's own answers, this week's cheaper director needs 9 calls instead of 45 (268 s).
- Cost of the whole run: about $1.75 of GPU.

## The steer on h05

Prompt: "Moths circle a dim porch bulb outside a dark house at night." Judged as a low-key night shot (dark is
intended, pure black is not).

| | crushed blacks | exposure | result |
|---|---|---|---|
| draft, predicted at step 3 | 54 % (limit 20 %) | 0.012 | stopped |
| draft, left to finish (checked only) | 76.4 % | 0.018 | fails |
| steered clip, delivered | 0.01 % | 0.093 | passes every check |

Muse Spark's reason at the stop, verbatim: "crush_pct predicted 54.0 vs 20.0 max misses by more than tolerance 24.5
and is worsening 42% to 49.5% to 54% with exposure collapsing, so mud will not self-correct." About $0.20 of GPU.
