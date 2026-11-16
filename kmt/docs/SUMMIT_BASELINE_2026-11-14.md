# Summit Baseline Measurements

Measured with the existing `python evaluate.py --data data --ckpt checkpoints` command on 2026-11-14. This is the pre-Qwen 721,536-parameter NumPy checkpoint.

| Metric | Value |
|---|---|
| Language ID, Krio | 0.80 |
| Language ID, English | 0.93 |
| Language ID, Mende | 0.97 |
| Language ID, Temne | 0.75 |
| Translation chrF, Krio -> English | 0.094 |
| Translation chrF, English -> Krio | 0.093 |
| Translation chrF, Mende -> English | 0.095 |
| Translation chrF, English -> Mende | 0.055 |
| Translation chrF, Temne -> English | 0.060 |
| Translation chrF, English -> Temne | 0.105 |
| Next-token top-1, Krio / English / Mende / Temne | 0.23 / 0.21 / 0.18 / 0.19 |

The system calibrates unseen translation to abstain. These results are the original baseline and must not be presented as Qwen results. The Qwen evaluation script reports small-sample overlap metrics; human conversation quality and the listed domain benchmarks still require approved, held-out tests.