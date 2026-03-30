# FlowTTA: Label-Free Test-Time Adaptation for Time Series Foundation Models

**Status: Round 4 Complete — NO-GO for method paper, pivoting to analysis contribution**

## The Idea

Time series foundation models (TSFMs) are deployed zero-shot but degrade under distribution shift. We explored **label-free TTA** by training a tiny adapter using self-supervised losses (multi-horizon reconstruction) at test time. Across 4 rounds of increasingly rigorous experiments, we found that **the method doesn't reliably add value over simpler baselines** — but the analysis of when normalization fails for TSFMs is a genuine contribution.

---

## Round 4: Definitive Results (16hr, chronos-t5-small)

**Setup:** Chronos-T5-Small (46M params), ETTh1, 80 windows, 20 samples, 5 seeds, MLP adapter (97 params), multi-horizon reconstruction loss (K=12,24,48).

### R4.1: All 8 Shift Types (mean +/- std over 5 seeds)

| Method | Mean | Variance | Trend | Trend-Intra | Frequency | Autocorr. | Outlier |
|--------|------|----------|-------|-------------|-----------|-----------|---------|
| Zero-shot | 20.30+/-0.17 | 7.39+/-0.11 | 18.66+/-0.11 | 17.50+/-0.11 | 10.87+/-0.11 | 2.11+/-0.05 | 2.23+/-0.10 |
| RevIN | 19.54+/-0.13 | 7.21+/-0.17 | 18.21+/-0.14 | 16.37+/-0.13 | 10.09+/-0.09 | 2.26+/-0.01 | 2.29+/-0.07 |
| **Ours** | 20.30+/-0.25 | **7.18+/-0.12** | 18.76+/-0.24 | 17.60+/-0.13 | 11.03+/-0.20 | **2.10+/-0.07** | **2.21+/-0.04** |
| RevIN+Ours | **19.47+/-0.21** | 7.47+/-0.16 | **18.18+/-0.12** | **16.25+/-0.11** | 10.15+/-0.03 | 2.26+/-0.06 | 2.35+/-0.08 |

**vs Zero-Shot (%):**

| Method | Mean | Variance | Trend | Trend-Intra | Frequency | Autocorr. | Outlier |
|--------|------|----------|-------|-------------|-----------|-----------|---------|
| RevIN | +3.7% | +2.4% | +2.4% | +6.4% | +7.2% | **-7.0%** | -2.7% |
| Ours | +0.0% | **+2.8%** | -0.6% | -0.6% | -1.4% | **+0.5%** | **+0.9%** |
| RevIN+Ours | **+4.1%** | -1.0% | **+2.5%** | **+7.1%** | +6.7% | -7.1% | -5.1% |

**Worst-case degradation:**
- Zero-shot: 0.0% (baseline)
- RevIN: **-7.0%** (autocorrelation)
- **Ours: -1.4%** (frequency) — best worst-case
- RevIN+Ours: -7.1% (autocorrelation)

### R4.4: Cross-Dataset Natural Shift

| Method | ETTh1 | ETTh2 | Weather |
|--------|-------|-------|---------|
| Zero-shot | 2.16 | 23.48 | 1.03 |
| RevIN | 2.29 | **21.57** | 1.14 |
| **Ours** | **2.09** | 24.37 | **1.04** |
| RevIN+Ours | 2.35 | **21.56** | 1.17 |

**vs Zero-Shot:**

| Method | ETTh1 | ETTh2 | Weather |
|--------|-------|-------|---------|
| RevIN | -5.6% | **+8.1%** | -10.8% |
| **Ours** | **+3.3%** | -3.8% | -0.4% |
| RevIN+Ours | -8.6% | +8.2% | -13.6% |

---

## What We Learned (The Actual Contribution)

### 1. RevIN Has Systematic Failure Modes for TSFMs

RevIN helps on moment-based shifts (mean +3.7%, frequency +7.2%) but **hurts on autocorrelation (-7.0%), ETTh1 in-domain (-5.6%), and Weather (-10.8%)**. This is poorly documented in the TSFM literature.

| Shift Type | RevIN Handles? | Empirical Result |
|------------|---------------|------------------|
| Mean shift | Yes | +3.7% |
| Variance scaling | Yes | +2.4% |
| Trend | Partially | +2.4% |
| Intra-window trend | Partially | +6.4% |
| Frequency change | No (but helps) | +7.2% |
| Autocorrelation | **No — HARMFUL** | **-7.0%** |
| Outlier injection | **No** | -2.7% |
| In-domain (ETTh1) | **HARMFUL** | **-5.6%** |
| Cross-domain (Weather) | **HARMFUL** | **-10.8%** |

### 2. Self-Supervised Reconstruction Has the Best Worst-Case

Our method's worst-case is only -1.4% (vs RevIN's -7.0%). It's the **safest** approach. On in-domain data (ETTh1), ours actually improves by +3.3% while RevIN degrades -5.6%.

### 3. The Two Methods Are Complementary But Stacking Doesn't Help

RevIN+Ours beats RevIN on mean (+0.3%), trend (+0.2%), and trend-intra (+0.7%), but inherits RevIN's failure modes on autocorrelation and outlier. The stacking doesn't provide the "best of both worlds" we hoped for.

### 4. The Self-Supervised Signal Is Real But Weak

Multi-horizon reconstruction produces genuine gradient signal (+2.8% on variance, +0.5% on autocorrelation, +0.9% on outlier, +3.3% on in-domain). But the gains are consistently modest and don't exceed RevIN's gains on the shifts where RevIN works.

---

## Honest Assessment

### Why This Is NOT a Method Paper

The self-supervised adaptation method doesn't produce strong enough improvements. On the 7 synthetic shifts, "ours" beats zero-shot on only 3 shifts by tiny margins (0.5-2.8%). RevIN, despite its failure modes, has larger positive effects where it works.

### What IS Publishable

**An analysis paper on normalization failure modes for TSFMs.** The finding that RevIN hurts on in-domain data (-5.6%), weather (-10.8%), and autocorrelation (-7.0%) is novel and practically important. Practitioners currently apply RevIN by default — our results show this can be actively harmful.

Possible framing: *"When Does Normalization Help? A Systematic Study of Test-Time Preprocessing for Time Series Foundation Models"*

### Recommended Next Steps

1. **Workshop paper** at NeurIPS 2026 Time Series Workshop or ICML PUT Workshop focusing on the analysis contribution
2. **Expand analysis** to 3+ FMs (Moirai, Sundial, TimesFM) to strengthen the normalization failure finding
3. **If pursuing method further**: the embedding-level approach showed +8% on one seed in Round 3 — this direction is unexplored and could be the technical novelty for a future full paper

---

## Experiment History

| Round | Model | Windows | Seeds | Key Finding |
|-------|-------|---------|-------|-------------|
| R1 | Chronos-T5-Tiny | 8 | 1 | +23.6% (variance artifact) |
| R2 | Chronos-T5-Tiny | 30 | 3 | +1.1% best, nothing beats RevIN |
| R3 | Chronos-T5-Tiny | 30 | 3 | RevIN fails on outlier (-14%), cross-dataset (-52%) |
| **R4** | **Chronos-T5-Small** | **80** | **5** | **Ours: best worst-case (-1.4% vs -7.0%). Not enough for method paper.** |

## Setup

```bash
conda create -n flowtta python=3.10 -y
conda activate flowtta
pip install torch torchvision torchaudio
pip install transformers accelerate safetensors scipy
pip install gluonts datasets pandas matplotlib einops tqdm
pip install chronos-forecasting

# Run Round 4 (~16 hours on GPU)
python run_round4.py
```
