# FlowTTA: Label-Free Test-Time Adaptation for Time Series Foundation Models

**Status: Feasibility Round 2 Complete — WEAK GO**

## The Idea

Time series foundation models (TSFMs) are deployed zero-shot but degrade under distribution shift. All existing test-time adaptation (TTA) methods for time series require ground-truth labels. We propose **label-free TTA** by training a tiny adapter on the TSFM's own embeddings using self-supervised losses that exploit temporal structure. The adapter is trained *at test time* on each batch of shifted data, then discarded.

```
Shifted test window → [Frozen TSFM Encoder] → shifted embedding z
                                                      ↓
                                              [Tiny Adapter θ] ← trained with self-supervised losses
                                                      ↓
                                              adapted embedding z'
                                                      ↓
                                        [Frozen TSFM Head] → better prediction
```

---

## Round 2 Results (Rigorous Evaluation)

**Setup:** Chronos-T5-Tiny, ETTh1 (OT column), 30 test windows, context=512, prediction=24, 20 forecast samples, **3 random seeds** (mean +/- std). All results report MSE.

### Experiment 1: Does distribution shift degrade FMs?

**YES — massively.** Clean baseline MSE: 2.76.

| Shift Type | Mag 1.0 | Mag 2.0 | Mag 3.0 |
|------------|---------|---------|---------|
| Mean       | +192%   | +716%   | +1540%  |
| Variance   | -4%     | +180%   | +835%   |
| Trend      | +166%   | +604%   | +1371%  |

### Experiment 2: Loss Ablation (mean shift, mag=2.0)

Fixed losses: temporal (multi-window), reconstruction (FM leave-last-out), entropy (new). Added RevIN oracle baseline.

| Config | MSE (mean +/- std) | vs Zero-Shot | vs RevIN |
|--------|-------------------|--------------|----------|
| **Zero-shot** | **21.94 +/- 0.05** | baseline | -3.3% |
| **RevIN oracle** | **21.25 +/- 0.11** | **+3.2%** | baseline |
| Temporal | 22.60 +/- 0.34 | -3.0% | -6.4% |
| Spectral | 22.02 +/- 0.62 | -0.4% | -3.6% |
| **Reconstruction** | **21.71 +/- 0.47** | **+1.1%** | -2.2% |
| Entropy | 21.97 +/- 0.50 | -0.1% | -3.4% |
| Temporal+Entropy | 22.05 +/- 0.37 | -0.5% | -3.8% |
| All four | 21.82 +/- 0.42 | +0.6% | -2.7% |

**Key finding:** Reconstruction loss (leave-last-out) is the only loss that consistently improves over zero-shot (+1.1%), but it doesn't beat RevIN. The other losses either hurt or are neutral. Combining losses doesn't help — temporal loss drags down the signal.

### Experiment 3: Adapter Comparison (mean shift, mag=2.0)

Using the best loss from Exp 2 for each seed.

| Adapter | Params | MSE (mean +/- std) | vs Zero-Shot | vs RevIN |
|---------|--------|-------------------|--------------|----------|
| **Zero-shot** | 0 | **21.79 +/- 0.09** | baseline | -1.7% |
| **RevIN** | 0 | **21.43 +/- 0.49** | **+1.6%** | baseline |
| Segmented | 16 | 22.21 +/- 0.36 | -1.9% | -3.6% |
| MLP | 97 | 22.11 +/- 0.07 | -1.5% | -3.2% |

**Key finding:** Neither adapter beats zero-shot when evaluated rigorously. The Round 1 "+23.6% improvement" for MLP was an artifact of 8-window high variance. With 30 windows and 3 seeds, adapters slightly hurt.

### Experiment 4: Trend Shift (mag=2.0)

Testing if the method generalizes beyond mean shift.

| Config | MSE (mean +/- std) | vs Zero-Shot | vs RevIN |
|--------|-------------------|--------------|----------|
| **Zero-shot** | **19.94 +/- 0.22** | baseline | -10.1% |
| **RevIN** | **18.11 +/- 0.16** | **+9.2%** | baseline |
| Segmented | 19.86 +/- 0.37 | +0.4% | -9.7% |
| MLP | 19.61 +/- 0.53 | +1.6% | -8.3% |

**Key finding:** RevIN dominates on trend shift (+9.2%). Our adapters show marginal improvement that doesn't beat the simplest baseline.

---

## Assessment: WEAK GO (Honest)

### What works
1. **The problem is real and severe**: FMs degrade 200-1500% under distribution shift
2. **Reconstruction loss (leave-last-out) shows signal**: +1.1% vs zero-shot — the FM's own predictions on withheld context provide genuine self-supervision
3. **The experimental infrastructure is solid**: Reproducible, seeded, properly evaluated

### What doesn't work
1. **No method beats RevIN**: Simple instance normalization (zero parameters, no learning) outperforms all our self-supervised adapters
2. **Temporal consistency loss hurts**: Even with the multi-window fix, it degrades predictions (-3.0%)
3. **Entropy minimization is neutral**: Minimizing forecast spread doesn't improve point accuracy
4. **Adapter complexity doesn't help**: MLP (97 params) performs comparably to segmented affine (16 params), and both lose to zero parameters (RevIN)

### Why the idea struggles
The fundamental issue: **self-supervised signals in embedding/input space don't align with prediction accuracy.** The losses optimize for internal consistency (temporal agreement, spectral preservation, reconstruction) but these objectives are orthogonal to — or even at odds with — producing accurate forecasts under shift. RevIN works because it directly addresses the distribution mismatch at the input level with a known-correct normalization, not because it learns from self-supervised signals.

### What would need to change for a paper
1. **The losses need to be predictive, not just consistent**: Reconstruction loss (leave-last-out) is the closest because it uses the FM's own predictions as signal. This direction could be amplified — e.g., multi-scale leave-out, contrastive objectives between adapted and unadapted predictions
2. **Must demonstrably beat RevIN**: Without this, reviewers will (correctly) point out that the simplest baseline solves the problem
3. **Natural shifts, not synthetic**: Synthetic shifts are convenient but may not represent real deployment scenarios. TTFBench or DRIFT datasets needed
4. **Multiple FMs**: Results on one tiny model are insufficient

### NeurIPS Spotlight Viability
**Unlikely in current form.** The core contribution (self-supervised losses for TTA) doesn't outperform a zero-parameter baseline. The reconstruction loss direction has potential but needs significant development. This could become a workshop paper or a component of a larger system, but not a standalone spotlight contribution.

---

## Round 1 Results (Preliminary, 8 windows — superseded)

<details>
<summary>Click to expand Round 1 results</summary>

Round 1 used 8 windows, 5 samples, no seeding. Results showed +23.6% for MLP adapter, but this was due to high variance from small sample size. See `results.json` for raw Round 1 data.

</details>

---

## Project Structure

```
FlowTTA/
├── config/default.yaml          # Hyperparameters
├── data/
│   ├── load_etth.py             # ETTh1 dataset loader
│   └── synthetic_shift.py       # Synthetic distribution shift functions
├── models/
│   ├── fm_wrapper.py            # Chronos wrapper (encode/decode/predict)
│   ├── adapters.py              # Affine, MLP, Flow, Input adapters
│   └── losses.py                # Self-supervised losses
├── experiments/                 # Individual experiment scripts (Round 1)
├── evaluate.py                  # MSE, MAE, relative improvement
├── run_fast.py                  # Round 2 experiment runner (all fixes)
├── results.json                 # Round 1 raw results
└── results_r2.json              # Round 2 raw results (3 seeds)
```

## Setup

```bash
conda create -n flowtta python=3.10 -y
conda activate flowtta
pip install torch torchvision torchaudio
pip install transformers accelerate safetensors
pip install gluonts datasets pandas matplotlib scipy einops tqdm
pip install chronos-forecasting

# Run Round 2 experiments (~2 hours on GPU)
python run_fast.py
```
