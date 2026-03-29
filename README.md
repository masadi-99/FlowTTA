# FlowTTA: Label-Free Test-Time Adaptation for Time Series Foundation Models

**Status: Feasibility Study Complete — WEAK GO**

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

## Feasibility Experiment Results

**Setup:** Chronos-T5-Tiny, ETTh1 dataset (OT column), 8 test windows, context=512, prediction=24, 5 forecast samples.

### Experiment 1: Does distribution shift degrade FMs?

**YES — massively.** This validates the core premise.

| Shift Type | Mag 1.0 | Mag 2.0 | Mag 3.0 |
|------------|---------|---------|---------|
| Mean       | +99%    | +333%   | +797%   |
| Variance   | -33%    | +145%   | +692%   |
| Trend      | +116%   | +350%   | +717%   |

Clean baseline MSE: 5.95. Mean shift at magnitude 3.0 pushes MSE to 53.4.

### Experiment 2: Do self-supervised losses help individually?

**NO — all individual losses hurt performance.**

| Configuration    | MSE     | vs Zero-Shot |
|-----------------|---------|--------------|
| Zero-shot       | 20.63   | baseline     |
| Temporal only   | 23.70   | -14.9%       |
| Spectral only   | 28.74   | -39.3%       |
| Reconstruction  | 25.11   | -21.7%       |
| All three       | 26.03   | -26.2%       |

The self-supervised losses optimize for internal consistency but not prediction accuracy. This is the **Fallback A** scenario: losses decrease during adaptation but MSE doesn't improve.

### Experiment 3: Does adapter type matter?

**MLP adapter shows +23.6% improvement**, but with caveats.

| Adapter    | Params | MSE   | vs Zero-Shot | ms/batch |
|------------|--------|-------|--------------|----------|
| Zero-shot  | 0      | 35.24 | baseline     | 0        |
| Affine     | 2      | 31.44 | +10.8%       | 132      |
| MLP        | 97     | 26.93 | +23.6%       | 99       |

Note: Exp 2 and Exp 3 zero-shot MSEs differ (20.63 vs 35.24) despite same shift config, indicating high variance with only 8 windows.

## Assessment: WEAK GO

### What works
- **The problem is real**: FMs degrade severely under shift (up to 800% MSE increase)
- **Input-level adaptation has potential**: MLP adapter showed +23.6% improvement in one experiment
- **Adaptation is fast**: ~100ms per batch overhead

### What doesn't work
- **Self-supervised losses are unreliable**: No individual loss consistently improves predictions
- **High variance**: Results swing significantly between runs with only 8 windows
- **Novelty concern**: If input-level affine adaptation is the main mechanism, this is essentially learned RevIN (reversible instance normalization) — a well-known technique

### Recommended next steps
1. **Try entropy minimization** on FM's output distribution (Fallback A from plan) — minimize spread of Chronos's sampled forecasts
2. **Scale up evaluation** with GPU access: more windows, larger model (chronos-t5-small), more samples
3. **Try embedding-level adaptation** when GPU memory allows — this is the original proposal
4. **Test on natural shifts** using TTFBench datasets instead of only synthetic shifts
5. **If input-level affine is all that works**, the paper needs repositioning — the contribution would be the self-supervised loss design, not the adapter architecture

### Key risk for NeurIPS
The self-supervised losses (temporal consistency, spectral consistency, masked reconstruction) are proposed as the core novelty, but they don't improve predictions in initial tests. If entropy minimization or simple input normalization is what actually works, the novelty bar for a spotlight paper becomes harder to clear.

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
├── experiments/
│   ├── exp1_degradation.py      # Shift degradation analysis
│   ├── exp2_loss_ablation.py    # Loss ablation study
│   └── exp3_adapter_ablation.py # Adapter comparison
├── evaluate.py                  # MSE, MAE, relative improvement
├── run_fast.py                  # Fast feasibility runner (used for results above)
├── run_all.py                   # Full experiment runner
└── results.json                 # Raw experiment results
```

## Setup

```bash
conda create -n flowtta python=3.10 -y
conda activate flowtta
pip install torch torchvision torchaudio
pip install transformers accelerate safetensors
pip install gluonts datasets pandas matplotlib scipy einops tqdm
pip install chronos-forecasting

# Run fast feasibility experiment
python run_fast.py
```
