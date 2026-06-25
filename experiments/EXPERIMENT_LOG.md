# Experiment Log

This log tracks all experiments, observations, and decisions for the CT slice interpolation paper.

## How to Use This Log

**Before running experiments**:
1. Write hypothesis in the batch section below
2. Document motivation, expected outcome, and key question

**After each experiment completes**:
1. Review the training curves and visualizations
2. Fill in the Results section
3. Document conclusion and next steps

### Template Entry

```
## [YYYY-MM-DD] experiment_name
### Hypothesis
- **Motivation**: Why are we running this? What previous result prompted it?
- **Expected outcome**: What do we expect to happen?
- **Key question**: What specific question does this experiment answer?

### Results
- **Status**: EARLY_STOPPING / FINISHED_EPOCHS / ERROR
- **Config**: loss=X, lr=X, batch_size=X
- **Best val loss**: X.XXXX at epoch Y
- **Metrics**: SSIM=X, MS-SSIM=X, MAE=X, PSNR=X
- **Observations**: What did we learn?
- **Conclusion**: Did results match hypothesis? What did we learn?
- **Next steps**: What should we try based on this?
- **Paper relevance**: Which section/figure does this support?
```

### Adding Notes via Code

```python
from pathlib import Path
from phd.training.registry import add_experiment_notes

add_experiment_notes(
    experiments_dir=Path("./experiments/train_nn1_cropped"),
    exp_name="unet_..._ssim_lr_0.003_...",
    notes="Early stopping at epoch 11. Loss plateau suggests lr too high."
)
```

---

## Configuration Decisions

### [2026-02-01] Scheduler Selection: cosine_warmup over cosine_restarts

**Decision**: Use `cosine_warmup` with `early_stopping_patience=15` as default scheduler.

**Rationale**:
- Using pretrained EfficientNetV2 encoder requires stable early training
- Warmup prevents destabilizing pretrained encoder features
- Randomly initialized decoder adapts during warmup before full LR kicks in
- `num_epochs=500` serves as ceiling; early stopping is the intended stopping mechanism

**Configuration**:
```python
"scheduler": {"name": "cosine_warmup", "params": {"warmup_epochs": 5, "eta_min": 1e-6}}
"early_stopping_patience": 15
```

**Why warmup_epochs=5**:
- batch_size=96, train_size=2400 -> 25 batches/epoch
- 5 epochs = 125 gradient updates during warmup
- Sufficient for pretrained encoder (decoder catches up)
- If training unstable early, can increase to 10

**Paper relevance**: Methods section - training procedure and hyperparameters.

---

### [2026-02-05] torch.compile Mode Selection: default (CUDA graphs cause OOM)

**Decision**: Use `torch.compile(mode="default")` for training.

**Benchmark Results** (RTX 3080 Ti, batch_size=64, input shape `(N, 2, 256, 256)`):

| Mode | CUDA Graphs | ms/iter | Speedup vs Eager |
|------|:-----------:|---------|------------------|
| eager (no compile) | No | 217.87 | 1.00x |
| **default** | **No** | **178.88** | **1.22x** |
| reduce-overhead | Yes | 175.69 | 1.24x |
| max-autotune-no-cudagraphs | No | 171.87 | 1.27x |
| max-autotune | Yes | 160.34 | 1.36x |

**Rationale**:
- CUDA graphs (used by `reduce-overhead` and `max-autotune`) pre-allocate ~9.4GB GPU memory
- This memory is NOT released between training and validation phases
- Validation requires additional memory for 9 crops per sample, causing OOM
- `default` mode avoids CUDA graphs entirely with minimal speedup loss (1.22x vs 1.24x)

**History**: Initially chose `max-autotune`, then `reduce-overhead`, but both caused CUDA OOM during validation due to graph memory not being freed. Three experiments (`msssim+l1_lr8e-4_e6d845`, `msssim+l1_lr8e-4_e660c8`, `ssim+l1_lr8e-4_767702`) failed with OOM errors before this was identified.

**Benchmark script**: `scripts/benchmark_compile_modes.py`

**Paper relevance**: Methods section - training optimization and implementation details.

---

### [2026-02-06] Batch Size: 64 -> 96 revert

**Decision**: Revert batch_size from 64 back to 96.

**Evidence**: Batch 4 experiments (L1 and MSE at lr=3e-4 and lr=1e-4, run at batch_size=64) were dominated by NaN terminations, while the same losses at batch_size=96 (Batch 1) were stable. Batch_size=64 was introduced as a memory optimization alongside the compile mode fix, but confounded the LR ablation results.

**Impact**: Batch 4 results are unreliable and must be re-tested at batch_size=96 (L1 and MSE runs at the default batch size are in Batch 1).

---

### [2026-02-06] SSIM Stability Fix: K2=0.4 + nonnegative_ssim

**Decision**: For SSIM-based losses, use `K=(0.01, 0.4)` and `nonnegative_ssim=True`.

**Root cause analysis**:
- SSIM formula uses C2 = (K2 * data_range)^2 as a stability constant in the denominator
- With default K2=0.03: C2 = 0.0009 (very small, poor numerical stability)
- With K2=0.4: C2 = 0.16 (much more stable, prevents near-zero denominators)
- SSIM can return negative values (theoretically in [-1, 1]), so `1 - SSIM` can exceed 1.0
- `nonnegative_ssim=True` clamps SSIM to [0, 1], keeping loss in [0, 1]
- Additional `torch.clamp(loss, 0.0, 2.0)` provides a safety net

**Source**: `pytorch_msssim` documentation recommends K2=0.4 and nonnegative_ssim=True for numerical stability.

**Paper relevance**: Methods section - loss function implementation details. Important for reproducibility.

---

## Experiments

## [2026-02-11] msssim+l1_lr3e-4_9b7aea (transfer init from msssim+l1_lr8e-4_e6d845)

### Hypothesis
- **Motivation**: `msssim+l1_lr8e-4_e6d845` was promising (best valid loss `0.0723`, SSIM `0.7376`) but stopped after NaN/inf detection at epoch 21. Reusing its learned weights should skip unstable early dynamics while lowering LR for safer optimization. Weighting the loss toward L1 (`0.8`) is intended to bias reconstruction toward local/intensity detail while keeping MS-SSIM regularization.
- **Expected outcome**: Stable training without NaN/inf, equal or better low-level reconstruction detail, and competitive or improved validation loss versus the source run.
- **Key question**: Can transfer-initialized continuation at lower LR outperform the original run before instability appears?

### Planned Config
- **Experiment**: `msssim+l1_lr3e-4_9b7aea`
- **Source checkpoint**: `experiments/train_nn1_cropped/msssim+l1_lr8e-4_e6d845/epochs/20/weights.pth` (explicit best checkpoint)
- **Loss**: `msssim+l1` with `msssim_weight=0.2`, `l1_weight=0.8`
- **Optimizer**: `AdamW`, `lr=3e-4`, `weight_decay=1e-2`
- **Scheduler**: `cosine_warmup` (`warmup_epochs=5`, `eta_min=1e-6`)
- **Batch size**: `64` (kept equal to source run)
- **Epoch budget**: `300` with early stopping (`patience=15`)

### Pre-Run Expectations
- **Stability target**: No NaN/inf status (`EARLY_STOPPING` or `FINISHED_EPOCHS` expected).
- **Loss target**: Beat source best validation loss `0.0723`.
- **Quality target**: Match or improve source best SSIM `0.7376`.
- **Behavior target**: Faster convergence in early epochs than training-from-scratch at `3e-4`, with better MAE/detail retention from stronger L1 weighting.

### Results
- **Status**: `NAN_VALUE_DETECTED` (stopped at epoch 23)
- **Best epoch**: `19` (CSV epoch index)
- **Best checkpoint**: `experiments/train_nn1_cropped/msssim+l1_lr3e-4_9b7aea/epochs/18/weights.pth`
- **Best val loss**: `0.05884`
- **Best metrics**: `SSIM=0.7424`, `MS-SSIM=0.8521`, `MAE=0.03657`, `PSNR=20.16`
- **Observations**: Training/validation were stable through epoch 22, then validation produced `nan` in epoch 23 (intermittent `nan` batches). Despite failure, this run improved over the source run metrics before crashing.
- **Conclusion**: Transfer initialization succeeded in improving quality, but did not eliminate late-epoch numerical instability for this loss regime and batch size.
- **Next steps**: Retry from the best checkpoint with additional numerical stabilization (lower LR and MS-SSIM stability constants) or move to batch_size=96, which has historically been more stable for this loss family.

## [2026-02-12] msssim+l1_lr3e-4_f43627 (NaN-recovery continuation from best checkpoint)

### Hypothesis
- **Motivation**: `msssim+l1_lr3e-4_9b7aea` reached stronger quality (`valid_loss=0.05884`, `SSIM=0.7424`) but failed with NaN at epoch 23. Continue from its best checkpoint instead of restarting.
- **Expected outcome**: Preserve the achieved quality and extend stable training beyond the previous NaN point.
- **Key question**: Does re-starting from the best checkpoint with the same strategy avoid the late-epoch NaN?

### Planned Config
- **Experiment**: `msssim+l1_lr3e-4_f43627`
- **Source checkpoint**: `experiments/train_nn1_cropped/msssim+l1_lr3e-4_9b7aea/epochs/18/weights.pth`
- **Loss**: `msssim+l1` with `msssim_weight=0.2`, `l1_weight=0.8` (unchanged strategy)
- **Optimizer**: `AdamW`, `lr=3e-4`, `weight_decay=1e-2`
- **Scheduler**: `cosine_warmup` (`warmup_epochs=5`, `eta_min=1e-6`)
- **Batch size**: `64`
- **Epoch budget**: `300` with early stopping (`patience=15`)

### Pre-Run Expectations
- **Stability target**: Reach at least epoch 30 without NaN.
- **Quality target**: Keep `valid_loss <= 0.059` and `SSIM >= 0.742` in the best epoch.
- **Behavior target**: Resume from a stronger basin and avoid abrupt validation collapse seen at epoch 23.

### Results
- **Status**: `NAN_VALUE_DETECTED` (stopped at epoch 16)
- **Observations**: Improvement appeared only in the first epoch after restart; instability returned before extended fine-tuning.
- **Conclusion**: Repeating the same setup from the best checkpoint did not solve late-epoch NaN behavior.

## [2026-02-12] Fine-Tuning Sweep From `msssim+l1_lr3e-4_9b7aea/epochs/18/weights.pth`

### Hypothesis
- **Motivation**: The model is already strong; we need gentler updates and targeted stability knobs instead of large config shifts.
- **Expected outcome**: At least one conservative fine-tuning setup should maintain or improve `valid_loss<=0.05884` and avoid early NaN.
- **Key question**: Is the remaining instability mainly optimizer step size (LR/scheduler) or MS-SSIM numerical conditioning (`K`)?

### Experiment A: lower LR, constant schedule
- **Experiment**: `msssim+l1_lr1e-4_7ee268`
- **Source checkpoint**: `experiments/train_nn1_cropped/msssim+l1_lr3e-4_9b7aea/epochs/18/weights.pth`
- **Config**: `lr=1e-4`, `scheduler=none`, `batch_size=64`, `msssim/l1=0.2/0.8`
- **Expectation**: Smaller step size should reduce overshoot and keep validation stable longer than the 3e-4 continuation.

### Experiment B: lower LR + MS-SSIM stability constants
- **Experiment**: `msssim+l1_lr1e-4_0e2444`
- **Source checkpoint**: `experiments/train_nn1_cropped/msssim+l1_lr3e-4_9b7aea/epochs/18/weights.pth`
- **Config**: same as A + `K=[0.01, 0.4]`
- **Expectation**: If NaN is tied to MS-SSIM numerics, this should be more stable than A at similar quality.

### Experiment C: very low LR + MS-SSIM stability constants
- **Experiment**: `msssim+l1_lr5e-5_1494e0`
- **Source checkpoint**: `experiments/train_nn1_cropped/msssim+l1_lr3e-4_9b7aea/epochs/18/weights.pth`
- **Config**: `lr=5e-5`, `scheduler=none`, `batch_size=64`, `msssim/l1=0.2/0.8`, `K=[0.01, 0.4]`
- **Expectation**: Most conservative option; may improve very slowly but should maximize stability.

### Sweep status
- **Status**: `QUEUED` (three experiments)
- **Run order goal**: A -> B -> C (from least to most conservative).

## [2026-02-02] Batch 1: Initial Loss Function Survey

### Hypothesis
- **Motivation**: Establish baseline performance with all available loss functions before exploring combinations. Need to identify which losses are fundamentally compatible with our architecture (UNet + EfficientNetV2-S encoder) and task (CT slice interpolation).
- **Expected outcome**: MSE/L1 should be stable baselines. SSIM/MS-SSIM may provide better perceptual quality. Combined SSIM+L1 / MS-SSIM+L1 hybrids should improve over single losses.
- **Key question**: Which loss functions produce usable results with this architecture?

### mse_lr8e-4_b558b9
- **Status**: EARLY_STOPPING
- **Config**: loss=mse, lr=8e-4, batch_size=96
- **Best val loss**: 0.00905 at epoch 42
- **Metrics**: SSIM=0.721, MS-SSIM=0.837, MAE=0.041, PSNR=20.51
- **Final epoch**: 50
- **Observations**:
  - Completely stable training - zero NaN values
  - Fast convergence (best at epoch 42)
  - Good baseline performance
- **Conclusion**: MSE is reliable and stable. Use as baseline for comparison.
- **Paper relevance**: Baseline in loss function comparison table

### l1_lr8e-4_b39be9
- **Status**: EARLY_STOPPING
- **Config**: loss=l1, lr=8e-4, batch_size=96
- **Best val loss**: 0.0360 at epoch 35
- **Metrics**: SSIM=0.733, MS-SSIM=0.836, MAE=0.038, PSNR=19.67
- **Final epoch**: 140
- **Observations**:
  - Very stable training - only 5/140 epochs with NaN (3.5%)
  - Trains longer than MSE but achieves better SSIM (0.733 vs 0.721)
  - Better edge preservation than MSE (lower MAE)
- **Conclusion**: L1 outperforms MSE on perceptual metrics. Best single-loss performer.
- **Paper relevance**: Best single loss in comparison, basis for combined losses

### ssim_lr8e-4_1b8c15
- **Status**: EARLY_STOPPING (FAILED)
- **Config**: loss=ssim, lr=8e-4, batch_size=96
- **Best val loss**: 0.788 at epoch 6
- **Metrics**: SSIM=0.714, MS-SSIM=0.821, MAE=0.042
- **Final epoch**: 19
- **Observations**: SSIM loss alone diverged early with high loss values
- **Conclusion**: Pure SSIM is unstable. Never use alone.
- **Paper relevance**: Negative result for ablation study

### msssim_lr8e-4_b76c5f
- **Status**: EARLY_STOPPING (FAILED)
- **Config**: loss=msssim, lr=8e-4, batch_size=96
- **Best val loss**: 0.557 at epoch 1
- **Metrics**: SSIM=0.343, MS-SSIM=0.453, MAE=0.422
- **Final epoch**: 16
- **Observations**: Complete divergence from epoch 2. Even more unstable than SSIM.
- **Conclusion**: Never use MS-SSIM alone.
- **Paper relevance**: Negative result for ablation study

### ssim+l1_lr8e-4_5b448a (0.8/0.2)
- **Status**: EARLY_STOPPING (FAILED)
- **Config**: loss=ssim+l1 (0.8/0.2), lr=8e-4, batch_size=96
- **Best val loss**: 1.061 at epoch 1
- **Metrics**: SSIM=0.225, MS-SSIM=0.225, MAE=1.443
- **Final epoch**: 43
- **Observations**: 14% NaN epochs. SSIM weight of 0.8 too high.
- **Conclusion**: SSIM-dominant weights cause instability.
- **Paper relevance**: Demonstrates importance of weight tuning

### msssim+l1_lr8e-4_41ef70 (0.8/0.2)
- **Status**: EARLY_STOPPING (FAILED)
- **Config**: loss=msssim+l1 (0.8/0.2), lr=8e-4, batch_size=96
- **Best val loss**: 1.573 at epoch 1
- **Metrics**: SSIM=0.018, MS-SSIM=0.059, MAE=3.99, PSNR=-14.1
- **Final epoch**: 16
- **Observations**: Loss exploded to 49+. Negative PSNR.
- **Conclusion**: MS-SSIM weight of 0.8 causes complete divergence.
- **Paper relevance**: Negative result showing weight sensitivity

### Batch 1 Conclusion
Only 2 of 6 losses produced usable results: **MSE** (stable baseline) and **L1** (best performer). All SSIM-based losses failed.

---

## [2026-02-05] Batch 2: L1-Dominant SSIM+L1 Combinations (lr=8e-4)

### Hypothesis
- **Motivation**: Batch 1 showed SSIM-dominant (0.8) weights fail. Perhaps reducing SSIM weight to 0.2-0.5 stabilizes training by letting L1 dominate.
- **Expected outcome**: Lower SSIM weights should be more stable. 0.2 SSIM weight may work since L1 is 80% dominant.
- **Key question**: Is there a SSIM weight ratio that stabilizes training at lr=8e-4?

### ssim+l1_c37bea (0.5/0.5 balanced)
- **Status**: EARLY_STOPPING (FAILED)
- **Config**: loss=ssim+l1 (0.5/0.5), lr=8e-4, batch_size=96
- **Best val loss**: 1.1611 at epoch 1
- **Final epoch**: 17
- **Observations**: Never improved past epoch 1; loss exploded to 10.77

### ssim+l1_520eb4 (0.3/0.7 L1-dominant)
- **Status**: NAN_VALUE_DETECTED
- **Config**: loss=ssim+l1 (0.3/0.7), lr=8e-4, batch_size=96
- **Best val loss**: 0.5836 at epoch 1
- **Final epoch**: 5

### ssim+l1_aac33b (0.2/0.8 very L1-dominant)
- **Status**: NAN_VALUE_DETECTED
- **Config**: loss=ssim+l1 (0.2/0.8), lr=8e-4, batch_size=96
- **Best val loss**: 0.8090 at epoch 1
- **Final epoch**: 2
- **Observations**: Even 0.2 SSIM weight causes NaN within 2 epochs

### msssim+l1_d95aaa (0.3/0.7 L1-dominant)
- **Status**: EARLY_STOPPING (FAILED)
- **Config**: loss=msssim+l1 (0.3/0.7), lr=8e-4, batch_size=96
- **Best val loss**: 0.5659 at epoch 1
- **Final epoch**: 16
- **Observations**: Loss exploded to 85.17 by epoch 16

### msssim+l1_729d64 (0.5/0.5 balanced)
- **Status**: EARLY_STOPPING (FAILED)
- **Config**: loss=msssim+l1 (0.5/0.5), lr=8e-4, batch_size=96
- **Best val loss**: 0.8579 at epoch 2
- **Final epoch**: 6

### Batch 2 Conclusion
**All SSIM+L1 combinations fail at lr=8e-4**, regardless of weight ratio. Even 0.2 SSIM weight causes NaN within 2 epochs. The problem is not the weight ratio - it's SSIM loss instability at this learning rate.

---

## [2026-02-05] Repeat Runs (same configs, num_workers=2)

These experiments duplicated Batch 2 configs with `num_workers=2` (reduced from 4 for stability). Results confirm Batch 2 findings.

### ssim+l1_lr8e-4_061577 (0.5/0.5)
- **Status**: NAN_VALUE_DETECTED at epoch 6
- **Confirms**: ssim+l1_c37bea result

### ssim+l1_lr8e-4_7bf0f4 (0.3/0.7)
- **Status**: NAN_VALUE_DETECTED at epoch 5
- **Confirms**: ssim+l1_520eb4 result

### ssim+l1_lr8e-4_767702 (0.2/0.8)
- **Status**: NAN_VALUE_DETECTED at epoch 8, then CUDA OOM
- **Confirms**: ssim+l1_aac33b result + revealed CUDA graph OOM issue

### msssim+l1_lr8e-4_e6d845 (0.3/0.7)
- **Status**: ERROR - CUDA OOM
- **Note**: OOM caused by CUDA graphs from `reduce-overhead` compile mode

### msssim+l1_lr8e-4_e660c8 (0.5/0.5)
- **Status**: ERROR - CUDA OOM
- **Note**: Same CUDA graph OOM issue; led to switching to `default` compile mode

---

## [2026-02-06] Batch 3: Lower LR with SSIM-Dominant (ablation study)

### Hypothesis
- **Motivation**: All SSIM combinations fail at lr=8e-4 regardless of weight. Maybe the learning rate is too high for SSIM's gradient landscape.
- **Expected outcome**: Lower LR (1e-4, 3e-4) should delay or prevent divergence. If lr=1e-4 works, the problem is gradient magnitude, not SSIM itself.
- **Key question**: Does lower LR stabilize SSIM-based losses?

NOTE: These ran with batch_size=64 (CUDA OOM fix), which later proved to be a confounding variable.

### ssim+l1_lr1e-4_319ee2 (0.8/0.2, lr=1e-4)
- **Status**: NAN_VALUE_DETECTED
- **Config**: loss=ssim+l1 (0.8/0.2), lr=1e-4, batch_size=64
- **Best val loss**: 0.7912 at epoch 1
- **Final epoch**: 15
- **Observations**: Lower LR delayed divergence (15 epochs vs immediate) but still failed
- **Conclusion**: SSIM instability is not purely a learning rate problem

### ssim+l1_lr3e-4_675c17 (0.8/0.2, lr=3e-4)
- **Status**: EARLY_STOPPING (FAILED)
- **Config**: loss=ssim+l1 (0.8/0.2), lr=3e-4, batch_size=64
- **Best val loss**: 1.0073 at epoch 2
- **Final epoch**: 17
- **Observations**: More stable than lr=8e-4 but still diverged. Best model is garbage (SSIM=0.060).

### msssim+l1_lr1e-4_060c58 (0.8/0.2, lr=1e-4)
- **Status**: NAN_VALUE_DETECTED
- **Config**: loss=msssim+l1 (0.8/0.2), lr=1e-4, batch_size=64
- **Best val loss**: 0.5949 at epoch 1
- **Final epoch**: 3 (validation loss became -inf)
- **Observations**: Negative SSIM values by epoch 3 (center=-0.186) confirm total failure

### Batch 3 Conclusion
**Lower LR does NOT fix SSIM-based losses.** SSIM+L1 lasted 15 epochs at lr=1e-4 vs ~2-5 at lr=8e-4, but still diverged. The SSIM loss formulation is fundamentally incompatible at any tested LR. Root cause investigation needed (led to discovering K2/nonnegative_ssim fix).

---

## [2026-02-06] Batch 4: LR Ablation on Successful Losses

### Hypothesis
- **Motivation**: L1 and MSE produce usable results. Need to optimize LR for each to find the best configuration for the paper.
- **Expected outcome**: Lower LR (3e-4, 1e-4) may find deeper optima for L1/MSE.
- **Key question**: What is the optimal LR for each successful loss?

NOTE: **These ran at batch_size=64**, which was later found to confound results. All produced NaN, making LR conclusions unreliable. Later batches re-test L1 and MSE at batch_size=96.

### l1_lr3e-4_a55ade
- **Status**: (results pending documentation)
- **Config**: loss=l1, lr=3e-4, batch_size=64

### l1_lr1e-4_66e5e2
- **Status**: (results pending documentation)
- **Config**: loss=l1, lr=1e-4, batch_size=64

### mse_lr3e-4_bdee4f
- **Status**: (results pending documentation)
- **Config**: loss=mse, lr=3e-4, batch_size=64

### Batch 4 Conclusion
Results confounded by batch_size=64 (causes NaN in previously-stable losses). Must re-run at batch_size=96 to get reliable LR comparisons.

---

## [2026-02-06] Batch 5: SSIM Reproduction (matching old working config)

### Hypothesis
- **Motivation**: An older version of the code (no AMP, no torch.compile, batch_size=32, lr=3e-3) successfully trained with SSIM. Need to test if those hyperparameters work with the current code.
- **Expected outcome**: If the old config works, the problem is in our new settings (AMP/compile/batch_size). If it fails, the problem is deeper.
- **Key question**: Can we reproduce the old SSIM success with current code infrastructure?

### ssim_lr3e-3_94f982
- **Status**: (results pending documentation)
- **Config**: loss=ssim (pure), lr=3e-3, batch_size=32

### ssim+l1_lr3e-4_a55ade (0.8/0.2, batch_size=32)
- **Status**: (results pending documentation)
- **Config**: loss=ssim+l1 (0.8/0.2), lr=3e-4, batch_size=32

### Batch 5 Conclusion
(Results pending)

---

## [2026-02-06] Batch 6: SSIM Stability Fix (PRE-EXPERIMENT)

### Hypothesis
- **Motivation**: Root cause analysis revealed SSIM instability is caused by (1) default K2=0.03 giving poor numerical stability in SSIM's C2 constant, (2) SSIM can return negative values, making `1-SSIM > 1`, and (3) no loss clamping. The `pytorch_msssim` library documents the fix: `K2=0.4` and `nonnegative_ssim=True`. Additionally, batch_size=64 (introduced in Batch 4) confounded results.
- **Expected outcome**:
  - SSIM+L1 experiments (#18, #19) should now converge where they previously diverged
  - MS-SSIM+L1 (#20) should stabilize (K2 fix applies to MS-SSIM too)
  - Pure SSIM (#21) is the acid test - if this works, the fix is confirmed
  - LR ablation and additional MS-SSIM weight points complete the sweep at batch_size=96
- **Key question**: Do pytorch_msssim stability parameters (K2=0.4, nonnegative_ssim=True) fix the SSIM divergence that plagued 25+ experiments?

### Experiment Details

| # | Name | Loss | Key Params | LR | batch_size | Rationale |
|---|------|------|------------|-----|------------|-----------|
| 18 | ssim+l1 (0.8/0.2) | ssim+l1 | K=[0.01,0.4], nonneg=True | 8e-4 | 96 | SSIM-dominant with stability fix |
| 19 | ssim+l1 (0.5/0.5) | ssim+l1 | K=[0.01,0.4], nonneg=True | 8e-4 | 96 | Balanced with stability fix |
| 20 | msssim+l1 (0.5/0.5) | msssim+l1 | K=[0.01,0.4] | 8e-4 | 96 | MS-SSIM balanced with stability fix |
| 21 | ssim (pure) | ssim | K=[0.01,0.4], nonneg=True | 3e-3 | 32 | Acid test: reproduce old success with fix |
| 22 | ssim+l1 (0.8/0.2) | ssim+l1 | K=[0.01,0.4], nonneg=True | 3e-4 | 96 | LR ablation for SSIM-dominant with fix (complement to #18 at lr=8e-4) |
| 23 | msssim+l1 (0.3/0.7) | msssim+l1 | K=[0.01,0.4] | 8e-4 | 96 | L1-dominant MS-SSIM with fix (completes MS-SSIM weight sweep) |

### Results
(Experiments not yet run)

---

## Key Findings Summary (Updated 2026-02-06)

### Successful Losses

| Loss | Experiment | Best SSIM | Best MAE | Stable? |
|------|-----------|-----------|----------|---------|
| **L1** | l1_lr8e-4_b39be9 | **0.733** | **0.038** | Yes |
| MSE | mse_lr8e-4_b558b9 | 0.721 | 0.041 | Yes |

### Failed SSIM+L1 Combinations (Complete Sweep)

| Weights (SSIM/L1) | lr=8e-4 | lr=3e-4 | lr=1e-4 |
|-------------------|---------|---------|---------|
| 0.8/0.2 | NaN (ep 1) | Diverged (ep 17) | NaN (ep 15) |
| 0.5/0.5 | Diverged (ep 17) | - | - |
| 0.3/0.7 | NaN (ep 5) | - | - |
| 0.2/0.8 | NaN (ep 2) | - | - |

### Failed MS-SSIM+L1 Combinations

| Weights (MS-SSIM/L1) | lr=8e-4 | lr=1e-4 |
|----------------------|---------|---------|
| 0.8/0.2 | Diverged (ep 16) | NaN (ep 3) |
| 0.5/0.5 | Diverged (ep 6) | - |
| 0.3/0.7 | Diverged (ep 16) | - |

### Other Failed Losses

| Loss | Status | Notes |
|------|--------|-------|
| SSIM alone | Diverged | Never use alone (without stability fix) |
| MS-SSIM alone | Diverged | Never use alone |

### Conclusions (Pre-Batch 6)

1. **SSIM-based losses failed in 25+ experiments** at every weight ratio and learning rate tested
2. **Root cause identified**: Default K2=0.03 gives poor numerical stability; SSIM can go negative
3. **Fix applied**: K2=0.4, nonnegative_ssim=True, loss clamping (Batch 6 will validate)
4. **L1 is the best stable loss** (SSIM=0.733)
5. **MSE is the most reliable baseline** (SSIM=0.721, zero NaN)
6. **batch_size=64 causes NaN** in previously-stable losses (confounded Batch 4)
7. **Next direction**: Batch 6 tests the stability fix on SSIM-family losses
8. **Batch 7**: Extends Batch 6 with additional weight ratios and LR ablation

---

## [2026-02-07] Batch 7: Extended Stability Fix + Weight Sweep (PRE-EXPERIMENT)

### Hypothesis
- **Motivation**: Batch 6 tests the SSIM stability fix (K2=0.4, nonnegative_ssim=True) at specific weight ratios, but leaves gaps in the parameter space. Specifically: (1) SSIM+L1 at 0.3/0.7 with the fix is untested, (2) MS-SSIM dominant weighting (0.8/0.2) was never attempted with the fix, and (3) no LR ablation exists for the fixed SSIM+L1 losses.
- **Expected outcome**:
  - SSIM+L1 (0.3/0.7, #24) should be the most stable SSIM combo since L1 dominates heavily and the fix prevents SSIM instability
  - MS-SSIM+L1 (0.8/0.2, #25) is the riskiest experiment; the fix should help but MS-SSIM dominant weight hasn't worked before
  - SSIM+L1 at lr=3e-4 (#26) provides an LR ablation point for the fixed SSIM+L1 losses
- **Key question**: Does the SSIM stability fix enable previously-impossible MS-SSIM-dominant configurations, and what weight ratio produces the best results?

### Experiment Details

| # | Name | Loss | Key Params | LR | batch_size | Rationale |
|---|------|------|------------|-----|------------|-----------|
| 24 | ssim+l1 (0.3/0.7) | ssim+l1 | K=[0.01,0.4], nonneg=True | 8e-4 | 96 | Complete SSIM+L1 weight sweep (adds to #18 0.8/0.2 and #19 0.5/0.5) |
| 25 | msssim+l1 (0.8/0.2) | msssim+l1 | K=[0.01,0.4] | 8e-4 | 96 | MS-SSIM dominant with fix (never tested with fix) |
| 26 | ssim+l1 (0.5/0.5) | ssim+l1 | K=[0.01,0.4], nonneg=True | 3e-4 | 96 | LR ablation for fixed SSIM+L1 (complement to #19 at 8e-4) |

### Paper Relevance
- **Experiments #24-25**: Complete the loss function comparison table for the paper (SSIM+L1 weight sweep and MS-SSIM dominant)
- **Experiment #26**: Provides LR sensitivity data for the paper's hyperparameter analysis

### Results
(Experiments not yet run)

---

## Proposed Batch 4 (Original, for reference)

Based on the finding that only L1 and MSE losses produce usable results, the next experiments optimize learning rate for these two losses.

| # | Loss | LR | Rationale |
|---|------|----|-----------|
| 1 | L1 | 3e-4 | L1 is best; lower LR may find deeper optimum |
| 2 | L1 | 1e-4 | Conservative LR for maximum stability |
| 3 | MSE | 3e-4 | Baseline at different LR for comparison |

**Why these 3**:
- L1 trained for 140 epochs at lr=8e-4; lower LR could converge to better optimum
- MSE provides a stable reference point at the alternative LR
- Both losses have proven stable enough to complete training

---

## [2026-02-08] Comprehensive Review of `experiments/train_nn1_cropped` (47 experiments)

Snapshot taken on **2026-02-08** while `ssim+l1_lr8e-4_1fa946` was still running, so that one row can change.

### Review Rubric

- **Success flag**:
  - `SUCCESS`: converged to a usable checkpoint
  - `PARTIAL`: interrupted/NaN, but produced at least one potentially usable checkpoint
  - `FAILED`: diverged or produced unusable quality
  - `PENDING`: still running or not started
- **Paper flag**:
  - `CORE`: include in main paper quantitative comparison
  - `SUPPORT`: include as baseline, ablation, or stability analysis
  - `LOW`: keep in internal log only (redundant/weak)
  - `PENDING`: decision deferred

### Overall Flag Counts

- **Success flags**: `SUCCESS=7`, `PARTIAL=12`, `FAILED=21`, `PENDING=7`
- **Paper flags**: `CORE=3`, `SUPPORT=19`, `LOW=18`, `PENDING=7`

### One-by-One Review

| Experiment | Registry status | Best metrics (SSIM / MS-SSIM / MAE / PSNR) | Success flag | Paper flag | Notes |
|---|---|---|---|---|---|
| `mse_lr8e-4_b558b9` | EARLY_STOPPING | 0.721 / 0.837 / 0.041 / 20.540 | SUCCESS | SUPPORT | Stable MSE baseline; keep in main comparison table. |
| `l1_lr8e-4_b39be9` | EARLY_STOPPING | 0.741 / 0.847 / 0.036 / 20.031 | SUCCESS | CORE | Best stable L1 run; strong candidate for final model comparison. |
| `ssim_lr8e-4_1b8c15` | EARLY_STOPPING | 0.210 / 0.090 / 12.159 / -30.513 | FAILED | SUPPORT | Pure SSIM at 8e-4 is unstable and produces unusable reconstructions. |
| `msssim_lr8e-4_b76c5f` | EARLY_STOPPING | 0.343 / 0.453 / 0.422 / 2.210 | FAILED | SUPPORT | Pure MS-SSIM fails quickly; useful negative control. |
| `ssim+l1_lr8e-4_5b448a` | EARLY_STOPPING | 0.123 / 0.157 / 1.443 / -4.026 | FAILED | LOW | SSIM-dominant (0.8/0.2) diverged; redundant with later repeats. |
| `msssim+l1_lr8e-4_41ef70` | EARLY_STOPPING | -0.015 / 0.047 / 3.987 / -14.143 | FAILED | SUPPORT | MS-SSIM-dominant run collapsed with severe artifacts; useful failure case. |
| `ssim+l1_c37bea` | EARLY_STOPPING | 0.085 / 0.112 / 1.330 / -6.096 | FAILED | LOW | 0.5/0.5 pre-fix failed immediately; redundant failure. |
| `ssim+l1_520eb4` | NAN_VALUE_DETECTED | 0.088 / 0.283 / 0.434 / 2.954 | FAILED | LOW | 0.3/0.7 pre-fix NaN early; redundant failure. |
| `ssim+l1_aac33b` | NAN_VALUE_DETECTED | 0.070 / 0.136 / 0.767 / 0.413 | FAILED | LOW | 0.2/0.8 pre-fix NaN at epoch 2; redundant failure. |
| `msssim+l1_d95aaa` | EARLY_STOPPING | 0.281 / 0.370 / 0.532 / 0.533 | FAILED | LOW | 0.3/0.7 pre-fix diverged; low additional information. |
| `msssim+l1_729d64` | EARLY_STOPPING | 0.008 / 0.008 / 2.091 / -8.394 | FAILED | LOW | 0.5/0.5 pre-fix diverged; low additional information. |
| `ssim+l1_lr8e-4_061577` | NAN_VALUE_DETECTED | 0.051 / 0.178 / 0.759 / -1.183 | FAILED | LOW | Repeat with num_workers=2 still failed; confirms pre-fix instability. |
| `ssim+l1_lr8e-4_7bf0f4` | NAN_VALUE_DETECTED | 0.113 / 0.293 / 0.468 / 2.600 | FAILED | LOW | Repeat with num_workers=2 still failed; no extra value for paper tables. |
| `ssim+l1_lr8e-4_767702` | NAN_VALUE_DETECTED | 0.123 / 0.188 / 0.591 / 0.464 | FAILED | SUPPORT | Failed and exposed CUDA graph/OOM behavior; relevant for implementation notes. |
| `msssim+l1_lr8e-4_e6d845` | NOT_STARTED | 0.731 / 0.842 / 0.040 / 19.705 | PARTIAL | SUPPORT | Partial run reached strong metrics but was interrupted by shared-memory failure; rerun needed. |
| `msssim+l1_lr8e-4_e660c8` | NAN_VALUE_DETECTED | 0.697 / 0.819 / 0.047 / 19.000 | PARTIAL | SUPPORT | Only 2 epochs before stop; early metrics promising but evidence is insufficient. |
| `ssim+l1_lr1e-4_319ee2` | NAN_VALUE_DETECTED | 0.125 / 0.242 / 0.434 / 3.524 | FAILED | SUPPORT | Lower LR delayed but did not prevent instability; useful ablation evidence. |
| `ssim+l1_lr3e-4_675c17` | EARLY_STOPPING | 0.060 / 0.079 / 1.305 / -4.087 | FAILED | LOW | Lower LR alone still produced unusable quality. |
| `msssim+l1_lr1e-4_060c58` | NAN_VALUE_DETECTED | 0.132 / 0.356 / 0.381 / 3.550 | FAILED | SUPPORT | Lower LR for MS-SSIM+L1 still failed almost immediately. |
| `l1_lr3e-4_dadf41` | NAN_VALUE_DETECTED | 0.673 / 0.795 / 0.048 / 18.750 | PARTIAL | LOW | Batch_size=64 run stopped early with NaN; not reliable for final conclusions. |
| `l1_lr1e-4_66e5e2` | NAN_VALUE_DETECTED | 0.725 / 0.832 / 0.039 / 19.665 | PARTIAL | SUPPORT | Good early quality but batch_size=64 confounds interpretation; keep as auxiliary evidence only. |
| `mse_lr3e-4_bdee4f` | EARLY_STOPPING | 0.721 / 0.836 / 0.041 / 20.525 | SUCCESS | SUPPORT | Stable MSE LR-ablation point with strong PSNR. |
| `ssim_lr3e-3_94f982` | EARLY_STOPPING | 0.746 / 0.848 / 0.041 / 19.835 | SUCCESS | CORE | Highest SSIM among completed runs; key result for the paper. |
| `ssim+l1_lr3e-4_a55ade` | NAN_VALUE_DETECTED | 0.729 / 0.829 / 0.042 / 19.010 | PARTIAL | SUPPORT | bs32 reproduction had good early quality but still NaN; useful transitional evidence. |
| `l1_lr3e-4_80e941` | NAN_VALUE_DETECTED | 0.696 / 0.795 / 0.046 / 18.626 | PARTIAL | LOW | Short run and NaN stop; below stronger L1 baselines. |
| `l1_lr1e-4_30c3cb` | EARLY_STOPPING | 0.721 / 0.829 / 0.040 / 19.559 | SUCCESS | SUPPORT | Stable low-LR L1 baseline for sensitivity analysis. |
| `mse_lr3e-4_c2d626` | NAN_VALUE_DETECTED | 0.648 / 0.798 / 0.050 / 19.601 | FAILED | LOW | NaN stop and weaker quality than stable MSE baselines. |
| `ssim+l1_lr8e-4_0921fd` | NAN_VALUE_DETECTED | 0.711 / 0.817 / 0.044 / 19.419 | PARTIAL | SUPPORT | With K/nonnegative fix reached competitive quality until epoch 7, then NaN at epoch 8. |
| `ssim+l1_lr8e-4_2707b3` | NAN_VALUE_DETECTED | 0.714 / 0.820 / 0.043 / 19.465 | PARTIAL | SUPPORT | With K/nonnegative fix reached competitive quality until epoch 7, then NaN at epoch 8. |
| `msssim+l1_lr8e-4_bc1d65` | EARLY_STOPPING | 0.738 / 0.846 / 0.037 / 20.224 | SUCCESS | CORE | Best overall trade-off (high SSIM/MS-SSIM with low MAE); strongest paper candidate. |
| `ssim_lr3e-3_92c8b6` | EARLY_STOPPING | 0.731 / 0.836 / 0.041 / 20.162 | SUCCESS | SUPPORT | Stable rerun with K/nonnegative settings; good reproducibility check. |
| `ssim+l1_lr8e-4_1fa946` | RUNNING | 0.722 / 0.829 / 0.041 / 19.639 | PENDING | PENDING | Running and improving; high-priority candidate if stability continues. |
| `msssim+l1_lr8e-4_19d5c5` | NOT_STARTED | N/A | PENDING | PENDING | Not started; important MS-SSIM-dominant test with stability fix. |
| `ssim+l1_lr3e-4_3ceddb` | NOT_STARTED | N/A | PENDING | PENDING | Not started; LR ablation for fixed 0.5/0.5 setup. |
| `ssim+l1_lr3e-4_6c62ad` | NOT_STARTED | N/A | PENDING | PENDING | Not started; LR ablation for fixed 0.8/0.2 setup. |
| `msssim+l1_lr8e-4_4e35ec` | NOT_STARTED | N/A | PENDING | PENDING | Not started; L1-dominant MS-SSIM test with stability fix. |

### Shortlist for Paper Draft Integration Now (before queue finishes)

1. `ssim_lr3e-3_94f982` (`CORE`): best SSIM among completed experiments.
2. `l1_lr8e-4_b39be9` (`CORE`): best stable L1 baseline with very low MAE.
3. `msssim+l1_lr8e-4_bc1d65` (`CORE`): best overall balance of SSIM/MS-SSIM/MAE.
4. `mse_lr8e-4_b558b9` and `mse_lr3e-4_bdee4f` (`SUPPORT`): MSE baselines.
5. `ssim_lr8e-4_1b8c15`, `msssim_lr8e-4_b76c5f` (`SUPPORT`): representative failure ablations.

### High-Priority Queue Items Before Final Manuscript Lock

1. Finish `ssim+l1_lr8e-4_1fa946` (currently RUNNING).
2. Run `msssim+l1_lr8e-4_19d5c5` and `msssim+l1_lr8e-4_4e35ec` (complete fixed MS-SSIM weight sweep).
3. Re-run shared-memory-interrupted experiments with enough `/dev/shm`: `msssim+l1_lr8e-4_e6d845`.

---

## [2026-02-08] Statistical Evidence Freeze for Paper Draft (patient-level primary)

Generated with:

```bash
./.venv/bin/python scripts/build_paper_tables.py \
  --snapshot-date 2026-02-08 \
  --device cuda \
  --batch-size 16 \
  --reference-experiment msssim+l1_lr8e-4_bc1d65 \
  --test-experiment baseline_mean \
  --test-experiment msssim+l1_lr8e-4_bc1d65 \
  --test-experiment msssim+l1_lr8e-4_e6d845 \
  --test-experiment ssim_lr3e-3_94f982 \
  --test-experiment l1_lr8e-4_b39be9 \
  --test-experiment mse_lr8e-4_b558b9 \
  --test-experiment l1_lr1e-4_66e5e2 \
  --test-experiment ssim_lr8e-4_1b8c15
```

Artifacts:

- `results/tables/test_patient_summary_2026-02-08.csv`
- `results/tables/test_paired_stats_2026-02-08.csv`
- `results/tables/test_slice_metrics_2026-02-08.csv`

### Primary Findings

- All selected learned models strongly outperform `baseline_mean`.
- Best patient-level **SSIM** among completed runs: `ssim_lr3e-3_94f982` (0.892).
- Best patient-level **MAE** among completed runs: `l1_lr8e-4_b39be9` (0.0166).
- Best balanced reference model for paired analysis: `msssim+l1_lr8e-4_bc1d65`.

### Partial/Interrupted Run Decisions

- `l1_lr1e-4_66e5e2`: keep as **SUPPORT** only.
  - Despite decent absolute quality, paired patient-level stats show it is significantly worse than `msssim+l1_lr8e-4_bc1d65` on key metrics.
- `msssim+l1_lr8e-4_e6d845`: keep as **SUPPORT** only.
  - Full test evaluation confirms it is significantly worse than the selected reference on SSIM, MAE, and PSNR.
- `ssim_lr8e-4_1b8c15`: keep as **negative control** (catastrophic failure).

### Paper-Integration Flags (Updated)

- `CORE`: `ssim_lr3e-3_94f982`, `l1_lr8e-4_b39be9`, `msssim+l1_lr8e-4_bc1d65`
- `SUPPORT`: `mse_lr8e-4_b558b9`, `l1_lr1e-4_66e5e2`, `msssim+l1_lr8e-4_e6d845`, `ssim_lr8e-4_1b8c15`, `baseline_mean` (classical)
- `PENDING`: all runs still not started or not completed as of this snapshot
---

## [2026-02-12] Comprehensive Review Refresh of `experiments/train_nn1_cropped` (41 experiments)

Snapshot date: **2026-02-12**.

### Review Rules
- `valid_loss` is **not compared across different loss families** (different scales); ranking is based mainly on **SSIM (up), MAE (down), PSNR (up)** and stability.
- `NAN_VALUE_DETECTED` runs are only considered usable via their best checkpoint, never final state.
- Goal: identify which runs are production-worthy for next-batch initialization and visual test-set comparison.

### Counts
- Registry status counts: `EARLY_STOPPING=19`, `NAN_VALUE_DETECTED=22`
- Evaluation counts: `SELECTED=5`, `STRONG_BACKUP=3`, `PROMISING=5`, `BASELINE=4`, `PARTIAL=5`, `FAILED=19`

### One-by-One Evaluation

| Experiment | Status | Best metrics (SSIM / MS-SSIM / MAE / PSNR) | Eval | Decision |
|---|---|---|---|---|
| `l1_lr1e-4_30c3cb` | `EARLY_STOPPING` | 0.7209 / 0.8287 / 0.0399 / 19.559 | **BASELINE** | Stable comparator baseline. |
| `l1_lr1e-4_66e5e2` | `NAN_VALUE_DETECTED` | 0.7252 / 0.8315 / 0.0391 / 19.665 | **PARTIAL** | Some usable quality, but unstable or weaker. |
| `l1_lr3e-4_80e941` | `NAN_VALUE_DETECTED` | 0.6961 / 0.7950 / 0.0458 / 18.626 | **FAILED** | Diverged or low-quality output. |
| `l1_lr3e-4_dadf41` | `NAN_VALUE_DETECTED` | 0.6730 / 0.7946 / 0.0484 / 18.750 | **FAILED** | Diverged or low-quality output. |
| `l1_lr8e-4_b39be9` | `EARLY_STOPPING` | 0.7415 / 0.8467 / 0.0360 / 20.031 | **SELECTED** | Selected for next-batch seeds and DCM/image comparison. |
| `mse_lr3e-4_bdee4f` | `EARLY_STOPPING` | 0.7211 / 0.8364 / 0.0407 / 20.525 | **BASELINE** | Stable comparator baseline. |
| `mse_lr3e-4_c2d626` | `NAN_VALUE_DETECTED` | 0.6482 / 0.7976 / 0.0498 / 19.601 | **FAILED** | Diverged or low-quality output. |
| `mse_lr8e-4_b558b9` | `EARLY_STOPPING` | 0.7207 / 0.8370 / 0.0408 / 20.540 | **SELECTED** | Selected for next-batch seeds and DCM/image comparison. |
| `msssim+l1_729d64` | `EARLY_STOPPING` | 0.0081 / 0.0076 / 2.0910 / -8.394 | **FAILED** | Diverged or low-quality output. |
| `msssim+l1_d95aaa` | `EARLY_STOPPING` | 0.2812 / 0.3696 / 0.5324 / 0.533 | **FAILED** | Diverged or low-quality output. |
| `msssim+l1_lr1e-4_060c58` | `NAN_VALUE_DETECTED` | 0.1322 / 0.3564 / 0.3808 / 3.550 | **FAILED** | Diverged or low-quality output. |
| `msssim+l1_lr1e-4_0e2444` | `NAN_VALUE_DETECTED` | 0.7417 / 0.8505 / 0.0364 / 20.140 | **PROMISING** | High-quality checkpoint but terminal NaN; use best checkpoint only. |
| `msssim+l1_lr1e-4_7ee268` | `NAN_VALUE_DETECTED` | 0.7434 / 0.8532 / 0.0364 / 20.187 | **PROMISING** | High-quality checkpoint but terminal NaN; use best checkpoint only. |
| `msssim+l1_lr3e-4_9b7aea` | `NAN_VALUE_DETECTED` | 0.7424 / 0.8521 / 0.0366 / 20.160 | **SELECTED** | Selected for next-batch seeds and DCM/image comparison. |
| `msssim+l1_lr3e-4_f43627` | `NAN_VALUE_DETECTED` | 0.7432 / 0.8529 / 0.0364 / 20.175 | **PROMISING** | High-quality checkpoint but terminal NaN; use best checkpoint only. |
| `msssim+l1_lr5e-5_1494e0` | `NAN_VALUE_DETECTED` | 0.7420 / 0.8510 / 0.0363 / 20.172 | **PARTIAL** | Some usable quality, but unstable or weaker. |
| `msssim+l1_lr8e-4_19d5c5` | `EARLY_STOPPING` | 0.7357 / 0.8448 / 0.0382 / 20.426 | **BASELINE** | Stable comparator baseline. |
| `msssim+l1_lr8e-4_41ef70` | `EARLY_STOPPING` | -0.0147 / 0.0469 / 3.9869 / -14.143 | **FAILED** | Diverged or low-quality output. |
| `msssim+l1_lr8e-4_4e35ec` | `NAN_VALUE_DETECTED` | 0.7380 / 0.8445 / 0.0369 / 20.079 | **PROMISING** | High-quality checkpoint but terminal NaN; use best checkpoint only. |
| `msssim+l1_lr8e-4_bc1d65` | `EARLY_STOPPING` | 0.7385 / 0.8455 / 0.0369 / 20.224 | **STRONG_BACKUP** | Stable + strong; keep as swap-in backup. |
| `msssim+l1_lr8e-4_e660c8` | `NAN_VALUE_DETECTED` | 0.6972 / 0.8192 / 0.0467 / 19.000 | **FAILED** | Diverged or low-quality output. |
| `msssim+l1_lr8e-4_e6d845` | `NAN_VALUE_DETECTED` | 0.7376 / 0.8482 / 0.0382 / 19.946 | **PROMISING** | High-quality checkpoint but terminal NaN; use best checkpoint only. |
| `msssim_lr8e-4_b76c5f` | `EARLY_STOPPING` | 0.3431 / 0.4529 / 0.4220 / 2.210 | **FAILED** | Diverged or low-quality output. |
| `ssim+l1_520eb4` | `NAN_VALUE_DETECTED` | 0.0884 / 0.2830 / 0.4340 / 2.954 | **FAILED** | Diverged or low-quality output. |
| `ssim+l1_aac33b` | `NAN_VALUE_DETECTED` | 0.0704 / 0.1361 / 0.7667 / 0.413 | **FAILED** | Diverged or low-quality output. |
| `ssim+l1_c37bea` | `EARLY_STOPPING` | 0.0847 / 0.1117 / 1.3304 / -6.096 | **FAILED** | Diverged or low-quality output. |
| `ssim+l1_lr1e-4_319ee2` | `NAN_VALUE_DETECTED` | 0.1253 / 0.2424 / 0.4336 / 3.524 | **FAILED** | Diverged or low-quality output. |
| `ssim+l1_lr3e-4_3ceddb` | `EARLY_STOPPING` | 0.7392 / 0.8439 / 0.0371 / 20.220 | **STRONG_BACKUP** | Stable + strong; keep as swap-in backup. |
| `ssim+l1_lr3e-4_675c17` | `EARLY_STOPPING` | 0.0598 / 0.0790 / 1.3053 / -4.087 | **FAILED** | Diverged or low-quality output. |
| `ssim+l1_lr3e-4_6c62ad` | `EARLY_STOPPING` | 0.7410 / 0.8454 / 0.0370 / 20.320 | **STRONG_BACKUP** | Stable + strong; keep as swap-in backup. |
| `ssim+l1_lr3e-4_a55ade` | `NAN_VALUE_DETECTED` | 0.7293 / 0.8290 / 0.0421 / 19.010 | **PARTIAL** | Some usable quality, but unstable or weaker. |
| `ssim+l1_lr8e-4_061577` | `NAN_VALUE_DETECTED` | 0.0507 / 0.1775 / 0.7593 / -1.183 | **FAILED** | Diverged or low-quality output. |
| `ssim+l1_lr8e-4_0921fd` | `NAN_VALUE_DETECTED` | 0.7108 / 0.8168 / 0.0440 / 19.419 | **PARTIAL** | Some usable quality, but unstable or weaker. |
| `ssim+l1_lr8e-4_1fa946` | `EARLY_STOPPING` | 0.7422 / 0.8468 / 0.0363 / 20.241 | **SELECTED** | Selected for next-batch seeds and DCM/image comparison. |
| `ssim+l1_lr8e-4_2707b3` | `NAN_VALUE_DETECTED` | 0.7143 / 0.8197 / 0.0434 / 19.465 | **PARTIAL** | Some usable quality, but unstable or weaker. |
| `ssim+l1_lr8e-4_5b448a` | `EARLY_STOPPING` | 0.1234 / 0.1567 / 1.4431 / -4.026 | **FAILED** | Diverged or low-quality output. |
| `ssim+l1_lr8e-4_767702` | `NAN_VALUE_DETECTED` | 0.1227 / 0.1885 / 0.5914 / 0.464 | **FAILED** | Diverged or low-quality output. |
| `ssim+l1_lr8e-4_7bf0f4` | `NAN_VALUE_DETECTED` | 0.1132 / 0.2928 / 0.4684 / 2.600 | **FAILED** | Diverged or low-quality output. |
| `ssim_lr3e-3_92c8b6` | `EARLY_STOPPING` | 0.7310 / 0.8358 / 0.0412 / 20.162 | **BASELINE** | Stable comparator baseline. |
| `ssim_lr3e-3_94f982` | `EARLY_STOPPING` | 0.7464 / 0.8478 / 0.0412 / 19.835 | **SELECTED** | Selected for next-batch seeds and DCM/image comparison. |
| `ssim_lr8e-4_1b8c15` | `EARLY_STOPPING` | 0.2096 / 0.0901 / 12.1586 / -30.513 | **FAILED** | Diverged or low-quality output. |

### Selected Set (Max 5) for Next Batch + DCM/Test Images

1. `ssim_lr3e-3_94f982` (ssim, lr=0.003, bs=32)
   - Best checkpoint: `experiments/train_nn1_cropped/ssim_lr3e-3_94f982/epochs/43/weights.pth`
   - Best metrics: `SSIM=0.7464`, `MAE=0.0412`, `PSNR=19.835`, `valid_loss=0.25361`
   - Why selected: Highest SSIM among stable runs (best perceptual structure).
2. `ssim+l1_lr8e-4_1fa946` (ssim+l1, lr=0.0008, bs=96)
   - Best checkpoint: `experiments/train_nn1_cropped/ssim+l1_lr8e-4_1fa946/epochs/70/weights.pth`
   - Best metrics: `SSIM=0.7422`, `MAE=0.0363`, `PSNR=20.241`, `valid_loss=0.03673`
   - Why selected: Best stable SSIM+L1 trade-off (high SSIM with very low MAE).
3. `l1_lr8e-4_b39be9` (l1, lr=0.0008, bs=96)
   - Best checkpoint: `experiments/train_nn1_cropped/l1_lr8e-4_b39be9/epochs/129/weights.pth`
   - Best metrics: `SSIM=0.7415`, `MAE=0.0360`, `PSNR=20.031`, `valid_loss=0.03603`
   - Why selected: Lowest MAE among full stable runs; reliable detail baseline.
4. `mse_lr8e-4_b558b9` (mse, lr=0.0008, bs=96)
   - Best checkpoint: `experiments/train_nn1_cropped/mse_lr8e-4_b558b9/epochs/44/weights.pth`
   - Best metrics: `SSIM=0.7207`, `MAE=0.0408`, `PSNR=20.540`, `valid_loss=0.00905`
   - Why selected: Most PSNR-efficient stable baseline; useful smoothing reference.
5. `msssim+l1_lr3e-4_9b7aea` (msssim+l1, lr=0.0003, bs=64)
   - Best checkpoint: `experiments/train_nn1_cropped/msssim+l1_lr3e-4_9b7aea/epochs/18/weights.pth`
   - Best metrics: `SSIM=0.7424`, `MAE=0.0366`, `PSNR=20.160`, `valid_loss=0.05884`
   - Why selected: Most promising MS-SSIM+L1 checkpoint quality; good next-batch continuation seed.

### Practical Note for DCM/Image Generation
- For the NaN-terminated selected run (`msssim+l1_lr3e-4_9b7aea`), generate outputs from the listed best checkpoint, not from latest state.
- If you prefer a fully stable replacement for that slot, swap in `msssim+l1_lr8e-4_bc1d65` (stable backup).

---

## [2026-02-12] Final Chosen Experiments for Next Batch (Confirmed)

These are the **final 5 experiments** to use as the base for:
- next batch of training experiments, and
- DCM + test-set image generation for visual inspection/comparison.

1. `ssim_lr3e-3_94f982`
   - Checkpoint: `experiments/train_nn1_cropped/ssim_lr3e-3_94f982/epochs/43/weights.pth`
2. `ssim+l1_lr8e-4_1fa946`
   - Checkpoint: `experiments/train_nn1_cropped/ssim+l1_lr8e-4_1fa946/epochs/70/weights.pth`
3. `l1_lr8e-4_b39be9`
   - Checkpoint: `experiments/train_nn1_cropped/l1_lr8e-4_b39be9/epochs/129/weights.pth`
4. `mse_lr8e-4_b558b9`
   - Checkpoint: `experiments/train_nn1_cropped/mse_lr8e-4_b558b9/epochs/44/weights.pth`
5. `msssim+l1_lr1e-4_7ee268`  *(selected over `msssim+l1_lr3e-4_9b7aea`)*
   - Checkpoint: `experiments/train_nn1_cropped/msssim+l1_lr1e-4_7ee268/epochs/2/weights.pth`

Note: `msssim+l1_lr1e-4_7ee268` ended as `NAN_VALUE_DETECTED`, so always use the listed best checkpoint for downstream generation/evaluation.

---

## [2026-03-03] Best Experiment Per Metric (Validation Set) — Final Summary

Across all 20 completed experiments (EARLY_STOPPING or FINISHED_EPOCHS status), the top 12 have SSIM > 0.7 and are usable. The best per metric on the **validation set** (best-epoch checkpoints):

| Metric | Best Experiment | Value | Loss Function | LR | Best Epoch |
|--------|----------------|-------|---------------|----|------------|
| **SSIM** | `ssim_lr3e-3_94f982` | **0.7464** | ssim | 3e-3 | 44 |
| **MS-SSIM** | `ssim_lr3e-3_94f982` | **0.8478** | ssim | 3e-3 | 44 |
| **MAE** | `l1_lr8e-4_b39be9` | **0.0360** | l1 | 8e-4 | 130 |
| **PSNR** | `mse_lr8e-4_b558b9` | **20.54 dB** | mse | 8e-4 | 45 |
| **Gradient MAE** | `ssim_lr3e-3_94f982` | **0.1418** | ssim | 3e-3 | 44 |

**Key observations:**
- `ssim_lr3e-3_94f982` dominates 3 of 5 metrics (SSIM, MS-SSIM, Gradient MAE) — best perceptual quality.
- `l1_lr8e-4_b39be9` has the lowest pixel-level error (MAE).
- `mse_lr8e-4_b558b9` has the highest PSNR (MSE directly optimizes for this).
- `ssim+l1_lr8e-4_1fa946` is the best balanced model: SSIM=0.742, MAE=0.036, PSNR=20.24.
- `msssim+l1_lr1e-4_7ee268` (transfer-initialized, NaN at epoch 7, checkpoint from epoch 3): SSIM=0.743, MAE=0.036 — competitive despite instability.

### Best Per Loss Family (Stable Runs)

| Loss Family | Best Experiment | SSIM | MAE | PSNR |
|-------------|----------------|------|-----|------|
| MSE | `mse_lr8e-4_b558b9` | 0.721 | 0.041 | **20.54** |
| L1 | `l1_lr8e-4_b39be9` | 0.741 | **0.036** | 20.03 |
| SSIM | `ssim_lr3e-3_94f982` | **0.746** | 0.041 | 19.84 |
| SSIM+L1 | `ssim+l1_lr8e-4_1fa946` | 0.742 | 0.036 | 20.24 |
| MSSSIM+L1 | `msssim+l1_lr8e-4_bc1d65` (stable) | 0.738 | 0.037 | 20.22 |

### Test Set Export Validation (30 patients)

All 5 final chosen experiments have been exported to `output/test_interpolated_export/train_nn1_cropped/`:

| Experiment | Epoch (0-indexed) | Patients | DICOM Consistency |
|---|---|---|---|
| `ssim_lr3e-3_94f982` | epoch_43 | 30 | OK |
| `ssim+l1_lr8e-4_1fa946` | epoch_70 | 30 | OK |
| `l1_lr8e-4_b39be9` | epoch_129 | 30 | OK |
| `mse_lr8e-4_b558b9` | epoch_44 | 30 | OK |
| `msssim+l1_lr1e-4_7ee268` | epoch_2 | 30 | OK |

**DICOM validation (2026-03-03):**
- `dcm_mixed/` (real+AI interleaved as Secondary Capture): All slices are consistent 8-bit uint8, WindowCenter=127.5, WindowWidth=255.0, Modality=OT, no RescaleSlope/Intercept. **Use this directory for Weasis viewing.**
- `dcm_real/` (original raw DICOMs preserved for reference): 16-bit int16, RescaleSlope=1.0, RescaleIntercept=-1024, WindowCenter=35, WindowWidth=135, Modality=CT. **Do NOT mix with dcm_mixed when viewing** — different encoding causes visual jumps between slices.

**Paper relevance**: Results section (quantitative comparison table), Discussion section (loss function analysis).

---

## [2026-03-03] Selected Test Patients for VM Demo (6 of 30)

From the 30 test patients, 6 were selected to create a portable export (< 10 GB with experiment weights) for visual inspection on a VM. Selection criteria: cover all 5 hemorrhage subtypes plus one normal case, prefer smaller slice counts, and ensure visually diverse pathology.

### Selection

| # | Patient | Slices | Hemorrhage | Types | Positive slices | Why selected |
|---|---------|--------|------------|-------|-----------------|--------------|
| 1 | `ID_615f69e3` | 32 | No | none | 0 | **Normal control** — clean brain anatomy for baseline visual comparison |
| 2 | `ID_8c298617` | 32 | Yes | epidural | 5 | **Epidural only** — subtle finding (5/32 slices); tests whether AI preserves rare pathology |
| 3 | `ID_db7578d3` | 32 | Yes | subdural | 8 | **Subdural only** — classic bright collection along posterior skull, visually striking |
| 4 | `ID_d53d2b8d` | 32 | Yes | intraparenchymal | 9 | **Intraparenchymal only** — focal bleeding within brain parenchyma |
| 5 | `ID_cc734ebe` | 40 | Yes | intraventricular, subarachnoid | 20 | **Multi-type** — two hemorrhage types, longer series (40 slices) for scrolling comparison |
| 6 | `ID_fc4fcd34` | 34 | Yes | epidural, intraparenchymal, intraventricular, subarachnoid, subdural | 23 | **All 5 types** — severe case with midline shift; stress-tests AI interpolation on complex anatomy |

### Hemorrhage Coverage

| Subtype | Covered by patients |
|---------|-------------------|
| Epidural | #2, #6 |
| Intraparenchymal | #4, #6 |
| Intraventricular | #5, #6 |
| Subarachnoid | #5, #6 |
| Subdural | #3, #6 |
| Normal (no hemorrhage) | #1 |

### Export Contents

- **Included**: `dcm_mixed/` (real+AI as unified 8-bit Secondary Capture series), `viz/` (PNG visualizations), `info.json` (slice metadata)
- **Excluded**: `dcm_real/` (raw 16-bit DICOMs — redundant, can be regenerated from raw dataset)
- **Size**: ~849 MB across 6 patients × 5 experiments
- **Zip file**: `test_export_selected_6patients.zip`

### VM Storage Budget

| File | Size |
|------|------|
| `experiments_best5.zip` (5 best experiments with weights) | 2.0 GB |
| `test_export_selected_6patients.zip` (6 patients × 5 experiments) | ~0.9 GB |
| **Total** | **~2.9 GB** (well within 10 GB VM limit) |

**Paper relevance**: Visual inspection figures, DICOM viewer screenshots for qualitative comparison.

---

## [2026-03-11] VFI Baselines: RIFE and FILM

### Hypothesis
- **Motivation**: Reviewer feedback noted the absence of comparison against state-of-the-art video frame interpolation (VFI) methods. RIFE and FILM are leading optical-flow-based VFI methods trained on natural video.
- **Expected outcome**: VFI methods should produce perceptually natural outputs but lack CT-specific structural fidelity, underperforming our domain-specific models on SSIM/MAE/PSNR.
- **Key question**: How large is the domain gap between pretrained natural-video VFI and domain-specific CT slice interpolation?

### Results
- **Status**: COMPLETED
- **Method**: Off-the-shelf pretrained weights (no fine-tuning on CT data); grayscale-to-RGB conversion (channel repeat) for inference
- **Script**: `scripts/evaluate_vfi_baselines.py --device cuda --batch-size 8`

**Patient-level results (30 patients, 968 triplets):**

| Method | SSIM | MS-SSIM | MAE | Grad-MAE | PSNR | LPIPS |
|--------|------|---------|-----|----------|------|-------|
| RIFE | 0.848 [0.839, 0.856] | 0.887 [0.882, 0.893] | 0.0262 [0.0249, 0.0276] | 0.0798 [0.076, 0.084] | 22.31 [21.86, 22.76] | 0.083 [0.080, 0.087] |
| FILM | 0.848 [0.840, 0.857] | 0.891 [0.885, 0.896] | 0.0255 [0.024, 0.027] | 0.0817 [0.077, 0.086] | 22.42 [21.94, 22.90] | 0.078 [0.074, 0.081] |

**Paired tests vs reference (MS-SSIM+L1@8e-4), all p < 1e-8:**
- RIFE: ΔSSIM = −0.041, ΔMAE = −0.009, ΔPSNR = −3.114, ΔLPIPS = +0.046
- FILM: ΔSSIM = −0.040, ΔMAE = −0.008, ΔPSNR = −2.996, ΔLPIPS = +0.052

- **Observations**: VFI methods sit between classical baselines (SSIM ~0.78) and our models (SSIM ~0.89) on structural metrics. Most striking: VFI achieves the best LPIPS of ANY method (0.078-0.083), dramatically better than both classical baselines (0.123-0.126) and all learned models (0.124-0.154). This strongly reinforces that LPIPS captures perceptual naturalness rather than anatomical fidelity.
- **Conclusion**: Hypothesis confirmed. Domain gap is large (~0.04 SSIM, ~3 dB PSNR), validating the need for domain-specific training. The inverted LPIPS ranking is the most important finding — it provides the strongest evidence yet that LPIPS alone is insufficient for evaluating medical image synthesis quality.
- **Paper relevance**: Added to Tables 2 and 3, Discussion (LPIPS section), Limitations (updated), Abstract, Conclusion.
