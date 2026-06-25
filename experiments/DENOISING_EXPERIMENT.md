# Denoising Experiment: Repurposing the Interpolation Model

## Motivation

Our interpolation model, trained to predict slice k+1 from (slice k, slice k+2), inherently produces denoised outputs because regression losses converge to the conditional expectation (see Noise2Noise, Lehtinen et al. 2018). This means we can repurpose the same model — without any retraining — as a CT denoiser for the existing acquired slices.

## Core Idea

Given a volume with N acquired (noisy) slices [S1, S2, S3, ..., SN]:

**Denoising pass:** For each slice Si where 2 <= i <= N-1, feed (S_{i-1}, S_{i+1}) into the model. The output is a denoised version of Si, because the model predicts the conditional expectation of the middle slice given its neighbors.

- S1 and SN cannot be denoised (no neighbor on one side)
- All other slices get a denoised version

**Interpolation pass (standard):** For each consecutive pair of original noisy slices (Si, S_{i+1}), feed them into the model to produce a new intermediate slice. This is the existing interpolation task — unchanged.

## Output Directories

### Directory 1: Denoised-only

Contains only the denoised versions of real slices:

```
denoised_S2, denoised_S3, denoised_S4, ..., denoised_S{N-1}
```

- N-2 slices total (first and last are lost)
- Same spatial positions as the originals, just denoised
- Useful for direct comparison with originals (noise reduction metrics, visual quality)

### Directory 2: Denoised + AI interpolated (double resolution)

Interleaves denoised real slices with AI-generated intermediate slices:

```
denoised_S2, AI_between_S2_S3, denoised_S3, AI_between_S3_S4, denoised_S4, ...
```

- The denoised slices come from the denoising pass above
- The AI interpolated slices come from the standard interpolation pass using the **original noisy** slices as input (NOT the denoised outputs — the model was trained on noisy inputs and we have not validated it on denoised inputs)
- This produces a volume with ~2x the slice count, where every slice is noise-free
- Similar to the existing "mixed" directory structure but with denoised real slices instead of noisy ones

### Existing "mixed" directory (for comparison)

The current mixed output has:

```
real_S1, AI_between_S1_S2, real_S2, AI_between_S2_S3, real_S3, ...
```

Here real slices are noisy and AI slices are denoised. Directory 2 above replaces the noisy real slices with denoised versions, creating a fully noise-free volume.

## Implementation Notes

- Use the same model checkpoint (best-SSIM or reference MS-SSIM+L1 model)
- The denoising pass and interpolation pass are independent — both use original noisy slices as input
- Slices must be sorted by DICOM position (already handled by existing data pipeline)
- The denoising "trick" requires no new training, no new architecture, no changes to the model — it is purely a different application of the same inference

## What This Demonstrates

1. **Dual-purpose model:** One trained model provides both interpolation (new slices) and denoising (cleaner existing slices) — two clinical benefits from a single architecture
2. **No additional training cost:** The denoising capability is free — it falls directly from the regression loss formulation
3. **Concrete evidence for the paper's denoising claim:** Currently the paper argues theoretically that outputs are denoised. This experiment provides direct visual and quantitative evidence by comparing denoised outputs against the noisy originals at the same spatial position

## Metrics to Compute

- Visual comparison: noisy original vs denoised version (same slice, same position)
- Noise standard deviation in homogeneous regions (e.g., background air, uniform brain parenchyma)
- SSIM/PSNR between denoised and original (lower SSIM here is expected and desirable — it means noise was removed)
- Possibly: have the denoised volume reviewed by Itziar alongside the original

## Relationship to Paper

This experiment strengthens Section 5.5 ("Implicit denoising as a property of regression-based synthesis") with concrete evidence rather than purely theoretical argument. It could be presented as:
- A new figure showing original vs denoised slice side-by-side
- A noise measurement table (noise std in homogeneous regions)
- A qualitative panel in the paper showing the fully denoised + interpolated volume
