---
title: "Understanding Score-Based Diffusion and Flow Matching with PyTorch"
date: 2026-12-31
permalink: /posts/score-based-diffusion-and-flow-matching
tags:
  - Generative Models
  - Diffusion Models
  - PyTorch
---

Building on [VAE and DDPM](/posts/cartoon-vae-ddpm), this post explores learning scores and velocities to transform noise into data. The learning path moves from intuition to small PyTorch implementations, then compares the approaches.

**Prerequisites:** DDPM training and sampling, Gaussian distributions, gradients, and basic PyTorch. Review ordinary and stochastic differential equations (ODEs and SDEs) as needed.

## Part I: Score-Based Diffusion

### 1. From DDPM Noise Prediction to Scores

- Introduce the score of a probability density and its geometric meaning.
- Connect DDPM noise prediction to estimating scores at different noise levels.

### 2. Learning Scores with Denoising Score Matching

- Outline noise perturbations, time conditioning, training targets, and loss weighting.
- Identify how these choices map to a PyTorch training loop.

### 3. Continuous-Time Diffusion and Sampling

- Introduce forward SDEs and reverse-time generation.
- Compare reverse-SDE sampling with the probability flow ODE.
- State the time convention and distinguish shared marginal distributions from individual trajectories.

## Part II: Flow Matching

### 4. From Scores to Velocity Fields

- Use the probability flow ODE as the bridge to learning a velocity field directly.
- Introduce probability paths and distinguish the training objective from the sampling procedure.

### 5. Conditional Flow Matching

- Begin with independent Gaussian noise and data pairs, linear interpolation, and velocity regression.
- Outline the training loop and ODE sampling.
- Distinguish conditional training paths from learned sampling trajectories; place rectified flow and optimal transport as further study.

## Part III: Implementation and Comparison

### 6. A Minimal 2D PyTorch Project

- Train a score model and a flow-matching model on two moons or Gaussian mixtures.
- Visualize learned vector fields, intermediate distributions, and sample trajectories.
- Implement reverse-SDE, probability-flow-ODE, and flow-matching-ODE samplers.

### 7. Extending to Cartoon Images

- Reuse the CartoonSet pipeline and adapt a time-conditioned U-Net.
- Compare sample quality and diversity, runtime, and network function evaluations.
- Keep model capacity and training budgets comparable; vary sampling steps.

### 8. Connections and Lessons Learned

- Revisit the Gaussian diffusion–flow-matching connection using [Diffusion Meets Flow Matching](https://diffusionflow.github.io/).
- Separate the effects of paths, prediction targets, loss weighting, and numerical solvers.
- Record implementation pitfalls, experimental observations, and open questions.

## References

- [Diffusion Meets Flow Matching: Two Sides of the Same Coin](https://diffusionflow.github.io/)
- [Score-Based Generative Modeling: Official PyTorch Implementation](https://github.com/yang-song/score_sde_pytorch)
- [Flow Matching: Meta's PyTorch Library and Examples](https://github.com/facebookresearch/flow_matching)
