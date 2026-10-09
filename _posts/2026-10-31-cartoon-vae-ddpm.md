---
title: "Implementation Details of Cartoon-VAE-DDPM"
date: 2026-10-31

permalink: /posts/cartoon-vae-ddpm
tags:
  - Generative Models
  - Diffusion Models
  - Computer Vision
---
**Variational Autoencoder (VAE)** learns to compress data into a latent Gaussian space and reconstruct it in a single shot. [**Denoising Diffusion Probabilistic Model (DDPM)**](https://arxiv.org/pdf/2006.11239) tackles the same evidence-lower-bound objective from another direction: it begins with pure noise and iteratively denoise through hundreds of steps, exchanging speed for high-fidelity, stable synthesis. Both frameworks connect random noise to data, yet VAE rely on an explicit **encoder–decoder pair**, whereas DDPM use a learned Markov chain that inverts a forward noising process. This blog traces the progression from VAE to DDPM, clarifying their shared principles, with code examples available at this <i class="fa-brands fa-github"></i> [repository](https://github.com/lihanlian/cartoon-vae-ddpm).

## 1. Image Generation as a Maximum-Likelihood Problem

Given a dataset $$\mathcal{D}=\{x^{(i)}\}_{i=1}^{N}$$, we want to learn a distribution over images from which new images can be generated. A variational autoencoder (VAE) approaches this through a **latent-variable generative model**.

- ### 1.1 Notation

  **Variables and network parameters**

  | Symbol | Meaning |
  |---|---|
  | $$x\in\mathbb{R}^{D}$$ | An image represented by its pixel values. During a single-example loss calculation, $$x$$ is fixed. |
  | $$z\in\mathbb{R}^{d}$$ | An unobserved latent vector; the dataset provides no corresponding latent labels. |
  | $$\theta$$ | Decoder network's trainable parameters. |
  | $$\phi$$ | Encoder network's trainable parameters. |
  | $$N$$ | Number of training images. |
  | $$D,d$$ | Image dimension and latent dimension, respectively. |

  **Distributions**

  | Distribution | Statistical meaning | Role |
  |---|---|---|
  | $$p(z)$$ | Prior over latent variables | Usually fixed as $$\mathcal{N}(0,I)$$. |
  | $$p_\theta(x\mid z)$$ | Conditional image likelihood | Probabilistic decoder. |
  | $$p_\theta(x,z)$$ | Joint distribution | Complete generative model. |
  | $$p_\theta(x)$$ | <span style="color:red">Marginal likelihood, or evidence </span> | <span style="color:red"> Quantity targeted by maximum likelihood. </span> |
  | $$p_\theta(z\mid x)$$ | Exact posterior under the current model | Implied by the prior and decoder. |
  | $$q_\phi(z\mid x)$$ | Approximate posterior | Probabilistic encoder. |

  For continuous variables, these functions represent probability **densities**, not probabilities of individual exact vectors.

- ### 1.2 Generative model and marginalization

  The model generates an image in two stages:

  $$
  z\sim p(z)=\mathcal{N}(0,I),
  \qquad
  x\sim p_\theta(x\mid z).
  $$

  The decoder takes a latent vector and predicts distribution parameters **in pixel space**. <span style="color:red"> For example, a Gaussian decoder predicts an image-shaped mean. We can sample pixels from its distribution or display the mean directly. </span> The decoder distribution need not be Gaussian. ([Doersch, 2016][doersch])

  The probability chain rule gives

  $$
  \boxed{
  p_\theta(x,z)=p(z)p_\theta(x\mid z).
  }
  $$

  Because the dataset contains images but not their latent variables, the image likelihood includes all possible latent explanations:

  $$
  \boxed{
  p_\theta(x)
  =
  \int p_\theta(x,z)\,dz
  =
  \int p(z)p_\theta(x\mid z)\,dz.
  }
  $$

  **The joint distribution is not a replacement objective.** It connects the image to its hidden explanation; marginalizing it gives the image likelihood.

- ### 1.3 Maximum likelihood estimation

  For independent training images, maximum likelihood estimation (MLE) seeks

  $$
  \begin{aligned}
  &\max_\theta
  \sum_{i=1}^{N}\log p_\theta(x^{(i)})
  \\
  &\qquad=
  \max_\theta
  \sum_{i=1}^{N}
  \log\int p(z)p_\theta(x^{(i)}\mid z)\,dz.
  \end{aligned}
  $$

  Evaluating the decoder likelihood at a specified latent code is generally straightforward. <span style="color:red"> The difficulty is integrating over all codes:</span> nonlinear neural decoders generally lack an analytical marginal likelihood, while accurate numerical integration can be expensive. ([Kingma and Welling, 2013][aevb])

## - Variational Autoencoders (VAE)
Let's first dive into the technical details of VAE:

## 2. Variational Inference: Approximating the Latent Posterior

- ### 2.1 Bayesian inference

  <span style="color:red"> Fix the current decoder parameters $$\theta$$ and observe an image $$x$$. </span> Bayes' rule defines the distribution of latent codes that could explain it:

  $$
  \boxed{
  p_\theta(z\mid x)
  =
  \frac{p(z)p_\theta(x\mid z)}
  {\int p(z')p_\theta(x\mid z')\,dz'}.
  }
  $$

  Conceptually, each candidate latent code goes **forward through the decoder**. We evaluate how well its predicted distribution explains the observed image, weight this by the code's prior density, and normalize across all candidates.

  This does not mean feeding the image into the decoder or reversing the network. **The posterior depends on the same $$\theta$$ because Bayes' rule uses the decoder likelihood.** "Exact" means exact under the current model, even if that model poorly represents real images.

  The normalization requires the same difficult marginal likelihood, so ordinary VAE training does not explicitly calculate this posterior. ([Kingma and Welling, 2019][introduction])

- ### 2.2 From inference to optimization

  **Variational inference replaces difficult posterior computation with optimization over a tractable family of distributions.** For a fixed image and decoder, the approximation target is

  $$
  \boxed{
  \min_\phi
  D_{\mathrm{KL}}\!\left(
  q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)
  \right),
  \qquad x,\theta\text{ fixed}.
  }
  $$

  Both distributions concern the **same latent variable conditioned on the same image**. We approximate a distribution of explanations, rather than select one latent vector. ([Blei et al., 2017][vi])

  A common encoder parameterization is

  $$
  \boxed{
  q_\phi(z\mid x)
  =
  \mathcal{N}\!\left(
  z;
  \mu_\phi(x),
  \operatorname{diag}\!\left(\sigma_\phi^2(x)\right)
  \right).
  }
  $$

  Unlike the decoder, the encoder takes the image as input and predicts latent means and standard deviations. Sharing this network across images is **amortized variational inference**. Choosing a Gaussian approximation does not imply that the exact posterior is Gaussian. ([Kingma and Welling, 2019][introduction])

  The target KL still involves the intractable posterior. We therefore need an equivalent encoder-training objective that avoids evaluating its normalization.

## 3. The Evidence Lower Bound and VAE Loss

- ### 3.1 Deriving the ELBO

  For a fixed image $$x$$, expectations are taken over latent codes sampled from the encoder distribution:

  $$
  \mathbb{E}_{q_\phi(z\mid x)}[f(z)]
  =
  \int q_\phi(z\mid x)f(z)\,dz.
  $$

  By definition, the **KL divergence between the approximate and exact posteriors** is

  $$
  \begin{aligned}
  &D_{\mathrm{KL}}\!\left(
  q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)
  \right)
  \\[4pt]
  &=
  \int q_\phi(z\mid x)
  \log\frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}
  \,dz
  \\[4pt]
  &=
  \int q_\phi(z\mid x)
  \left[
  \log q_\phi(z\mid x)
  -
  \log p_\theta(z\mid x)
  \right]dz
  \\[4pt]
  &=
  \mathbb{E}_{q_\phi(z\mid x)}\!\left[
  \log q_\phi(z\mid x)
  -
  \log p_\theta(z\mid x)
  \right].
  \end{aligned}
  $$

  The second equality uses the logarithm-of-a-ratio identity. The last equality uses the expectation–integral identity with

  $$
  f(z)
  =
  \log q_\phi(z\mid x)
  -
  \log p_\theta(z\mid x).
  $$

  Next, Bayes' rule gives

  $$
  p_\theta(z\mid x)
  =
  \frac{p_\theta(x,z)}{p_\theta(x)},
  $$

  so

  $$
  \log p_\theta(z\mid x)
  =
  \log p_\theta(x,z)
  -
  \log p_\theta(x).
  $$

  Substituting this into the KL expression and using linearity of expectation,

  $$
  \begin{aligned}
  &D_{\mathrm{KL}}\!\left(
  q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)
  \right)
  \\[4pt]
  &=
  \mathbb{E}_{q_\phi(z\mid x)}\!\left[
  \log q_\phi(z\mid x)
  -
  \log p_\theta(x,z)
  +
  \log p_\theta(x)
  \right]
  \\[4pt]
  &=
  \mathbb{E}_{q_\phi(z\mid x)}\!\left[
  \log q_\phi(z\mid x)
  -
  \log p_\theta(x,z)
  \right]
  +
  \mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x)].
  \end{aligned}
  $$

  **The evidence term leaves the expectation because it does not depend on the integration variable $$z$$.** With $$x$$ and $$\theta$$ fixed, the expectation–integral identity gives

  $$
  \begin{aligned}
  \mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x)]
  &=
  \int q_\phi(z\mid x)\log p_\theta(x)\,dz
  \\[4pt]
  &=
  \log p_\theta(x)
  \underbrace{
  \int q_\phi(z\mid x)\,dz
  }_{1}
  \\[4pt]
  &=
  \log p_\theta(x).
  \end{aligned}
  $$

  The integral equals one because the encoder distribution is normalized. Therefore,

  $$
  \begin{aligned}
  &D_{\mathrm{KL}}\!\left(
  q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)
  \right)
  \\[4pt]
  &=
  \log p_\theta(x)
  -
  \mathbb{E}_{q_\phi(z\mid x)}\!\left[
  \log p_\theta(x,z)
  -
  \log q_\phi(z\mid x)
  \right].
  \end{aligned}
  $$

  Define the **evidence lower bound (ELBO)**:

  $$
  \boxed{
  \mathcal{E}(\theta,\phi;x)
  =
  \mathbb{E}_{q_\phi(z\mid x)}\!\left[
  \log p_\theta(x,z)
  -
  \log q_\phi(z\mid x)
  \right].
  }
  $$

  Consequently,

  $$
  \begin{aligned}
  \log p_\theta(x)
  &=
  \mathcal{E}(\theta,\phi;x)
  +
  D_{\mathrm{KL}}\!\left(
  q_\phi(z\mid x)\,\Vert\,p_\theta(z\mid x)
  \right)
  \\[4pt]
  &\geq
  \mathcal{E}(\theta,\phi;x).
  \end{aligned}
  $$

  The inequality follows from nonnegative KL divergence. For fixed $$\theta$$, maximizing the ELBO over $$\phi$$ minimizes posterior-approximation error. Equality holds when the approximate and exact posteriors coincide; otherwise, the ELBO is a surrogate for the MLE objective. ([Blei et al., 2017][vi])

- ### 3.2 Recovering the two loss terms

  Substitute the generative factorization of the joint:

  $$
  \begin{aligned}
  \mathcal{E}(\theta,\phi;x)
  &=
  \mathbb{E}_{q_\phi(z\mid x)}\!\left[
  \log p_\theta(x\mid z)
  +
  \log p(z)
  -
  \log q_\phi(z\mid x)
  \right]
  \\[4pt]
  &=
  \mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
  -
  D_{\mathrm{KL}}\!\left(
  q_\phi(z\mid x)\,\Vert\,p(z)
  \right).
  \end{aligned}
  $$

  Negating the ELBO gives the minimized loss:

  $$
  \boxed{
  \begin{aligned}
  \mathcal{J}(\theta,\phi;x)
  ={}&
  \underbrace{
  -\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
  }_{\text{reconstruction loss}}
  \\[4pt]
  &+
  \underbrace{
  D_{\mathrm{KL}}\!\left(
  q_\phi(z\mid x)\,\Vert\,p(z)
  \right)
  }_{\text{prior-matching KL}}.
  \end{aligned}
  }
  $$

  **The original image is the $$x$$ inside the reconstruction term.** Whereas $$p_\theta(\,\cdot\mid z)$$ denotes the predicted distribution, $$p_\theta(x\mid z)$$ is its scalar density **evaluated at the observed image**. Its negative logarithm penalizes predictions that explain that image poorly. ([Kingma and Welling, 2013][aevb])

  The explicit KL penalizes departure from the **prior**. It differs from the **posterior KL** measuring the ELBO's gap. Together, reconstruction and prior matching implement the variational objective.

  Across the dataset, training solves

  $$
  \min_{\theta,\phi}
  \frac{1}{N}
  \sum_{i=1}^{N}
  \mathcal{J}(\theta,\phi;x^{(i)}).
  $$

- ### 3.3 Concrete Gaussian terms

  Choose a fixed-variance Gaussian decoder:

  $$
  p_\theta(x\mid z)
  =
  \mathcal{N}\!\left(x;m_\theta(z),\tau^2I\right).
  $$

  Its negative log-likelihood is

  $$
  -\log p_\theta(x\mid z)
  =
  \frac{1}{2\tau^2}
  \sum_{j=1}^{D}
  \left(x_j-m_{\theta,j}(z)\right)^2
  +
  C,
  $$

  where

  $$
  C=\frac{D}{2}\log(2\pi\tau^2).
  $$

  **This explicitly compares original pixel values with predicted pixel means.** The reconstruction term averages that comparison over encoder samples:

  $$
  -\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
  =
  \frac{
  \mathbb{E}_{q_\phi(z\mid x)}
  [\lVert x-m_\theta(z)\rVert^2]
  }{2\tau^2}
  +
  C.
  $$

  <span style="color:red">Squared error therefore follows from a fixed-variance Gaussian likelihood.</span> A Bernoulli likelihood for binary pixels instead produces binary cross-entropy. ([Doersch, 2016][doersch])

  For the diagonal-Gaussian encoder, abbreviate its outputs as $$\mu_j=\mu_{\phi,j}(x)$$ and $$\sigma_j=\sigma_{\phi,j}(x)$$. Using

  $$
  \begin{aligned}
  \mathbb{E}_{q_\phi(z\mid x)}[(z_j-\mu_j)^2]&=\sigma_j^2,\\
  \mathbb{E}_{q_\phi(z\mid x)}[z_j^2]&=\mu_j^2+\sigma_j^2,
  \end{aligned}
  $$

  the KL to the standard-normal prior becomes

  $$
  \boxed{
  D_{\mathrm{KL}}\!\left(
  q_\phi(z\mid x)\,\Vert\,p(z)
  \right)
  =
  \frac12\sum_{j=1}^{d}
  \left(
  \mu_j^2+\sigma_j^2-1-\log\sigma_j^2
  \right).
  }
  $$

- ### 3.4 Sampling and joint training

  Estimate the reconstruction expectation using the **reparameterization trick**:

  $$
  \epsilon\sim\mathcal{N}(0,I),
  \qquad
  z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon.
  $$

  Here $$\odot$$ denotes elementwise multiplication. This produces image-dependent encoder samples while preserving a differentiable path through their parameters. A one-sample loss estimate is

  $$
  \widehat{\mathcal{J}}
  =
  -\log p_\theta(x\mid z)
  +
  D_{\mathrm{KL}}\!\left(
  q_\phi(z\mid x)\,\Vert\,p(z)
  \right).
  $$

  <span style="color:red">The same image is both the **encoder input and reconstruction target**.</span> We evaluate its likelihood without needing to sample another noisy image. With a fixed prior and separate weights, the decoder receives reconstruction gradients; the encoder receives both reconstruction and KL gradients. ([Kingma and Welling, 2013][aevb]; [Rezende et al., 2014][rezende])

  After training, unconditional generation samples $$z\sim p(z)$$ and uses the decoder, without the encoder.

  **The prior is chosen, the exact posterior is implied, and the encoder learns an approximation. The VAE loss connects them through a lower bound on image likelihood.**

  [aevb]: https://arxiv.org/abs/1312.6114 "Auto-Encoding Variational Bayes"
  [introduction]: https://arxiv.org/abs/1906.02691 "An Introduction to Variational Autoencoders"
  [doersch]: https://arxiv.org/abs/1606.05908 "Tutorial on Variational Autoencoders"
  [vi]: https://arxiv.org/abs/1601.00670 "Variational Inference: A Review for Statisticians"
  [rezende]: https://proceedings.mlr.press/v32/rezende14.html "Stochastic Backpropagation and Approximate Inference in Deep Generative Models"

 - ### VAE Results

  <figure style="display: block; margin: 0 auto; width: 80%;">
    <img src='/images/blog/blog8/vae_result.png' style="width: 100%;">
    <figcaption style="text-align: center;">VAE sampling result. Trained with z_dim = 512, epochs = 100.</figcaption>
  </figure>

## Diffusion Models

## DDPM

- ### Results

## References
 1. <i class="fab fa-youtube"></i> [Variational Autoencoder - Model, ELBO, loss function and maths explained easily!](https://www.youtube.com/watch?v=iwEzwTTalbg) 
 2. <i class="fab fa-youtube"></i> [Understanding Variational Autoencoders (VAEs)](https://www.youtube.com/watch?v=HBYQvKlaE0A), [Variational Autoencoders \| Generative AI Animated](https://www.youtube.com/watch?v=qJeaCHQ1k2w&t=225s)
 3. <i class="fab fa-youtube"></i> [The Breakthrough Behind Modern AI Image Generators - Diffusion Models Part 1](https://www.youtube.com/watch?v=1pgiu--4W3I&t=1s) 
 4. <i class="fa-solid fa-book-open"></i> [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (VAE Paper)
 5. <i class="fa-solid fa-book-open"></i> [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239) (DDPM Paper), [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (DDIM Paper) 
 6. <i class="fab fa-youtube"></i> [Diffusion Model Paper Explanation](https://www.youtube.com/watch?v=HoKDTa5jHvg), [PyTorch Implementation Walk Through](https://www.youtube.com/watch?v=TBCRlnwJtZU&t=874s) and corresponding <i class="fa-brands fa-github"></i> [github repo](https://github.com/dome272/Diffusion-Models-pytorch)
 7. <i class="fa-brands fa-github"></i> [diffusion-DDPM-pytorch](https://github.com/Alokia/diffusion-DDPM-pytorch) & [diffusion-DDIM-pytorch](https://github.com/Alokia/diffusion-DDIM-pytorch)
 8. <i class="fab fa-youtube"></i> [An Optimal Control Perspective on Diffusion-Based Generative Modeling](https://www.youtube.com/watch?v=wQpQg1xIlBA&list=LL&index=3&t=2299s) & [SDE/ODE Interpretation of Diffusion Model](https://www.youtube.com/watch?v=Ro4v4z8YAsk&list=LL&index=3)
