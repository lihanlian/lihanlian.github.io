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

## Part I: Variational Autoencoders (VAE)

We begin with the VAE’s latent-variable model, derive its variational training objective, and connect the reconstruction and KL terms to the implementation.

### 1. Image Generation as a Maximum-Likelihood Problem

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

### 2. Variational Inference: Approximating the Latent Posterior

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

### 3. The Evidence Lower Bound and VAE Loss

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

### VAE Results
  <figure style="display: block; margin: 0 auto; width: 80%;">
    <img src='/images/blog/cartoon-vae-ddpm/vae_result.png' style="width: 100%;">
    <figcaption style="text-align: center;">VAE sampling result. Trained with z_dim = 512, epochs = 100.</figcaption>
  </figure>

## Part II: Denoising Diffusion Probabilistic Models (DDPM)
A variational autoencoder connects a latent variable to an observed image through a decoder. DDPM instead introduces a sequence of latent states and generates an image through repeated denoising.

### 4. DDPM as a Hierarchical Latent-Variable Model

DDPM construction involves two probability models: a <span style="color:red"> prescribed </span> forward-noising process $$q$$, and a learned reverse-generation process $$p_\theta$$.

The distinction between **modeling choices** and **derived distributions** is important. <span style="color:red">We choose Gaussian transition families for the forward and learned reverse processes.</span> Other Gaussian distributions, including the clean-conditioned reverse posterior, follow mathematically from these choices. ([Ho et al., 2020][ddpm-paper])

- ###  4.1 Variables, distributions, and the connection to VAE

  Following the notation in Part I, let an image have $$D$$ scalar coordinates. DDPM introduces states

  $$
  x_0,x_1,\ldots,x_T,
  $$

  where $$x_0$$ is a clean image and the remaining states represent progressively corrupted versions of it under the forward process.

  | Symbol | Meaning |
  |---|---|
  | $$x_0$$ | The observed clean image during training, or the final output during generation. |
  | $$x_t\in\mathbb{R}^D$$ | An intermediate noisy image at timestep $$t$$, generally containing both signal and noise. |
  | $$z=x_{1:T}$$ | The complete latent variable: the collection $$(x_1,\ldots,x_T)$$. |
  | $$q$$ | The probability model defined by the forward-noising process. |
  | $$p_\theta$$ | The learned generative probability model. |
  | $$\theta$$ | Trainable parameters of the denoising network. |
  | $$\epsilon_\theta(x_t,t)$$ | The network's prediction of noise from a noisy image and its timestep. |

  All intermediate states have the same dimensionality as the observed image. Calling them *latent variables* does not mean that they must be low-dimensional or pure noise; it means that the dataset does not provide them as observations.

  The correspondence to VAE is:

  | Role | VAE | DDPM |
  |---|---|---|
  | Observed variable | $$x$$ | $$x_0$$ |
  | Complete latent variable | $$z$$ | $$z=x_{1:T}$$ |
  | Variational distribution | Learned encoder $$q_\phi(z\mid x)$$ | Fixed noising process $$q(x_{1:T}\mid x_0)$$ |
  | Prior over all latents | $$p(z)$$ | $$p_\theta(x_{1:T})$$ |
  | Observation likelihood | $$p_\theta(x\mid z)$$ | $$p_\theta(x_0\mid x_1)$$ |
  | Exact model posterior | $$p_\theta(z\mid x)$$ | $$p_\theta(x_{1:T}\mid x_0)$$ |

  - The prior over the complete latent trajectory 

    Since $$z=x_{1:T}$$, the notation $$p_\theta(z)$$ refers to the joint distribution of all latent states before conditioning on an observed image. To identify this distribution, separate the final observation-generating step from the reverse-process factorization:

    $$
    \begin{aligned}
    p_\theta(x_0,z)
    &=
    p(x_T)
    \left[
    \prod_{t=2}^{T}p_\theta(x_{t-1}\mid x_t)
    \right]
    p_\theta(x_0\mid x_1).
    \end{aligned}
    $$

    The factors inside the first part generate only the latent states: first $$x_T$$, then $$x_{T-1}$$, and eventually $$x_1$$. The final factor generates the observed image $$x_0$$. This gives the grouping

    $$
    \boxed{
    p_\theta(x_0,z)
    =
    \underbrace{
    p(x_T)\prod_{t=2}^{T}p_\theta(x_{t-1}\mid x_t)
    }_{p_\theta(z)}
    p_\theta(x_0\mid x_1).
    }
    $$

    This factorization is the DDPM counterpart of the VAE factorization $$p_\theta(x,z)=p(z)p_\theta(x\mid z)$$. ([Ho et al., 2020][ddpm-paper])

    **More formally, the latent-path prior is obtained by marginalizing out the observation.** For discrete images, sum over all possible image values $$u$$:

    $$
    \begin{aligned}
    p_\theta(z)
    &=
    \sum_u p_\theta(u,z)\\[4pt]
    &=
    p(x_T)
    \left[
    \prod_{t=2}^{T}p_\theta(x_{t-1}\mid x_t)
    \right]
    \underbrace{
    \sum_u p_\theta(u\mid x_1)
    }_{1}\\[4pt]
    &=
    p(x_T)\prod_{t=2}^{T}p_\theta(x_{t-1}\mid x_t).
    \end{aligned}
    $$

    For a continuous observation likelihood, the sum over $$u$$ is replaced by an integral, which also equals one. Thus, marginalizing the observation removes the normalized final decoder factor and leaves the joint distribution of the latent trajectory.

    <span style="color:red">The word *prior* means that this distribution is not conditioned on the observed image.</span> It does **not** require the distribution to be fixed, independent of $$\theta$$, or standard Gaussian. **In DDPM, only the starting prior $$p(x_T)=\mathcal{N}(0,I)$$ is prescribed to be standard Gaussian. The complete latent-path prior $$p_\theta(z)$$ also contains the learned transitions and therefore depends on $$\theta$$.**

  - Observation likelihood conditions only on the last latent state

    The complete observation likelihood is $$p_\theta(x_0\mid z)$$. Using the factorization above,

    $$
    \begin{aligned}
    p_\theta(x_0\mid z)
    &=
    \frac{p_\theta(x_0,z)}{p_\theta(z)}\\[4pt]
    &=
    \frac{
    p_\theta(z)p_\theta(x_0\mid x_1)
    }{
    p_\theta(z)
    }\\[4pt]
    &=
    p_\theta(x_0\mid x_1).
    \end{aligned}
    $$

    Therefore,

    $$
    \boxed{
    p_\theta(x_0\mid x_1,\ldots,x_T)
    =
    p_\theta(x_0\mid x_1).
    }
    $$

    <span style="color:red">This simplification follows from the reverse model's Markov structure, not from the Gaussian assumption.</span> Once $$x_1$$ is known, the earlier generative states provide no additional information for the final transition. Those states have not disappeared from the model: they still determine the distribution of $$x_1$$.

  - **The marginal likelihood and maximum likelihood estimation:** 

    Like a VAE, DDPM defines the probability of an observed image by integrating out its latent variables:

    $$
    \boxed{
    \begin{aligned}
    p_\theta(x_0)
    &=
    \int p_\theta(x_0,z)\,dz\\[4pt]
    &=
    \int p_\theta(x_0\mid z)p_\theta(z)\,dz\\[4pt]
    &=
    \int
    p_\theta(x_0\mid x_1)
    p(x_T)
    \left[
    \prod_{t=2}^{T}p_\theta(x_{t-1}\mid x_t)
    \right]
    \,dx_{1:T},
    \end{aligned}
    }
    $$

    where $$dx_{1:T}=dx_1\cdots dx_T$$.

    Although the final conditional likelihood depends only on $$x_1$$, the marginal likelihood averages over the entire latent trajectory that can lead to it. Accordingly, $$p_\theta(x_0\mid x_1)$$ and $$p_\theta(x_0)$$ are different quantities: the former is the observation likelihood given a latent state, while the latter is the model's likelihood of the observed image alone.

    For a dataset of $$N$$ independent clean images, the maximum likelihood estimation problem is

    $$
    \boxed{
    \hat\theta_{\mathrm{MLE}}
    \in
    \operatorname*{arg\,max}_{\theta}
    \frac1N\sum_{n=1}^{N}
    \log p_\theta\!\left(x_0^{(n)}\right).
    }
    $$

    This is the same marginal-likelihood principle used for VAE models, with a hierarchical latent trajectory replacing the single latent layer. The expression above states the exact likelihood-based objective; the practical DDPM training loss will be derived separately. ([Kingma and Welling, 2013][vae-paper])

    Finally, the exact model posterior is

    $$
    \boxed{
    p_\theta(z\mid x_0)
    =
    \frac{p_\theta(x_0,z)}{p_\theta(x_0)}
    =
    \frac{
    p_\theta(x_0\mid x_1)p_\theta(z)
    }{
    p_\theta(x_0)
    }.
    }
    $$

    This posterior is implied by the learned generator. It is distinct from the prescribed forward-noising distribution $$q(z\mid x_0)$$. The latter is exactly specified as a corruption process, but serves as an approximate inference distribution for the former.

- ### 4.2 The forward model and the original DDPM noise schedule

Choose a sequence of forward variances $$\beta_1,\ldots,\beta_T$$, and define

$$
\alpha_t=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s,
\qquad
\bar\alpha_0=1.
$$

The original DDPM experiments use $$T=1000$$, with variances increasing linearly from $$10^{-4}$$ to $$0.02$$. Images are scaled to $$[-1,1]$$. ([Ho et al., 2020, Section 4][ddpm-experiments])

Explicitly, the schedule is

$$
\boxed{
\begin{aligned}
\beta_t
&=
\beta_1+\frac{t-1}{T-1}(\beta_T-\beta_1)\\[4pt]
&=
10^{-4}+\frac{t-1}{999}(0.02-10^{-4}),
\qquad t=1,\ldots,1000.
\end{aligned}
}
$$

The linear progression is across **diffusion timesteps**, not training iterations. These values remain fixed throughout training.

The forward process <span style="color: red;"> is chosen to be </span> a Markov chain:

$$
\boxed{
q(x_{1:T}\mid x_0)
=
\prod_{t=1}^{T}q(x_t\mid x_{t-1}),
}
$$

with Gaussian transitions

$$
\boxed{
q(x_t\mid x_{t-1})
=
\mathcal{N}\!\left(
x_t;\sqrt{\alpha_t}x_{t-1},\beta_t I
\right).
}
$$

Equivalently, one step is sampled as

$$
x_t
=
\sqrt{\alpha_t}x_{t-1}
+
\sqrt{\beta_t}\eta_t,
\qquad
\eta_t\sim\mathcal{N}(0,I),
$$

where the forward noises are independent across steps and independent of the clean image.

Thus, $$\beta_t$$ is the **variance** of the newly added noise, while $$\sqrt{\beta_t}$$ is its standard deviation. Each step attenuates the previous state and adds Gaussian noise.

- ### 4.3 The reverse model and its variance convention

Generation starts from the prescribed top-level prior

$$
\boxed{
p(x_T)=\mathcal{N}(x_T;0,I)
}
$$

and proceeds through the reverse chain

$$
x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_0.
$$

The joint model is

$$
\boxed{
p_\theta(x_{0:T})
=
p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t).
}
$$

For intermediate steps, we choose

$$
\boxed{
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}\!\left(
x_{t-1};\mu_\theta(x_t,t),\sigma_t^2 I
\right),
\qquad t\geq2.
}
$$

<span style="color: red">**The Gaussian form of this learned reverse conditional is a modeling choice.**</span> It does not assert that the true reverse conditional $$q(x_{t-1}\mid x_t)$$ is exactly Gaussian. The model is a Markov chain with Gaussian conditional distributions; its full joint distribution need not be Gaussian.

The original DDPM considers two prescribed reverse-variance schedules:

$$
\sigma_t^2=\beta_t
\qquad\text{or}\qquad
\sigma_t^2=\tilde\beta_t.
$$

The derivation in this post adopts

$$
\boxed{
\sigma_t^2=\tilde\beta_t,
\qquad t\geq2,
}
$$

where $$\tilde\beta_t$$ is derived in Section 5.3. This is one of the fixed-variance choices investigated in the paper, rather than a unique mandatory DDPM setting. ([Ho et al., 2020, Section 3.2][ddpm-reverse])

"Fixed variance" means that the variance is prescribed rather than learned. It may differ between timesteps, but for a given timestep it does not depend on the image or the network parameters. Unlike the forward $$\beta_t$$ schedule, the resulting $$\tilde\beta_t$$ schedule is not generally linear.

The final observation likelihood $$p_\theta(x_0\mid x_1)$$ is handled separately. For discrete image pixels, the original DDPM uses a discretized Gaussian decoder. Its positive underlying variance must be specified separately rather than set equal to $$\tilde\beta_1=0$$. The reconstruction-loss section will explain this distinction. ([Ho et al., 2020, Section 3.3][ddpm-decoder])

Finally, distinguish the top-level prior from the prior over the complete latent path:

$$
\boxed{
p_\theta(z)
=
p_\theta(x_{1:T})
=
p(x_T)\prod_{t=2}^{T}p_\theta(x_{t-1}\mid x_t).
}
$$

Only $$p(x_T)$$ is prescribed to be standard Gaussian. The other latent distributions are determined by the learned reverse transitions.

### 5. Forward Diffusion and the Analytic Denoising Target

The forward process serves two purposes. It constructs noisy training inputs, and <span style="color: red">it provides an analytically computable distribution that can supervise each learned reverse transition.</span>

The first requires $$q(x_t\mid x_0)$$. <span style="color: red">The second requires $$q(x_{t-1}\mid x_t,x_0)$$.</span> These distributions answer different questions.

- ### 5.1 Construct a noisy image at an arbitrary timestep

Fix the clean image $$x_0$$. Let $$m_t$$ be the conditional mean, and let $$v_t$$ denote the scalar coefficient of the conditional covariance:

$$
m_t:=\mathbb{E}_q[x_t\mid x_0],
\qquad
\operatorname{Cov}_q(x_t\mid x_0)=v_t I.
$$

Thus, $$v_t$$ describes the **accumulated conditional variance at timestep $$t$$**, not just the variance of the noise introduced in that one step. The latter is $$\beta_t$$.

Starting from

$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\eta_t,
$$

the zero mean of $$\eta_t$$ gives

$$
m_t=\sqrt{\alpha_t}m_{t-1},
\qquad m_0=x_0.
$$

Repeated substitution therefore yields

$$
m_t=\sqrt{\bar\alpha_t}x_0.
$$

<span style="color:red">For the covariance, scaling a random vector by a scalar multiplies its covariance by the square of that scalar. Also, the fresh noise $$\eta_t$$ is independent of $$x_{t-1}$$ conditional on the fixed $$x_0$$, so the cross-covariance terms vanish.</span> ([can be add together for total variance](https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables)) Therefore,

$$
\begin{aligned}
\operatorname{Cov}_q(x_t\mid x_0)
&=
\alpha_t\operatorname{Cov}_q(x_{t-1}\mid x_0)
+
\beta_t\operatorname{Cov}(\eta_t)\\[4pt]
&=
\alpha_t v_{t-1}I+\beta_t I.
\end{aligned}
$$

Matching the scalar coefficients of $$I$$ gives the recursion

$$
\boxed{
v_t=\alpha_t v_{t-1}+\beta_t,
\qquad v_0=0.
}
$$

The initial variance is zero because $$x_0$$ is held fixed in this conditional calculation. The term $$\alpha_t v_{t-1}$$ is the retained variance from earlier steps; $$\beta_t$$ is the newly added variance.

The solution is

$$
\boxed{v_t=1-\bar\alpha_t.}
$$

To verify it, the initial condition satisfies $$v_0=0=1-\bar\alpha_0$$. If $$v_{t-1}=1-\bar\alpha_{t-1}$$, then

$$
\begin{aligned}
v_t
&=\alpha_t(1-\bar\alpha_{t-1})+\beta_t\\[4pt]
&=(\alpha_t+\beta_t)-\alpha_t\bar\alpha_{t-1}\\[4pt]
&=1-\bar\alpha_t.
\end{aligned}
$$

**The bar is essential:** $$v_{t-1}=1-\bar\alpha_{t-1}$$, whereas $$1-\alpha_{t-1}=\beta_{t-1}$$ describes only one step's added variance. For example,

$$
v_1=\beta_1,
\qquad
v_2=\alpha_2\beta_1+\beta_2
=1-\alpha_1\alpha_2.
$$

Conditioned on $$x_0$$, the state is an affine combination of independent Gaussian noises, so it is Gaussian. Its mean and covariance above therefore give

$$
\boxed{
q(x_t\mid x_0)
=
\mathcal{N}\!\left(
x_t;\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
\right).
}
$$

Consequently, a sample at timestep $$t$$ can be generated directly:

$$
\boxed{
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
}
\tag{1}
$$

This direct noising formula is part of the standard DDPM construction. ([Ho et al., 2020, Equation (4)][ddpm-paper])

The first term is the retained clean signal; the second is the accumulated noise. Here, $$\epsilon$$ denotes the normalized aggregate noise in this direct formula, not necessarily the noise from the last individual forward step.

Equation (1) allows training to select any timestep without simulating all earlier states. It samples the required marginal distribution; independently using the formula at several timesteps is not the same as sampling an entire Markov trajectory.

- ### 5.2 Why the terminal state approaches standard Gaussian noise

At the final timestep,

$$
x_T
=
\sqrt{\bar\alpha_T}x_0
+
\sqrt{1-\bar\alpha_T}\epsilon.
$$

When $$\bar\alpha_T$$ is small, the clean-signal contribution becomes small and the noise coefficient approaches one.

For the original 1000-step linear schedule, evaluating the product gives

$$
\bar\alpha_{1000}
=
\prod_{t=1}^{1000}(1-\beta_t)
\approx4.0358\times10^{-5}.
$$

Therefore,

$$
x_{1000}
\approx
0.006353\,x_0+0.999980\,\epsilon.
$$

This explains why the endpoint is close to standard Gaussian noise, while also showing that the clean contribution has not vanished exactly.

For every fixed $$x_0$$, the conditional distribution $$q(x_t\mid x_0)$$ is already Gaussian. The terminal approximation is more specific:

$$
q(x_T\mid x_0)\approx\mathcal{N}(0,I),
$$

with a mean close to zero and covariance close to $$I$$, nearly independent of the original image.

This does not imply that the marginal $$q(x_t)$$, obtained by averaging over different clean images, is Gaussian at every timestep. Such an average generally combines Gaussians with different means.

- ### 5.3 Derive a tractable reverse target by conditioning on the clean image

During training, $$x_0$$ is known. We can therefore ask:

> Given this clean image and its noisy version at timestep $$t$$, what previous noisy states are plausible?

The answer is the local posterior

$$
q(x_{t-1}\mid x_t,x_0).
$$

Conditional Bayes' rule and Gaussian square completion give

$$
\boxed{
q(x_{t-1}\mid x_t,x_0)
=
\mathcal{N}\!\left(
x_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_t I
\right),
\qquad t\geq2,
}
\tag{2}
$$

where

$$
\boxed{
\tilde\mu_t(x_t,x_0)
=
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}
{1-\bar\alpha_t}x_t
+
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}
{1-\bar\alpha_t}x_0
}
\tag{3}
$$

and

$$
\boxed{
\tilde\beta_t
=
\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.
}
\tag{4}
$$

These are the analytic posterior parameters from DDPM. The full derivation of Equations (2)-(4) is given in [Appendix A](#ddpm-appendix-a). ([Ho et al., 2020, Equations (6)-(7)][ddpm-paper])

<p style="color: red;">
The Gaussian form here is a <strong>derived consequence</strong> of the linear-Gaussian forward process. This is different from choosing a Gaussian family for the learned reverse model.
</p>

The parameters have a direct statistical meaning:

$$
\tilde\mu_t(x_t,x_0)
=
\mathbb{E}_q[x_{t-1}\mid x_t,x_0],
$$

$$
\tilde\beta_t I
=
\operatorname{Cov}_q(x_{t-1}\mid x_t,x_0).
$$

Thus, Equation (3) describes the **mean of possible previous states**, not a deterministic identity for $$x_{t-1}$$. It combines information about the current noisy state with information about its clean origin.

The variance in Equation (4) depends only on the forward schedule, not on $$x_t$$ or $$x_0$$. This is why it can also be used as a fixed reverse-model variance during generation, when the clean image is unavailable.

Under the convention adopted in Section 4.3,

$$
\boxed{
\sigma_t^2=\tilde\beta_t,
\qquad
\sigma_t
=
\sqrt{
\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
},
\quad t\geq2.
}
$$

The square root is the standard deviation multiplying fresh noise in reverse sampling. The authors' implementation precomputes such variance schedules and retrieves their values using the timestep. ([DDPM implementation][ddpm-code])

- ### 5.4 Why this posterior is useful for training

The target distribution and learned distribution play different roles:

| | Analytic target | Learned reverse model |
|---|---|---|
| Distribution | $$q(x_{t-1}\mid x_t,x_0)$$ | $$p_\theta(x_{t-1}\mid x_t)$$ |
| Mean | $$\tilde\mu_t(x_t,x_0)$$ | $$\mu_\theta(x_t,t)$$ |
| Variance | Derived $$\tilde\beta_t I$$ | Prescribed $$\sigma_t^2 I$$ |
| Uses the true clean image? | Yes, to construct the training target. | No. |

The likelihood-bound derivation will produce local comparisons of the form

$$
D_{\mathrm{KL}}\!\left(
q(x_{t-1}\mid x_t,x_0)
\,\Vert\,
p_\theta(x_{t-1}\mid x_t)
\right).
$$

Equations (3) and (4) make the first distribution explicitly available. Because both distributions are Gaussian, this comparison can be evaluated analytically and converted into a prediction loss.

Conditioning the target on $$x_0$$ does not add $$x_0$$ to the network input. The clean image provides supervision during training; the learned reverse transition must operate using only the current noisy image and timestep.

<!-- Place Appendix A after the remaining main DDPM sections in the full post. -->

<a id="ddpm-appendix-a"></a>

## Appendix A. Conditional Bayes' Rule and the Gaussian Reverse Posterior

This appendix derives the target distribution in Section 5.3. The calculation uses only the specified forward process; no neural-network approximation is involved.

### A.1 Identify the unknown variable

Fix an intermediate timestep $$t\geq2$$, and write

$$
y=x_{t-1}.
$$

In the following calculation, $$x_0$$ and $$x_t$$ are known and held fixed. The variable $$y\in\mathbb{R}^D$$ is the unknown whose conditional density we seek.

The forward process gives

$$
q(y\mid x_0)
=
\mathcal{N}\!\left(
y;\sqrt{\bar\alpha_{t-1}}x_0,
(1-\bar\alpha_{t-1})I
\right)
$$

and

$$
q(x_t\mid y)
=
\mathcal{N}\!\left(
x_t;\sqrt{\alpha_t}y,\beta_t I
\right).
$$

The first density describes plausible previous states given the clean origin. The second evaluates how compatible each candidate previous state is with the observed next state.

### A.2 Derive conditional Bayes' rule

The conditional joint density $$q(y,x_t\mid x_0)$$ can be factored in two ways:

$$
q(y,x_t\mid x_0)
=
q(y\mid x_t,x_0)\,q(x_t\mid x_0)
$$

and

$$
q(y,x_t\mid x_0)
=
q(x_t\mid y,x_0)\,q(y\mid x_0).
$$

Equating the two expressions and dividing by $$q(x_t\mid x_0)$$ gives

$$
q(y\mid x_t,x_0)
=
\frac{
q(x_t\mid y,x_0)\,q(y\mid x_0)
}{
q(x_t\mid x_0)
}.
$$

This is conditional Bayes' rule: ordinary Bayes' rule with the additional background information $$x_0$$ retained throughout. No independence assumption was used to obtain this identity.

The forward Markov property gives

$$
q(x_t\mid y,x_0)=q(x_t\mid y),
$$

so

$$
\boxed{
q(y\mid x_t,x_0)
=
\frac{
q(x_t\mid y)\,q(y\mid x_0)
}{
q(x_t\mid x_0)
}.
}
\tag{A.1}
$$

This Markov simplification does not remove $$x_0$$ from the posterior on the left-hand side. Knowing the clean origin still provides information about the missing previous state.

Because $$x_t,x_0$$ are fixed, the denominator is constant with respect to $$y$$. Therefore,

$$
\boxed{
q(y\mid x_t,x_0)
\propto
q(x_t\mid y)\,q(y\mid x_0).
}
\tag{A.2}
$$

The symbol $$\propto$$ means equality up to a multiplicative factor independent of $$y$$, not approximate equality.

The denominator supplies the normalization:

$$
q(x_t\mid x_0)
=
\int q(x_t\mid y)q(y\mid x_0)\,dy.
$$

Dividing by this quantity ensures that the posterior integrates to one.

### A.3 Substitute the Gaussian densities

An isotropic Gaussian density in $$D$$ dimensions has the form

$$
\mathcal{N}(r;m,vI)
=
\frac{1}{(2\pi v)^{D/2}}
\exp\!\left[
-\frac{\lVert r-m\rVert^2}{2v}
\right].
$$

Here, $$v$$ is the variance coefficient, so the covariance matrix is $$vI$$.

Thus,

$$
q(x_t\mid y)
=
\frac{1}{(2\pi\beta_t)^{D/2}}
\exp\!\left[
-\frac{\lVert x_t-\sqrt{\alpha_t}y\rVert^2}{2\beta_t}
\right]
$$

and

$$
\begin{aligned}
q(y\mid x_0)
={}&
\frac{1}{[2\pi(1-\bar\alpha_{t-1})]^{D/2}}\\[4pt]
&\times
\exp\!\left[
-\frac{\lVert y-\sqrt{\bar\alpha_{t-1}}x_0\rVert^2}
{2(1-\bar\alpha_{t-1})}
\right].
\end{aligned}
$$

Substituting these expressions into Equation (A.1) produces a product of two exponentials. Using $$e^a e^b=e^{a+b}$$,

$$
\boxed{
\begin{aligned}
&q(y\mid x_t,x_0)\\[4pt]
&=K\exp\!\left[
-\frac12
\left(
\frac{\lVert x_t-\sqrt{\alpha_t}y\rVert^2}{\beta_t}
+
\frac{\lVert y-\sqrt{\bar\alpha_{t-1}}x_0\rVert^2}
{1-\bar\alpha_{t-1}}
\right)
\right],
\end{aligned}
}
\tag{A.3}
$$

where

$$
K=
\frac{
1
}{
q(x_t\mid x_0)
(2\pi\beta_t)^{D/2}
[2\pi(1-\bar\alpha_{t-1})]^{D/2}
}
$$

does not depend on $$y$$.

This explains the exponential expression: it comes directly from multiplying the Gaussian densities in Bayes' rule. Writing $$\propto$$ instead of equality allows us to suppress the multiplicative constant $$K$$ while retaining the density's dependence on $$y$$.

### A.4 Collect the quadratic terms

Expand the first squared norm:

$$
\lVert x_t-\sqrt{\alpha_t}y\rVert^2
=
\lVert x_t\rVert^2
-2\sqrt{\alpha_t}\,x_t^\top y
+\alpha_t\lVert y\rVert^2.
$$

Expand the second:

$$
\begin{aligned}
\lVert y-\sqrt{\bar\alpha_{t-1}}x_0\rVert^2
={}&
\lVert y\rVert^2
-2\sqrt{\bar\alpha_{t-1}}\,x_0^\top y\\[4pt]
&+
\bar\alpha_{t-1}\lVert x_0\rVert^2.
\end{aligned}
$$

After division by the respective variances and collection of like terms,

$$
\begin{aligned}
&
\frac{\lVert x_t-\sqrt{\alpha_t}y\rVert^2}{\beta_t}
+
\frac{\lVert y-\sqrt{\bar\alpha_{t-1}}x_0\rVert^2}
{1-\bar\alpha_{t-1}}\\[4pt]
&\qquad=
A\lVert y\rVert^2-2h^\top y+C,
\end{aligned}
$$

where

$$
A=
\frac{\alpha_t}{\beta_t}
+
\frac{1}{1-\bar\alpha_{t-1}},
$$

$$
h=
\frac{\sqrt{\alpha_t}}{\beta_t}x_t
+
\frac{\sqrt{\bar\alpha_{t-1}}}
{1-\bar\alpha_{t-1}}x_0,
$$

and

$$
C=
\frac{\lVert x_t\rVert^2}{\beta_t}
+
\frac{\bar\alpha_{t-1}\lVert x_0\rVert^2}
{1-\bar\alpha_{t-1}}.
$$

The quantity $$A$$ is a positive scalar, $$h$$ is a vector, and $$C$$ is independent of $$y$$. Hence,

$$
q(y\mid x_t,x_0)
\propto
\exp\!\left[
-\frac12\left(A\lVert y\rVert^2-2h^\top y+C\right)
\right].
$$

### A.5 Complete the square and identify the Gaussian

A Gaussian density is recognizable from an exponent of the form

$$
-\frac{1}{2v}\lVert y-m\rVert^2.
$$

Completing the square rewrites our quadratic in precisely this form.

Expand

$$
A\left\lVert y-\frac hA\right\rVert^2
=
A\lVert y\rVert^2-2h^\top y+\frac{\lVert h\rVert^2}{A}.
$$

Rearranging,

$$
\boxed{
A\lVert y\rVert^2-2h^\top y+C
=
A\left\lVert y-\frac hA\right\rVert^2
+
C-\frac{\lVert h\rVert^2}{A}.
}
$$

Therefore,

$$
\begin{aligned}
q(y\mid x_t,x_0)
\propto{}&
\underbrace{
\exp\!\left[
-\frac12\left(C-\frac{\lVert h\rVert^2}{A}\right)
\right]
}_{\text{independent of }y}\\[4pt]
&\times
\exp\!\left[
-\frac A2\left\lVert y-\frac hA\right\rVert^2
\right].
\end{aligned}
$$

The first factor can be absorbed into the normalization constant:

$$
q(y\mid x_t,x_0)
\propto
\exp\!\left[
-\frac A2\left\lVert y-\frac hA\right\rVert^2
\right].
$$

Comparing the center of the square and its coefficient with the standard Gaussian form gives

$$
\boxed{
\tilde\mu_t=\frac hA,
\qquad
\tilde\beta_t=\frac1A.
}
$$

In particular, $$1/A$$ is the variance coefficient; the standard deviation in each coordinate is $$1/\sqrt A$$.

Restoring normalization,

$$
q(y\mid x_t,x_0)
=
\left(\frac{A}{2\pi}\right)^{D/2}
\exp\!\left[
-\frac A2\left\lVert y-\frac hA\right\rVert^2
\right].
$$

Thus, the posterior is exactly Gaussian under the chosen forward model. Completing the square has not changed or approximated the density. It has exposed its center and covariance.

### A.6 Recover the final mean and variance

First simplify $$A$$:

$$
\begin{aligned}
A
&=
\frac{
\alpha_t(1-\bar\alpha_{t-1})+\beta_t
}{
\beta_t(1-\bar\alpha_{t-1})
}\\[4pt]
&=
\frac{
\alpha_t+\beta_t-\alpha_t\bar\alpha_{t-1}
}{
\beta_t(1-\bar\alpha_{t-1})
}\\[4pt]
&=
\frac{1-\bar\alpha_t}
{\beta_t(1-\bar\alpha_{t-1})}.
\end{aligned}
$$

The last equality uses

$$
\alpha_t+\beta_t=1,
\qquad
\bar\alpha_t=\alpha_t\bar\alpha_{t-1}.
$$

Taking the reciprocal gives

$$
\boxed{
\tilde\beta_t
=
\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t,
}
$$

which is Equation (4).

Next,

$$
\begin{aligned}
\tilde\mu_t
=
\frac hA
={}&
\left(
\frac{\sqrt{\alpha_t}}{\beta_t}x_t
+
\frac{\sqrt{\bar\alpha_{t-1}}}
{1-\bar\alpha_{t-1}}x_0
\right)\\[4pt]
&\times
\frac{\beta_t(1-\bar\alpha_{t-1})}
{1-\bar\alpha_t}.
\end{aligned}
$$

Distributing the factor gives

$$
\boxed{
\tilde\mu_t(x_t,x_0)
=
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}
{1-\bar\alpha_t}x_t
+
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}
{1-\bar\alpha_t}x_0,
}
$$

which is Equation (3).

The restriction $$t\geq2$$ avoids the degenerate case $$1-\bar\alpha_0=0$$. At $$t=1$$, conditioning on $$x_0$$ makes the corresponding distribution over $$x_0$$ a point mass; the final reconstruction likelihood is therefore treated separately.

**The calculation derives a posterior inside the known forward process. Choosing the learned reverse model to be Gaussian, and choosing its variance to equal this posterior variance, are separate modeling decisions.**

[ddpm-paper]: https://arxiv.org/html/2006.11239v2
[ddpm-experiments]: https://arxiv.org/html/2006.11239v2#S4
[ddpm-reverse]: https://arxiv.org/html/2006.11239v2#S3.SS2
[ddpm-decoder]: https://arxiv.org/html/2006.11239v2#S3.SS3
[vae-paper]: https://arxiv.org/abs/1312.6114
[ddpm-code]: https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py

### DDPM Results

## Appendix

### 4. 

## References
 1. <i class="fab fa-youtube"></i> [Variational Autoencoder - Model, ELBO, loss function and maths explained easily!](https://www.youtube.com/watch?v=iwEzwTTalbg) 
 2. <i class="fab fa-youtube"></i> [Understanding Variational Autoencoders (VAEs)](https://www.youtube.com/watch?v=HBYQvKlaE0A), [Variational Autoencoders \| Generative AI Animated](https://www.youtube.com/watch?v=qJeaCHQ1k2w&t=225s)
 3. <i class="fab fa-youtube"></i> [The Breakthrough Behind Modern AI Image Generators - Diffusion Models Part 1](https://www.youtube.com/watch?v=1pgiu--4W3I&t=1s) 
 4. <i class="fa-solid fa-book-open"></i> [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (VAE Paper)
 5. <i class="fa-solid fa-book-open"></i> [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239) (DDPM Paper), [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (DDIM Paper) 
 6. <i class="fab fa-youtube"></i> [Diffusion Model Paper Explanation](https://www.youtube.com/watch?v=HoKDTa5jHvg), [PyTorch Implementation Walk Through](https://www.youtube.com/watch?v=TBCRlnwJtZU&t=874s) and corresponding <i class="fa-brands fa-github"></i> [github repo](https://github.com/dome272/Diffusion-Models-pytorch)
 7. <i class="fa-brands fa-github"></i> [diffusion-DDPM-pytorch](https://github.com/Alokia/diffusion-DDPM-pytorch) & [diffusion-DDIM-pytorch](https://github.com/Alokia/diffusion-DDIM-pytorch)
 8. <i class="fab fa-youtube"></i> [An Optimal Control Perspective on Diffusion-Based Generative Modeling](https://www.youtube.com/watch?v=wQpQg1xIlBA&list=LL&index=3&t=2299s) & [SDE/ODE Interpretation of Diffusion Model](https://www.youtube.com/watch?v=Ro4v4z8YAsk&list=LL&index=3)
