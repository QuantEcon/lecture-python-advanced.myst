---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(mcmc)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Markov Chain Monte Carlo

```{index} single: Markov Chains; Monte Carlo
```

## Overview

Let $\theta \in \Theta \subseteq \mathbb R$ be an unknown parameter and let $y = (y_1, \ldots, y_n) \in \mathbb R^n$ be observed data.

Bayesian inference centers on the **posterior distribution**

```{math}
:label: mcmc_bayes

\pi(\theta) := p(\theta \mid y)
= \frac{p(y \mid \theta) \, p(\theta)}{p(y)}
```

where $p(y \mid \theta)$ is the likelihood, $p(\theta)$ is the prior, and

```{math}
:label: mcmc_marginal

p(y) = \int_\Theta p(y \mid \theta) \, p(\theta) \, d\theta
```

is the marginal likelihood.

Throughout, $\pi$ denotes the posterior density {eq}`mcmc_bayes`, taken with respect to Lebesgue measure on $\Theta$.

The vector $y$ is held fixed throughout the following discussion.

When the prior and likelihood are conjugate, the integral in {eq}`mcmc_marginal` is available in closed form and the posterior is known analytically.

Outside the conjugate family, however, $p(y)$ is typically intractable.

We can nonetheless evaluate the **unnormalized posterior**

$$
\tilde p(\theta \mid y) := p(y \mid \theta) \, p(\theta)
= p(y) \, \pi(\theta)
$$

pointwise for any $\theta$, since that requires only an evaluation of the likelihood and the prior.

Metropolis-Hastings exploits exactly this: it constructs a Markov chain $(\theta_t)_{t \geq 0}$ whose stationary distribution is $\pi$ using only pointwise evaluations of $\tilde p(\theta \mid y)$.

In addition to having $\pi$ as the stationary distribution, the chain will also have the following ergodic property: for $\pi$-almost every choice of $\theta_0$,

$$
\frac{1}{T} \sum_{t=1}^{T} f(\theta_t)
\to \int_\Theta f(\theta) \, \pi(d\theta)
$$

with probability one as $T \to \infty$ for a large class of functions $f$.

This means that, by varying $f$, we can compute various features of the distribution $\pi$.

In the Bayesian setting, we use this convergence to estimate posterior means, variances, and quantiles from the sampled path: with $f(\theta) = \theta$ we recover the posterior mean, with $f = \mathbf 1_A$ the posterior probability of $A$, and so on.

The construction proceeds in two stages.

We first develop, in the abstract, the machinery of Markov transition kernels, detailed balance, and stationarity (see {ref}`mcmc_kernels`).

We then exhibit a specific kernel --- the Metropolis-Hastings kernel --- and verify that it has $\pi$ as its stationary distribution (see {ref}`mcmc_mh`).

Finally, we explain, informally, why the resulting chain actually delivers samples from $\pi$ (see {ref}`mcmc_mh_ergodicity`).

(mcmc_kernels)=
## Markov kernels

In this section we review some foundational aspects of Markov dynamics.

We met closely related ideas in [](stationary_densities), which studies Markov chains whose transition probabilities admit density representations; here we work with general kernels.

### Kernels and conditional distributions

Let $\Theta \subseteq \mathbb R$ be Borel, let $\mathcal B$ denote the Borel $\sigma$-algebra on $\Theta$, and let $\mu$ denote Lebesgue measure on $(\Theta, \mathcal B)$, which we use throughout as the reference measure.

The dynamics of a Markov chain are encoded in a transition kernel.

```{prf:definition} Markov kernel
:label: mcmc_def_kernel

A **Markov** (or **stochastic**) **kernel** on $(\Theta, \mathcal B)$ is a map $N \colon \Theta \times \mathcal B \to [0, 1]$ such that

1. for each fixed $\theta \in \Theta$, the map $A \mapsto N(\theta, A)$ is a probability measure on $(\Theta, \mathcal B)$; and
1. for each fixed $A \in \mathcal B$, the map $\theta \mapsto N(\theta, A)$ is $\mathcal B$-measurable.
```

We read $N(\theta, A)$ as the probability that the chain, currently at $\theta$, moves into the set $A$ at the next step:

$$
N(\theta, A) = \mathbb P \{ \theta_{t+1} \in A \mid \theta_t = \theta \}
$$

Many kernels are described not by their action on sets but by a **conditional density** --- a function of the next state given the current one.

````{prf:definition} Conditional density
:label: mcmc_def_density

A Markov kernel $N$ **admits a density representation** if there exists a conditional density $n$ such that, for every $\theta \in \Theta$ and every $A \in \mathcal B$,

```{math}
N(\theta, A) = \int_A n(\theta' \mid \theta) \, \mu(d\theta')
```
````

The statement that $n$ is a conditional density means that $n$ is measurable and nonnegative on $\Theta \times \Theta$ with $\int n(\theta' \mid \theta) \, \mu(d\theta') = 1$ for all $\theta \in \Theta$.

Not every kernel of interest admits a density representation.

Indeed, the Metropolis-Hastings kernel constructed below assigns positive probability to remaining at the current state, and so places an atom on the measure-zero diagonal $\{\theta' = \theta\}$.

````{prf:definition} Stationary distribution
:label: mcmc_def_stationary

A probability measure $\pi$ on $(\Theta, \mathcal B)$ is **stationary** for the kernel $N$ if

```{math}
:label: mcmc_stationary

(\pi N)(A) := \int_\Theta N(\theta, A) \, \pi(d\theta)
= \pi(A)
\qquad \text{for all } A \in \mathcal B
```
````

If $\theta_t \sim \pi$ and the chain evolves according to $N$, then {eq}`mcmc_stationary` says $\theta_{t+1} \sim \pi$ as well: once the chain reaches $\pi$, it stays in $\pi$.

Our goal is to construct a kernel for which the posterior {eq}`mcmc_bayes` is stationary.

### Detailed balance

The following sufficient condition for stationarity will be useful in what follows.

````{prf:definition} Detailed balance
:label: mcmc_def_db

A Markov kernel $N$ is said to be **reversible** with respect to a probability measure $\pi$, or to **satisfy detailed balance** with respect to $\pi$, if the bivariate measure

```{math}
:label: mcmc_joint

\Lambda(d\theta, d\theta') := \pi(d\theta) \, N(\theta, d\theta')
```

on $(\Theta \times \Theta, \, \mathcal B \otimes \mathcal B)$ is symmetric under interchange of its coordinates; that is, if

```{math}
:label: mcmc_db_measure

\int_{\Theta \times \Theta} g(\theta, \theta') \, \Lambda(d\theta, d\theta')
=
\int_{\Theta \times \Theta} g(\theta', \theta) \, \Lambda(d\theta, d\theta')
```

for every bounded measurable $g \colon \Theta \times \Theta \to \mathbb R$.
````

The measure $\Lambda$ is the joint law of a consecutive pair $(\theta_t, \theta_{t+1})$ when $\theta_t \sim \pi$.

Symmetry of $\Lambda$ says this pair is exchangeable: the chain looks the same run forwards as backwards in time.

When $N$ admits a density representation with conditional density $n(\cdot \mid \theta)$, and $\pi$ has density $\pi(\theta)$ with respect to $\mu$, condition {eq}`mcmc_db_measure` reduces to the familiar pointwise **detailed balance equation**

```{math}
:label: mcmc_db_density

\pi(\theta) \, n(\theta' \mid \theta)
=
\pi(\theta') \, n(\theta \mid \theta')
\qquad \text{for } \mu \otimes \mu \text{-almost every } (\theta, \theta')
```

Indeed, under domination both sides of {eq}`mcmc_db_measure` are integrals against the densities $\pi(\theta) \, n(\theta' \mid \theta)$ and $\pi(\theta') \, n(\theta \mid \theta')$ respectively, and these agree for all test functions $g$ whenever {eq}`mcmc_db_density` holds.

For us the significance of detailed balance lies in the following result.

```{prf:theorem} Detailed balance implies stationarity
:label: mcmc_thm_stat

If a Markov kernel $N$ satisfies detailed balance with respect to a probability measure $\pi$, then $\pi$ is stationary for $N$.
```

````{prf:proof}
Let $N$ satisfy detailed balance with respect to a probability measure $\pi$.

Fix $A \in \mathcal B$ and consider the bounded measurable test function $g(\theta, \theta') = \mathbf 1_A(\theta')$.

From the definitions we have

```{math}
:label: mcmc_stat_step1

(\pi N)(A)
= \int_\Theta \int_\Theta \mathbf 1_A(\theta') \, N(\theta, d\theta') \, \pi(d\theta)
= \int_{\Theta \times \Theta} \mathbf 1_A(\theta') \, \Lambda(d\theta, d\theta')
```

By detailed balance {eq}`mcmc_db_measure` we may swap the coordinates in the integrand, obtaining

```{math}
\int_{\Theta \times \Theta} \mathbf 1_A(\theta') \, \Lambda(d\theta, d\theta')
=
\int_{\Theta \times \Theta} \mathbf 1_A(\theta) \, \Lambda(d\theta, d\theta')
```

Now $\mathbf 1_A(\theta)$ does not depend on $\theta'$, so integrating out the second coordinate and using the fact that $N(\theta, \cdot)$ is a probability measure gives

```{math}
:label: mcmc_stat_step3

\int_{\Theta \times \Theta} \mathbf 1_A(\theta) \, \Lambda(d\theta, d\theta')
= \int_\Theta \mathbf 1_A(\theta)
  \int_\Theta N(\theta, d\theta')
  \, \pi(d\theta)
= \pi(A)
```

Chaining {eq}`mcmc_stat_step1`--{eq}`mcmc_stat_step3` yields $(\pi N)(A) = \pi(A)$.

As $A \in \mathcal B$ was arbitrary, this proves that $\pi$ is stationary for $N$.
````

(mcmc_ergo)=
### Ergodicity

We will use a classical ergodicity result described in this section.

Let $\pi$ be a probability measure on $\Theta$.

A kernel $N$ on $\Theta$ is called **$\pi$-irreducible** if, for every Borel set $A \subset \Theta$ with $\pi(A) > 0$ and every $\theta \in \Theta$, there is some $t \geq 1$ with $N^t(\theta, A) > 0$: every region of positive $\pi$-mass is eventually reachable from anywhere.

Here $N^t$ denotes the $t$-step kernel, defined inductively by $N^1 = N$ and

$$
N^{t+1}(\theta, A) = \int_\Theta N^t(\theta', A) \, N(\theta, d\theta')
$$

A kernel is called **periodic** if the state space splits into $d \geq 2$ disjoint classes that the chain visits in a fixed cyclic rotation, and **aperiodic** if no such split exists.

We can now state the following famous result.

````{prf:theorem} Ergodic theorem for Markov chains
:label: mcmc_thm_ergodic

Let $(\theta_t)_{t \geq 0}$ evolve under a kernel $N$ that is $\pi$-irreducible with stationary distribution $\pi$.

Then $\pi$ is the unique stationary distribution of $N$, and, for every $\pi$-integrable $f \colon \Theta \to \mathbb R$ and $\pi$-almost every initial $\theta_0$,

```{math}
:label: mcmc_ergodic

\frac{1}{T} \sum_{t=1}^{T} f(\theta_t)
\to
\mathbb E_\pi [ f(\theta) ]
= \int_\Theta f(\theta) \, \pi(d\theta)
\quad \text{as } T \to \infty
\qquad \text{almost surely}
```

If, in addition, $N$ is aperiodic, then, for $\pi$-almost every initial $\theta \in \Theta$,

```{math}
:label: mcmc_tvconv

\sup_{A \in \mathcal B} \,
\big| N^t(\theta, A) - \pi(A) \big|
\to 0
\quad \text{as } t \to \infty
```

that is, the distribution of $\theta_t$ converges to $\pi$ in total variation.
````

This theorem is an analogue of the classical law of large numbers, but for the dependent, non-IID draws produced by a Markov chain.

Note the division of labor between the hypotheses: irreducibility alone delivers the time-average convergence {eq}`mcmc_ergodic`, while aperiodicity supplies the distributional convergence {eq}`mcmc_tvconv`, which justifies treating $\theta_t$ as an approximate draw from $\pi$ when $t$ is large.

Proofs of the two parts can be found in chapters 17 and 13, respectively, of {cite}`MeynTweedie2009`.

(mcmc_mh)=
## The Metropolis-Hastings kernel

We now build a kernel that satisfies detailed balance with respect to the posterior $\pi$, which you will recall is equal to $p(\cdot \mid y)$.

This and {prf:ref}`mcmc_thm_stat` then imply that $\pi$ is stationary for the kernel, suggesting a means of sampling from $\pi$.

### The proposal

The chain moves in two stages: a candidate state is drawn from a **proposal kernel**, and is then accepted or rejected.

Throughout we impose the following assumption on the proposal.

````{prf:definition} Symmetric proposal
:label: mcmc_def_symm

A proposal kernel $q(\cdot \mid \theta)$, given by a conditional density, is **symmetric** if

```{math}
:label: mcmc_symm

q(\theta' \mid \theta) = q(\theta \mid \theta')
\qquad \text{for all } \theta, \theta' \in \Theta
```
````

```{prf:example} Gaussian random walk
:label: mcmc_eg_rw

Take $\Theta = \mathbb R$.

The canonical example satisfying {eq}`mcmc_symm` is the Gaussian random walk, where the update step is $\theta' = \theta + \varepsilon$, with $\varepsilon$ drawn, independently at each update, from a normal distribution with mean zero and standard deviation $\sigma$.

Its density $q(\theta' \mid \theta) = \sigma^{-1} \phi((\theta' - \theta)/\sigma)$, with $\phi$ the standard normal density, depends on $\theta, \theta'$ only through $|\theta' - \theta|$ and so is symmetric.
```

### The acceptance rule

Given the current state $\theta$ and a candidate $\theta'$, the candidate is accepted with probability

```{math}
:label: mcmc_alpha

\alpha(\theta, \theta')
= \min\left(1, \,
  \frac{\pi(\theta')}{\pi(\theta)}
  \right)
= \min\left(1, \,
  \frac{\tilde p(\theta' \mid y)}{\tilde p(\theta \mid y)}
  \right)
```

The last equality holds because $\pi = \tilde p(\cdot \mid y) / p(y)$, so the constant $p(y)$ cancels from the ratio.

Thus $\alpha$ is computable from the likelihood and prior alone.

If $\pi(\theta) = 0$ (equivalently, $\tilde p(\theta \mid y) = 0$), the ratio in {eq}`mcmc_alpha` is not defined, and we adopt the convention $\alpha(\theta, \theta') := 1$.

### The algorithm

With $\alpha$ and $q$ in hand we can state the Metropolis-Hastings algorithm for sampling from the posterior.

The symmetric-proposal version we state, shown in {prf:ref}`mcmc_algo_mh`, corresponds to the original Metropolis algorithm.

```{prf:algorithm} Metropolis-Hastings
:label: mcmc_algo_mh

**Inputs** initial state $\theta_0 \in \Theta$ with $\pi(\theta_0) > 0$; proposal density $q$; sample size $T$

**Output** the draws $\{\theta_t\}_{t=1}^T$

For $t = 1, 2, \ldots, T$:

1. propose $\theta' \sim q(\cdot \mid \theta_{t-1})$
1. draw $u \sim \mathrm{Uniform}(0, 1)$
1. if $u \leq \alpha(\theta_{t-1}, \theta')$, set $\theta_t = \theta'$; otherwise set $\theta_t = \theta_{t-1}$
```

In practice, when testing $u \leq \alpha(\theta_{t-1}, \theta')$ in {prf:ref}`mcmc_algo_mh`, we usually compute the log acceptance ratio

$$
\log \alpha(\theta_{t-1}, \theta')
= \min\big(0, \,
  \log \tilde p(\theta' \mid y)
  - \log \tilde p(\theta_{t-1} \mid y)\big)
$$

and test $\log u \leq \log \alpha(\theta_{t-1}, \theta')$.

### The transition kernel

We now construct the Markov kernel induced by {prf:ref}`mcmc_algo_mh`.

A move to a new state $\theta' \neq \theta$ requires both that $\theta'$ is proposed, with density $q(\theta' \mid \theta)$, and that it is accepted, with probability $\alpha(\theta, \theta')$.

Since the proposal draw and the accept/reject draw are independent, the resulting density of landing near $\theta' \neq \theta$ is

```{math}
:label: mcmc_taumove

\tau(\theta' \mid \theta)
= q(\theta' \mid \theta) \, \alpha(\theta, \theta'),
\qquad \theta' \neq \theta
```

Alternatively, the chain stays at $\theta$, which happens whenever the proposed candidate is rejected.

(For a continuous proposal the event $\theta' = \theta$ has probability zero, so staying is due to rejection alone.)

The total holding probability is

```{math}
:label: mcmc_reject

r(\theta) = 1 - \int_\Theta q(\theta' \mid \theta) \,
                  \alpha(\theta, \theta') \, \mu(d\theta')
```

Collecting the two cases, the Metropolis-Hastings kernel is the mixed kernel

```{math}
:label: mcmc_fullkernel

P(\theta, d\theta')
= \underbrace{q(\theta' \mid \theta) \, \alpha(\theta, \theta') \,
  \mu(d\theta')}_{\text{accepted move}}
+ \underbrace{r(\theta) \, \delta_\theta(d\theta')}_{\text{rejection}}
```

with absolutely continuous part $\tau(\cdot \mid \theta)$.

By construction the accepted-move mass $\int_\Theta q \, \alpha \, d\mu$ and the holding mass $r(\theta)$ sum to one, so $P(\theta, \cdot)$ is a probability measure and $P$ is a genuine Markov kernel.

### Stationarity of the posterior

In this section we show that $\pi$ is stationary for the Metropolis-Hastings kernel $P$.

```{prf:theorem} Stationarity of Metropolis-Hastings
:label: mcmc_thm_mhstat

If the proposal density $q$ satisfies the symmetry condition {eq}`mcmc_symm`, then, for each given $y \in \mathbb R^n$, the posterior $\pi = p(\cdot \mid y)$ is stationary for the Metropolis-Hastings kernel {eq}`mcmc_fullkernel`.
```

````{prf:proof}
Let $P$ be the Metropolis-Hastings kernel, so that

```{math}
P(\theta, d\theta')
= \tau(\theta' \mid \theta) \, \mu(d\theta')
+ r(\theta) \, \delta_\theta(d\theta')
```

In view of {prf:ref}`mcmc_thm_stat`, it suffices to show that $P$ satisfies detailed balance {eq}`mcmc_db_measure` with respect to the posterior $\pi$.

To see that this is so, observe that the joint measure {eq}`mcmc_joint` splits as

```{math}
\Lambda(d\theta, d\theta')
= \underbrace{\pi(\theta) \, \tau(\theta' \mid \theta) \,
  \mu(d\theta) \, \mu(d\theta')}_{=: \, \Lambda_{\mathrm{ac}}}
+ \underbrace{\pi(\theta) \, r(\theta) \,
  \mu(d\theta) \, \delta_\theta(d\theta')}_{=: \, \Lambda_{\mathrm{diag}}}
```

We show each part is symmetric under the swap $(\theta, \theta') \mapsto (\theta', \theta)$.

**The diagonal part.**

$\Lambda_{\mathrm{diag}}$ is concentrated on the diagonal $D = \{(\theta, \theta') : \theta' = \theta\}$, which the swap $(\theta, \theta') \mapsto (\theta', \theta)$ maps onto itself.

For any bounded measurable $g$, the integrand satisfies $g(\theta, \theta') = g(\theta', \theta)$ everywhere on $D$, since $\theta = \theta'$ there.

As $\Lambda_{\mathrm{diag}}$ charges only $D$, it follows that $\int g \, d\Lambda_{\mathrm{diag}} = \int g(\theta', \theta) \, \Lambda_{\mathrm{diag}}(d\theta, d\theta')$.

Hence $\Lambda_{\mathrm{diag}}$ is symmetric.

**The absolutely continuous part.**

It suffices to show that the density of $\Lambda_{\mathrm{ac}}$ with respect to $\mu \otimes \mu$ is symmetric --- equivalently, that it obeys the detailed balance equation {eq}`mcmc_db_density`:

```{math}
:label: mcmc_db_to_show

\pi(\theta) \, \tau(\theta' \mid \theta)
=
\pi(\theta') \, \tau(\theta \mid \theta')
\qquad \text{for all } \theta \neq \theta'
```

Substituting {eq}`mcmc_taumove` and noting that, by the proposal symmetry {eq}`mcmc_symm`, the factors $q(\theta' \mid \theta)$ and $q(\theta \mid \theta')$ on the two sides of {eq}`mcmc_db_to_show` are equal, it suffices to establish

```{math}
:label: mcmc_db_reduced

\pi(\theta) \, \alpha(\theta, \theta')
=
\pi(\theta') \, \alpha(\theta', \theta)
```

We verify {eq}`mcmc_db_reduced` by showing both sides equal $\min(\pi(\theta), \pi(\theta'))$.

If $\pi(\theta) > 0$, then

```{math}
\pi(\theta) \, \alpha(\theta, \theta')
= \pi(\theta) \, \min\left(1, \frac{\pi(\theta')}{\pi(\theta)}\right)
= \min\big(\pi(\theta), \, \pi(\theta')\big)
```

while if $\pi(\theta) = 0$, then $\alpha(\theta, \theta') = 1$ by convention, so that $\pi(\theta) \, \alpha(\theta, \theta') = 0 = \min\big(\pi(\theta), \pi(\theta')\big)$ once more.

In either case $\pi(\theta) \, \alpha(\theta, \theta') = \min\big(\pi(\theta), \pi(\theta')\big)$, and the right-hand side is symmetric in $(\theta, \theta')$.

Applying the same identity with the roles of $\theta$ and $\theta'$ exchanged gives $\pi(\theta') \, \alpha(\theta', \theta) = \min\big(\pi(\theta'), \pi(\theta)\big)$, which is the same quantity.

This establishes {eq}`mcmc_db_reduced`, hence {eq}`mcmc_db_to_show`, hence the symmetry of $\Lambda_{\mathrm{ac}}$.

Both parts of $\Lambda$ are symmetric, so $\Lambda$ is symmetric and detailed balance {eq}`mcmc_db_measure` holds.

Hence $\pi$ is stationary, as claimed.
````

(mcmc_mh_ergodicity)=
## Ergodicity of the Metropolis-Hastings kernel

Stationarity guarantees that if $\theta_0 \sim \pi$ then $\theta_t \sim \pi$ for all $t$.

For Monte Carlo to be useful we need more: starting from an arbitrary $\theta_0$, the time average of a function along a single trajectory should converge to its expectation under $\pi$.

This is the ergodicity property discussed in {ref}`mcmc_ergo`.

Here we investigate ergodicity in the setting of the Metropolis-Hastings algorithm.

### Ergodicity of the chain

Here is the key result.

(The argument can be extended to more general situations, but the present hypotheses keep the discussion simple.)

```{prf:theorem} Ergodicity of Metropolis-Hastings
:label: mcmc_thm_mherg

Let $\Theta = \mathbb R$, let $q$ be the Gaussian random walk proposal of {prf:ref}`mcmc_eg_rw`, and suppose that $\pi(\theta) > 0$ for all $\theta \in \Theta$.

If $(\theta_t)_{t \geq 0}$ is generated by the Metropolis-Hastings algorithm, then the kernel $P$ is $\pi$-irreducible and aperiodic, and the conclusions of {prf:ref}`mcmc_thm_ergodic` hold.
```

Stationarity of $\pi$ is already supplied by {prf:ref}`mcmc_thm_mhstat`, so, in view of {prf:ref}`mcmc_thm_ergodic`, it remains only to show that $P$ is $\pi$-irreducible and aperiodic.

Adopting the hypotheses of {prf:ref}`mcmc_thm_mherg` throughout the rest of this section, we discuss informally why these two properties hold.

### Why the chain is $\pi$-irreducible

For the random walk sampler this holds in a single step.

Discarding the holding term in {eq}`mcmc_fullkernel`, which only adds mass, we have, for any $A \in \mathcal B$,

```{math}
:label: mcmc_irred

P(\theta, A)
\geq \int_A q(\theta' \mid \theta) \, \alpha(\theta, \theta') \,
     \mu(d\theta')
```

Now fix $\theta$ and a target set $A$ with $\pi(A) > 0$.

The set $A$ has positive Lebesgue measure because $\pi(A) > 0$.

On this set the integrand in {eq}`mcmc_irred` is strictly positive: the Gaussian density satisfies $q(\theta' \mid \theta) > 0$ everywhere, and $\alpha(\theta, \theta') = \min\big(1, \pi(\theta')/\pi(\theta)\big) > 0$ because $\pi(\theta') > 0$.

An integral of a strictly positive function over a set of positive measure is strictly positive, so $P(\theta, A) > 0$.

Hence every positive-mass set is reached with positive probability in one step, and the chain is $\pi$-irreducible.

### Why the chain is aperiodic

A clean sufficient condition for aperiodicity (defined in {ref}`mcmc_ergo`) is that the chain can *remain in place*: if $P(\theta, \{\theta\}) > 0$ on a set of positive $\pi$-measure, then, combined with irreducibility, the chain is aperiodic, because a state that can be revisited at two consecutive times cannot belong to a nontrivial cycle.

The Metropolis-Hastings kernel has exactly this feature through its rejection mechanism.

Fix $\theta \in \Theta$ and consider the set

$$
L(\theta) := \{\theta' \in \Theta : \pi(\theta') < \pi(\theta)\}
$$

of candidates that are accepted with probability strictly less than one: on $L(\theta)$ we have $\alpha(\theta, \theta') = \pi(\theta')/\pi(\theta) < 1$.

We claim that $L(\theta)$ has positive Lebesgue measure.

To see this, observe that $\pi(\theta') \geq \pi(\theta)$ for every $\theta'$ in the complement $L(\theta)^c$, so integrating $\pi$ over $L(\theta)^c$ yields

$$
1
= \int_\Theta \pi \, d\mu
\geq \int_{L(\theta)^c} \pi \, d\mu
\geq \pi(\theta) \, \mu\big(L(\theta)^c\big)
$$

whence $\mu(L(\theta)^c) \leq 1/\pi(\theta) < \infty$.

Thus $L(\theta)$, being the complement in $\mathbb R$ of a set of finite Lebesgue measure, has infinite --- in particular, positive --- Lebesgue measure.

Moreover, $\alpha(\theta, \theta') = 1$ on $L(\theta)^c$, since $\pi(\theta') \geq \pi(\theta)$ there.

Splitting the integral in {eq}`mcmc_reject` over $L(\theta)$ and $L(\theta)^c$ now gives

$$
\int_\Theta q \, \alpha \, d\mu
= \int_{L(\theta)} q \, \alpha \, d\mu + \int_{L(\theta)^c} q \, d\mu
< \int_{L(\theta)} q \, d\mu + \int_{L(\theta)^c} q \, d\mu
= 1
$$

where the strict inequality holds because $\alpha < 1$ and $q(\cdot \mid \theta) > 0$ on $L(\theta)$, a set of positive $\mu$-measure.

By {eq}`mcmc_reject` this means $r(\theta) > 0$, and hence, by {eq}`mcmc_fullkernel`, $P(\theta, \{\theta\}) \geq r(\theta) > 0$.

As $\theta$ was arbitrary, the chain can remain in place at *every* state, which rules out any nontrivial period.

### Summary

{prf:ref}`mcmc_thm_mhstat` provides the stationary distribution, and the two arguments above supply $\pi$-irreducibility and aperiodicity.

By {prf:ref}`mcmc_thm_ergodic`, time averages along the simulated path converge almost surely to posterior expectations, and the distribution of $\theta_t$ converges to the posterior in total variation.

This is exactly what justifies using the output $\{\theta_t\}_{t=1}^T$ as a sample from the posterior.
