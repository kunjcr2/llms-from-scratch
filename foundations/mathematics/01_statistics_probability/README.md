# 📊 Statistics & Probability for Machine Learning

> *"All models are wrong, but some are useful."* — George Box

Statistics is not about memorizing formulas. It's about **reasoning under uncertainty** — the ability to draw conclusions from incomplete information.

---

## Table of Contents

1. [Populations & Sampling](#1-populations--sampling)
2. [Descriptive Statistics — Mean, Median, Mode](#2-descriptive-statistics--mean-median-mode)
3. [Expected Value](#3-expected-value)
4. [Variance & Standard Deviation](#4-variance--standard-deviation)
5. [Covariance & Correlation](#5-covariance--correlation)
6. [Random Variables](#6-random-variables)
7. [Probability Distributions](#7-probability-distributions)
8. [The Normal Distribution & Central Limit Theorem](#8-the-normal-distribution--central-limit-theorem)
9. [Conditional Probability & Bayes' Theorem](#9-conditional-probability--bayes-theorem)
10. [Maximum Likelihood Estimation (MLE)](#10-maximum-likelihood-estimation-mle)
11. [Linear Regression — The Statistical View](#11-linear-regression--the-statistical-view)
12. [Logistic Regression — Probability as Output](#12-logistic-regression--probability-as-output)

---

## 1. Populations & Sampling

### Intuition

You want to know the average height of all humans on Earth (~8 billion people). You can't measure everyone. So you **sample** — you pick a representative subset and use it to estimate the truth.

### Formal Definitions

- **Population**: The complete set of all entities you care about. Described by a true distribution $P$.
- **Sample**: A subset of $N$ observations drawn from the population: $\{x_1, x_2, \ldots, x_N\}$.
- **Statistic**: Any function of a sample (e.g., the sample mean $\bar{x}$).
- **Parameter**: The true value in the population (e.g., population mean $\mu$). Usually unknown.

The key tension: your sample is finite, but the population is vast. How accurately does what you measured reflect the true underlying reality? This is the central question of statistics — and the reason we have standard errors, confidence intervals, and hypothesis tests.

---

## 2. Descriptive Statistics — Mean, Median, Mode

### Mean (Arithmetic Average)

$$\bar{x} = \frac{1}{N} \sum_{i=1}^{N} x_i$$

The mean **minimizes the sum of squared errors** — i.e., if you had to guess one value for all data points, the mean minimizes $\sum(x_i - \hat{x})^2$. It's the "least wrong" single guess when you penalize large errors heavily (via squaring).

### Median

The **middle value** when data is sorted. Minimizes the **sum of absolute errors**: $\sum |x_i - \hat{x}|$. More robust to outliers than the mean.

**Why more robust?** If one person in a salary dataset earns $100M, the mean gets dragged up dramatically. But the median is unaffected — it's the middle value regardless of how extreme the tails are. This is why median household income is reported, not mean.

### Mode

The most frequently occurring value. Useful when you need the single most common outcome. For continuous data, mode is defined via the peak of the probability density function.

---

## 3. Expected Value

### Intuition

The expected value $\mathbb{E}[X]$ is the long-run average if you repeated an experiment infinitely many times. Roll a fair die a million times and average the results — you'll get very close to 3.5. That's $\mathbb{E}[\text{die roll}] = (1+2+3+4+5+6)/6 = 3.5$. It weights each possible outcome by how often you'd expect to see it.

### Formula

For a **discrete** random variable:

$$\mathbb{E}[X] = \sum_{x} x \cdot P(X = x)$$

For a **continuous** random variable with density $p(x)$:

$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot p(x) \, dx$$

### Properties

$$\mathbb{E}[aX + b] = a\mathbb{E}[X] + b \quad \text{(linearity)}$$
$$\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y] \quad \text{(always true)}$$
$$\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y] \quad \text{(only if X, Y are independent)}$$

The last property is **not** true in general. If $X$ and $Y$ are correlated (knowing $X$ tells you something about $Y$), then $\mathbb{E}[XY] \neq \mathbb{E}[X]\mathbb{E}[Y]$ — the correlation creates an extra "interaction" term.

---

## 4. Variance & Standard Deviation

### Intuition

The mean tells you *where* your data is centered. Variance tells you *how spread out* it is. A dataset $\{100, 100, 100\}$ and a dataset $\{50, 100, 150\}$ have the same mean (100) but very different spreads.

### Population Variance

$$\sigma^2 = \mathbb{E}[(X - \mu)^2] = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$$

We square the deviations for two reasons: (1) to make them positive (deviations can be positive or negative, but spread is always non-negative), and (2) to penalize large deviations more heavily than small ones.

### Sample Variance (Bessel's Correction)

$$s^2 = \frac{1}{N-1} \sum_{i=1}^{N} (x_i - \bar{x})^2$$

We divide by $N-1$ instead of $N$ to get an **unbiased estimator** of the population variance.

**Why N-1?** Imagine sampling $N=1$ observation. You can't measure spread from a single point. With $N-1 = 0$, the formula correctly gives undefined/infinite variance — acknowledging you have zero information about spread. More formally: when you compute $\bar{x}$ from the same sample you're measuring variance from, you've already "used up" one degree of freedom. The sample mean is the best guess for the true mean, but it's pulled toward the data — so deviations from the sample mean will be systematically smaller than deviations from the true mean. Dividing by $N-1$ instead of $N$ corrects for this underestimation.

### Standard Deviation

$$\sigma = \sqrt{\sigma^2}$$

Takes the square root to return to the original units. If your data is in grams, variance is in $\text{grams}^2$ — uninterpretable. Standard deviation is in grams — interpretable. "The typical deviation from the mean is $\sigma$ grams."

---

## 5. Covariance & Correlation

### Covariance

Covariance measures how two variables **move together**:

$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$$

- $\text{Cov}(X, Y) > 0$: When $X$ increases, $Y$ tends to increase (both above their means at the same time)
- $\text{Cov}(X, Y) < 0$: When $X$ increases, $Y$ tends to decrease (one above, one below their means)
- $\text{Cov}(X, Y) = 0$: No linear relationship

**Problem with raw covariance**: It's in units of $[\text{unit of X}] \times [\text{unit of Y}]$, which makes it impossible to compare across different pairs of variables.

### Correlation (Pearson)

Normalizes covariance to $[-1, 1]$:

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

By dividing by the standard deviations, you remove the units and scale. $\rho = 1$ means perfect positive linear relationship. $\rho = -1$ means perfect negative. $\rho = 0$ means no *linear* relationship (but there could still be a non-linear one).

### The Covariance Matrix

For a dataset with $d$ features, the covariance matrix $\Sigma$ is $d \times d$:

$$\Sigma_{ij} = \text{Cov}(X_i, X_j)$$

$$\Sigma = \frac{1}{N-1} (\mathbf{X} - \bar{\mathbf{X}})^T (\mathbf{X} - \bar{\mathbf{X}})$$

**What the structure tells you**:
- **Diagonal entries** ($\Sigma_{ii}$): variance of each individual feature — how spread out that feature is on its own.
- **Off-diagonal entries** ($\Sigma_{ij}$, $i \neq j$): covariance between features $i$ and $j$ — whether those two features move together. A large off-diagonal means those two features are highly correlated (knowing one tells you a lot about the other).

When $\Sigma$ is a diagonal matrix (all off-diagonals are zero), all features are uncorrelated — they carry independent information. PCA finds a rotation of the data that makes the covariance matrix diagonal in the new coordinate system.

---

## 6. Random Variables

### Intuition

A **random variable** $X$ is a function that maps outcomes of a random experiment to numbers. It's not "random" in the sense of chaotic — it has a precise probability structure.

**Example**: Flip a coin. Define $X = 1$ if heads, $X = 0$ if tails. $X$ maps the random outcome "heads/tails" to a number. The randomness comes from the experiment — $X$ is just the translation into numbers.

### Types

- **Discrete**: Takes countable values. E.g., number of words in a sentence, class label. Described by a **Probability Mass Function (PMF)**: $P(X = x)$ — the exact probability that $X$ takes value $x$.
- **Continuous**: Takes any value in an interval. E.g., temperature, pixel intensity. Described by a **Probability Density Function (PDF)**: $p(x)$, where $P(a \leq X \leq b) = \int_a^b p(x) \, dx$.

For continuous variables, $P(X = x) = 0$ for any specific point — the probability is zero! But the probability of falling in any interval is positive. Think of it like: the probability of measuring exactly 1.7000000... metres tall is zero (infinitely many decimal places), but the probability of being between 1.69 and 1.71 metres is non-zero.

### The PDF Constraint

$$\int_{-\infty}^{\infty} p(x) \, dx = 1 \quad \text{(total probability = 1)}$$

---

## 7. Probability Distributions

### 7.1 Uniform Distribution

$$X \sim \text{Uniform}(a, b), \quad p(x) = \frac{1}{b-a} \text{ for } x \in [a,b]$$

Every value in $[a, b]$ is equally likely. The distribution is flat — a rectangle. Simple and unbiased starting point when you have no prior reason to prefer any value over another.

### 7.2 Bernoulli Distribution

A single coin flip with probability $p$ of heads:

$$P(X=1) = p, \quad P(X=0) = 1-p$$
$$\mathbb{E}[X] = p, \quad \text{Var}(X) = p(1-p)$$

The variance is maximized at $p = 0.5$ (maximum uncertainty — you have no idea which way it'll go) and drops to zero at $p = 0$ or $p = 1$ (certainty — it always goes one way). This elegant connection between probability and variance is central to information theory.

**Binary cross-entropy** is the natural loss function for Bernoulli outcomes. When your model predicts $\hat{p}$ for a label $y \in \{0, 1\}$, cross-entropy is the negative log-likelihood under a Bernoulli model: $-[y \log \hat{p} + (1-y)\log(1-\hat{p})]$. See Section 10 for why this derivation matters.

### 7.3 Binomial Distribution

$n$ independent Bernoulli trials:

$$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$$

$$\mathbb{E}[X] = np, \quad \text{Var}(X) = np(1-p)$$

$\binom{n}{k}$ is "n choose k" — the number of distinct ways $k$ heads can appear in $n$ flips. $p^k(1-p)^{n-k}$ is the probability of any specific sequence with $k$ heads. Multiply them together: total probability of getting exactly $k$ heads regardless of order.

### 7.4 Normal (Gaussian) Distribution

The most important distribution in statistics:

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

$$X \sim \mathcal{N}(\mu, \sigma^2)$$

The bell curve, peaked at $\mu$, with spread controlled by $\sigma$. The $(x-\mu)^2$ in the exponent means values far from the mean fall off exponentially fast.

**Why it's everywhere:**
- The Central Limit Theorem makes it the natural limit of many averaging processes (Section 8 explains this)
- It is the **maximum entropy distribution** given only mean and variance constraints — meaning it makes the *fewest assumptions possible* about the data while matching those two statistics. If all you know is the mean and variance of your measurement errors, the Normal is the "least informative" choice — the most honest.

### 7.5 Categorical Distribution

Generalization of Bernoulli to $K$ outcomes:

$$P(Y = k) = \pi_k, \quad \sum_{k=1}^K \pi_k = 1$$

The **softmax function** maps any real-valued vector $z$ to a valid categorical distribution:

$$\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

**Why softmax?** A neural network's last layer might output any real numbers — including negatives, which can't be probabilities. Softmax does two things: (1) exponentiates to make everything positive, and (2) divides by the sum so everything adds to 1. The exponential also makes the largest input *dominate* — the largest value gets probability much closer to 1 than the second largest. This is why it's called "soft" max: as the temperature $T \to 0$ in $\text{softmax}(z/T)$, it approaches a hard max (all probability on the largest value).

---

## 8. The Normal Distribution & Central Limit Theorem

### Why the Normal Distribution is Special

**Central Limit Theorem (CLT)**: If you take the average of $n$ independent, identically distributed (i.i.d.) random variables $X_1, \ldots, X_n$ with mean $\mu$ and variance $\sigma^2$, then as $n \to \infty$:

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{d} \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)$$

**Translation**: No matter what shape the original distribution is — uniform, exponential, Bernoulli, completely crazy — averages of many samples become normally distributed. This is profound, and here's why:

Imagine you measure many independent sources of random error (manufacturing imprecision, measurement instrument noise, environmental variation). Each error might follow a completely different distribution. But their *sum* (or average) follows a Normal distribution. This is why measurement errors in physical experiments are Gaussian — they come from many small independent sources, and the CLT guarantees their sum is bell-shaped. It's the reason the Normal distribution appears *everywhere in nature*.

The variance shrinks as $1/n$: averaging more samples gives you a tighter, more accurate estimate of the true mean. That factor $\sigma^2/n$ is the variance of your sample mean estimate.

### The Standard Normal

$$Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$$

This **z-score** tells you how many standard deviations a data point is from the mean. Standardizing allows you to compare across different scales — a z-score of 2.5 means "extremely unusual" regardless of whether you're measuring grams or light-years.

### 68-95-99.7 Rule

$$P(\mu - \sigma \leq X \leq \mu + \sigma) \approx 68\%$$
$$P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) \approx 95\%$$
$$P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) \approx 99.7\%$$

These numbers are worth memorizing. If a measurement is 3 standard deviations from the mean, there's only a 0.3% chance of that happening by random chance — it's almost certainly a signal, not noise.

---

## 9. Conditional Probability & Bayes' Theorem

### Conditional Probability

The probability of event $A$ given that $B$ has already occurred:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

**Intuition**: You're restricting your universe to cases where $B$ happened, then asking how often $A$ also happens in that restricted universe. The total probability of $B$ normalizes back to 1 within that restricted universe.

**Example**: What's the probability that someone has a disease, given their test was positive? You can't just use the raw probability of disease — you need to restrict to the world of "positive tests" and ask how often they have the disease in that world. This is what drives all of medical diagnostics and Bayesian reasoning.

### The Product Rule

$$P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

Both forms are true simultaneously. Setting them equal gives you Bayes' theorem directly.

### Bayes' Theorem

Derived directly from the product rule — one of the most important equations in all of science:

$$\boxed{P(H \mid D) = \frac{P(D \mid H) \cdot P(H)}{P(D)}}$$

| Term | Name | Meaning |
|------|------|---------|
| $P(H \mid D)$ | **Posterior** | Probability of hypothesis given data |
| $P(D \mid H)$ | **Likelihood** | Probability of data given hypothesis |
| $P(H)$ | **Prior** | Initial belief in the hypothesis |
| $P(D)$ | **Evidence** | Normalizing constant |

**Intuition**: Bayes' theorem is a machine for **updating beliefs**. You start with a prior belief $P(H)$ about some hypothesis. Then you see data $D$. Bayes tells you exactly how much to update your belief in light of that evidence. The likelihood $P(D \mid H)$ measures how well the hypothesis explains the data — a hypothesis that predicted $D$ very precisely gets a strong boost; one that barely predicted it gets almost none.

### Law of Total Probability

$$P(D) = \sum_h P(D \mid H=h) \cdot P(H=h)$$

$P(D)$ is the probability of the data under *all* possible hypotheses, weighted by their prior probabilities. It's what normalizes the posterior to sum to 1.

---

## 10. Maximum Likelihood Estimation (MLE)

### Intuition

Given observed data $\mathcal{D} = \{x_1, \ldots, x_N\}$ and a model with parameters $\theta$, MLE asks:

> **What values of $\theta$ make this data most probable?**

Put differently: you've already seen the data. Now you adjust your model parameters so that, if you had generated data from your model, you'd have been most likely to get exactly the data you already have.

### The Likelihood Function

$$\mathcal{L}(\theta) = P(\mathcal{D} \mid \theta) = \prod_{i=1}^{N} p(x_i \mid \theta)$$

(Assumes i.i.d. data — each observation is independent, so joint probability factors into a product.)

### Log-Likelihood (The Math Gets Nicer)

Because products are hard to optimize (numerically unstable — products of small probabilities underflow to zero), we take the log:

$$\ell(\theta) = \log \mathcal{L}(\theta) = \sum_{i=1}^{N} \log p(x_i \mid \theta)$$

Since $\log$ is monotone increasing, maximizing $\ell(\theta)$ is equivalent to maximizing $\mathcal{L}(\theta)$. Products become sums under log — much friendlier for calculus and numerics.

### MLE → Familiar Loss Functions

**Case 1: Gaussian likelihood → MSE Loss**

If we model $y_i = f_\theta(x_i) + \epsilon_i$ where $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$:

$$\ell(\theta) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^N (y_i - f_\theta(x_i))^2$$

Maximizing log-likelihood = **minimizing MSE**. Mean squared error is not arbitrary — it's what you get from assuming your noise is Gaussian. If your noise were actually Laplacian instead, MLE would give you mean absolute error (L1 loss).

**Case 2: Bernoulli likelihood → Binary Cross-Entropy**

If $y_i \in \{0, 1\}$ and $P(y_i=1 \mid x_i) = \hat{p}_i$:

$$\ell(\theta) = \sum_{i=1}^N \left[ y_i \log \hat{p}_i + (1-y_i) \log(1-\hat{p}_i) \right]$$

Minimizing negative log-likelihood = **minimizing binary cross-entropy**. The loss function for binary classification isn't chosen arbitrarily — it falls out from assuming each label is a Bernoulli draw.

**Case 3: Categorical likelihood → Cross-Entropy**

For $K$-class classification:

$$\ell(\theta) = \sum_{i=1}^N \sum_{k=1}^K y_{ik} \log \hat{p}_{ik}$$

This is the **cross-entropy loss** used in virtually all classification networks.

### The Fundamental Insight

Your choice of loss function is not arbitrary — it follows directly from your assumption about the probability distribution of the data. MLE gives you the principled derivation of why each loss function takes the form it does.

---

## 11. Linear Regression — The Statistical View

### The Model

$$y = \mathbf{w}^T \mathbf{x} + b + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

We model the output $y$ as a linear function of features $\mathbf{x}$, plus Gaussian noise. The noise term $\epsilon$ captures everything we didn't model — measurement error, missing features, inherent randomness.

### MLE Solution

Under Gaussian noise, MLE gives us the **Ordinary Least Squares** solution:

$$\hat{\mathbf{w}} = \underset{\mathbf{w}}{\arg\min} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$

The closed-form **Normal Equation** (when $N$ is small enough to invert):

$$\hat{\mathbf{w}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

**Intuition**: $\mathbf{X}^T \mathbf{y}$ computes how much each feature correlates with the target. $(\mathbf{X}^T \mathbf{X})^{-1}$ normalizes for the correlations between features themselves (otherwise, if two features are highly correlated, you'd double-count their contribution). The result is the weights that minimize squared prediction error.

For large data, we use gradient descent instead (inverting $\mathbf{X}^T\mathbf{X}$ costs $O(n^3)$ and is infeasible for thousands of features).

### The Bias-Variance Tradeoff

**Total Error = Bias² + Variance + Irreducible Noise**

$$\mathbb{E}[(\hat{f}(x) - y)^2] = \underbrace{(\mathbb{E}[\hat{f}(x)] - f(x))^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]}_{\text{Variance}} + \sigma^2$$

- **Bias**: How wrong is your model on average? A simple linear model trying to fit a complex non-linear relationship has high bias — it's systematically off.
- **Variance**: How much does your model change across different training sets? A very complex model memorizes the training data exactly — change the training data slightly, and the model changes dramatically. That's high variance.
- **Irreducible noise** $\sigma^2$: The noise in the data itself — things that no model can predict.

The tradeoff: increasing model complexity reduces bias (fits patterns better) but increases variance (becomes sensitive to training data). Regularization adds a bias (pulls weights toward zero) to reduce variance.

---

## 12. Logistic Regression — Probability as Output

### Why Not Linear Regression for Classification?

Linear regression outputs $(-\infty, +\infty)$. Probabilities must be in $[0, 1]$. We need a **link function** that squashes real numbers into $[0, 1]$.

### The Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$$

**Why this particular function?** When $z \to +\infty$, $e^{-z} \to 0$ so $\sigma(z) \to 1$. When $z \to -\infty$, $e^{-z} \to \infty$ so $\sigma(z) \to 0$. At $z = 0$, $\sigma(0) = 0.5$. It's smooth, monotone, and has the output in $(0, 1)$ — perfect for a probability.

The model becomes:

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

### The Log-Odds (Logit)

If $p = P(y=1)$, then:

$$\text{logit}(p) = \log\frac{p}{1-p} = \mathbf{w}^T \mathbf{x} + b$$

**Intuition for log-odds**: The odds of an event are $p/(1-p)$ — a fair coin has odds 1:1. If $p = 0.9$, odds are 9:1. Logistic regression models the *log-odds* (logit) as linear in the features. The logit is unconstrained ($-\infty$ to $+\infty$), so it's the natural thing to model linearly. Taking sigmoid inverts the logit back to a probability.

Each weight $w_j$ represents: a one-unit increase in $x_j$ multiplies the **odds** by $e^{w_j}$. This is called the "odds ratio" — a multiplicative effect on odds rather than an additive effect on probability.

### Loss Function

From MLE under Bernoulli likelihood, the **Binary Cross-Entropy Loss**:

$$\mathcal{L}(\mathbf{w}) = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

**What this penalizes**: If $y_i = 1$ (true label is positive) and you predict $\hat{p}_i = 0.01$ (very confident it's negative), the loss is $-\log(0.01) = 4.6$ — a huge penalty. If you predict $\hat{p}_i = 0.99$, loss is $-\log(0.99) \approx 0.01$ — nearly zero. The loss is designed so that being confidently wrong is catastrophically penalized.

### Decision Boundary

The model predicts class 1 when $P(y=1 \mid x) > 0.5$, which is equivalent to $\mathbf{w}^T\mathbf{x} + b > 0$. This is a **hyperplane** in feature space — a line in 2D, a plane in 3D, a flat surface in higher dimensions. Logistic regression is a *linear* classifier: the boundary between classes is always a straight line (or hyperplane).

Once you understand logistic regression well, more complex predictive models become natural extensions: add non-linear feature transformations (kernels), stack multiple logistic regression layers (neural networks), or apply it to multiple classes via softmax.

---

## 📓 Notebook

Open [`statistics_probability.ipynb`](./statistics_probability.ipynb) for hands-on code for every concept above.
