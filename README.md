**Brownian Motion.py** contains a set of functions used for figures related to Brownian motion.
---

**GBM Calibration on NVDA.py** follows the logic I explain below:
---

Given that the stochastic process of the price of a stock \( S(t) \) follows a Geometric Brownian Motion (GBM), we have:

$$
\ln S(t) - \ln S(0) = \int_{0}^{t} M \, d\tau + \int_{0}^{t} \Sigma\, dW(\tau)
$$

where $t \in [0, T]$ and $M, \Sigma$ are constants. Therefore, the solution to the above equation is (following Itô's lemma):

$$
S(t) = S(0) \exp \{ M  t + \Sigma  dW(t) \}
$$

We will now compare this model with the historical data of NVDA stock from April 10, 2023, to April 10, 2024. The parameters $M$ and $\Sigma$ will be estimated using a logarithmic transformation on the entire dataset (from April 9, 2002, to April 10, 2024), which is available on [Yahoo! Finance](https://finance.yahoo.com/quote/NVDA/history).

First, we create a partition of the interval from April 9, 2002, to April 10, 2024, consisting of all the trading days, denoted as $\mathcal{T}$. Starting from the first such day, we collect the set:

$$
\mathcal{L} = (\\ln R_{t_i})_{t_i \in \mathcal{T}}
$$

where

$$
R_{t_i} = \frac{X_i}{X_{i-1}}
$$

with $X_i$ being the stock price at time $t_i \in \mathcal{T}$. Thus, we can calculate the parameters of the model:

$$
M = \overline{\mathcal{L}} - \frac{1}{2} Var(\mathcal{L})
$$

and

$$
\Sigma = \sqrt{Var(\mathcal{L})}
$$

where $\overline{\mathcal{L}}$ is the sample mean and $Var(\mathcal{L})$ is the sample variance of $\mathcal{L}$.

The parameters $M$ and $\Sigma$ can either be calculated using the full dataset (from 2002 to 2024) or segments of it. For example, we could calculate them for each week or month within the period from April 10, 2023, to April 10, 2024. This method allows the model to have greater correlation with real data.

## Weekly Parameter Estimation

To achieve more reliable results, we divide the period from April 10, 2023, to April 10, 2024, into 14 equally sized intervals and calculate the parameters $M$ and $\Sigma$ for each interval. Finally, using the Euler discretization, we arrive at the following two plots. The first plot shows the NVDA stock price alongside 10 different model simulations, while the second plot outlines the range between the maximum and minimum simulated values.

To confirm what we observed intuitively, we will compute the average correlation between all the simulations and the actual data within the study period. Specifically:

$$
\mathbb{E}[Cor(S(\tau), Y(\tau))]
$$

where $Y(\tau)$ is the time series of actual NVDA stock prices, and $S(\tau)$ is the set of model simulations. We assume that the simulations are independent and identically distributed. Let M denote the number of simulations, and we represent the $i$-th simulated path as $S^i(\tau)$. The correlation of the $i$-th simulation with the stock price is given by:

$$
Cor(S^i(\tau), Y(\tau))
$$

with an aggregate average correlation of:

$$
Cot^M = \frac{1}{M} \sum_{i=1}^{M} Cor(S^i(\tau), Y(\tau))
$$

By the weak law of large numbers, $Cot^M$ converges in probability to $\mathbb{E}[Cor(S(\tau), Y(\tau))]$. That is, as $M \to \infty$, $Cot^M$ should tend to a constant value.

Taking $M \to \infty$, we arrive at the result (see plots generated with .py):

$$
\lim_{M \to \infty} Cot^M \approx 0.82
$$

It should be noted that all the above results apply to this specific stock and parameter estimation method, so they cannot be generalized. Furthermore, the model used is one of the simplest methods of simulating a stock, and although it can approximate actual prices fairly well (since it has a high correlation with real data), it is not an optimal forecasting tool for stock prices.
