"""
Bayesian Hierarchical & Pooled Modeling
======================================

This module provides a full implementation of pooled and hierarchical Bayesian
models using Gibbs sampling and classical t-based inference. The code is written
for clarity, reproducibility, and ease of interpretation, following the tasks
typically required in Bayesian computational analysis.

Tasks implemented:
- Construct the mortality dataset in the required structure
- Fit a pooled Bayesian model using classical posterior intervals and simulations
- Implement a hierarchical Bayesian model with Gibbs sampling
- Generate predictive distributions for new hypothetical populations
- Provide optional plotting utilities for inspection of parameter uncertainty

All computations, sampling strategies, and statistical formulas correspond to
standard Bayesian modeling procedures. The implementation uses only the same
libraries that were present in the original notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats

# ---------------------------------------------------------------------------
# DATA CREATION
# ---------------------------------------------------------------------------

def create_default_dataframe():
    """Construct the dataset of annual mortality counts for each cause.

    Returns
    -------
    DataFrame with rows=causes, columns=years
    """
    data = {
        "Year": [2022, 2021, 2020, 2019, 2018],
        "Accidental injuries, transport": [1132, 1149, 1005, 1080, 1149],
        "Accidental injuries, other": [1179, 1255, 1169, 1125, 1151],
        "Malignant Neoplasms": [3641, 3615, 3573, 3577, 3684],
        "Diseases of heart": [4079, 4000, 3816, 3898, 4126],
        "Cerebrovascular diseases": [1630, 1602, 1629, 1596, 1622],
    }

    df = pd.DataFrame(data)
    df = df.set_index("Year").transpose()
    df = df.apply(pd.to_numeric)
    return df

# ---------------------------------------------------------------------------
# POOLED MODEL
# ---------------------------------------------------------------------------

class PooledModel(object):
    """Pooled Bayesian model treating all observations as one homogeneous group."""

    def __init__(self, df, sim_size=5000):
        self.df = df
        self.sim_size = sim_size
        self.posterior_interval = None
        self.simulation = None

    def pooled_array(self):
        """Flatten the dataset into a single vector for pooled inference."""
        return self.df.values.flatten()

    def model_data(self, data):
        """Compute posterior interval and simulation using a t-distribution.

        Parameters
        ----------
        data : 1-D numpy array of observations
        """
        n = len(data)
        y_bar = np.mean(data)
        s2 = np.var(data, ddof=1)
        stderr = np.sqrt(s2 / n)

        # Compute t-based 95% interval
        lower = stats.t.ppf(0.025, df=n - 1, loc=y_bar, scale=stderr)
        upper = stats.t.ppf(0.975, df=n - 1, loc=y_bar, scale=stderr)

        # Generate simulated posterior draws
        simulation = stats.t.rvs(df=n - 1, loc=y_bar, scale=stderr, size=self.sim_size)

        return (lower, upper), simulation

    def run(self):
        """Execute pooled Bayesian analysis on the dataset."""
        data = self.pooled_array()
        self.posterior_interval, self.simulation = self.model_data(data)
        print("Pooled model completed. Posterior interval:", self.posterior_interval)
        return self.posterior_interval, self.simulation

    def plot_simulation(self, bins=30, show=True):
        """Plot histogram of posterior simulation draws."""
        if self.simulation is None:
            raise RuntimeError("Simulation not computed. Run .run() first.")
        plt.figure()
        plt.hist(self.simulation, bins=bins)
        plt.title("Pooled Simulation Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        if show:
            plt.show()

# ---------------------------------------------------------------------------
# HIERARCHICAL MODEL WITH GIBBS SAMPLING
# ---------------------------------------------------------------------------

class HierarchicalModel(object):
    """Hierarchical Bayesian model with Gibbs sampling for group-level effects."""

    def __init__(self, df, iterations=5000):
        self.df = df
        self.iterations = iterations
        self.causes = list(self.df.index)

    def theta_update(self, mu, sigma2, tau2, y):
        """Update group-specific mean based on precision-weighted combination."""
        y_bar = np.mean(y)
        J = len(y)
        return float((tau2 * J * y_bar + sigma2 * mu) / (tau2 * J + sigma2))

    def sigma2_update(self, theta, y):
        """Update within-group variance using scaled inverse-chi-square logic."""
        J = len(y)
        rss = np.sum((y - theta) ** 2)
        draw = stats.chi2.rvs(df=J - 1)
        return float(rss / draw)

    def mu_update(self, theta_vec, tau2):
        """Update the global mean using Normal posterior sampling."""
        K = len(theta_vec)
        theta_bar = np.mean(theta_vec)
        return float(stats.norm.rvs(loc=theta_bar, scale=np.sqrt(tau2 / K)))

    def tau2_update(self, theta_vec, mu):
        """Update between-group variance using scaled inverse-chi-square sampling."""
        J = len(theta_vec)
        s = np.sum((theta_vec - mu) ** 2)
        draw = stats.chi2.rvs(df=J - 1)
        return float(s / draw)

    def gibbs_sampler(self):
        """Run Gibbs sampling for hierarchical Bayesian inference."""
        theta_samples = {cause: [] for cause in self.causes}
        mu_samples = []
        sigma2_samples = []
        tau2_samples = []

        mu = 1.0
        sigma2 = 1.0
        tau2 = 1.0

        print("Starting Gibbs sampler with", self.iterations, "iterations")

        for it in range(self.iterations):
            theta_vec = []
            for cause in self.causes:
                y = self.df.loc[cause].values
                theta = self.theta_update(mu, sigma2, tau2, y)
                theta_vec.append(theta)
                theta_samples[cause].append(theta)

            theta_vec = np.array(theta_vec)

            mu = self.mu_update(theta_vec, tau2)
            tau2 = self.tau2_update(theta_vec, mu)

            sigma2_vals = [self.sigma2_update(theta_samples[c][-1], self.df.loc[c].values) for c in self.causes]
            sigma2 = float(np.mean(sigma2_vals))

            mu_samples.append(mu)
            sigma2_samples.append(sigma2)
            tau2_samples.append(tau2)

            if (it + 1) % 1000 == 0:
                print("Completed iteration", it + 1)

        print("Gibbs sampler finished")
        return theta_samples, mu_samples, sigma2_samples, tau2_samples

    def predictive_for_new_population(self, theta_samples, n_draws=5000):
        """Generate predictive values for a new population based on theta traces."""
        concatenated = np.concatenate([np.array(v) for v in theta_samples.values()])
        idx = np.random.choice(len(concatenated), size=n_draws, replace=True)
        return concatenated[idx]

    def predictive_y_from_theta(self, theta_draws, sigma2_draws=None):
        """Generate predictive observations using sampled theta values."""
        if sigma2_draws is None:
            sigma2_mean = 1.0
        else:
            sigma2_mean = float(np.mean(sigma2_draws))

        return np.random.normal(loc=theta_draws, scale=np.sqrt(sigma2_mean))

# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def summarize_interval(interval):
    return "95% interval: (%.2f, %.2f)" % (interval[0], interval[1])


def main(plot=True):
    """Execute pooled and hierarchical Bayesian analyses from start to finish."""
    df = create_default_dataframe()

    pooled = PooledModel(df)
    pooled_interval, pooled_sim = pooled.run()
    if plot:
        pooled.plot_simulation()

    hierarchical = HierarchicalModel(df)
    theta_samples, mu_samples, sigma2_samples, tau2_samples = hierarchical.gibbs_sampler()

    predictive_theta = hierarchical.predictive_for_new_population(theta_samples)
    predictive_y = hierarchical.predictive_y_from_theta(predictive_theta, np.array(sigma2_samples))

    if plot:
        plt.figure(); plt.hist(predictive_theta, bins=30); plt.title("Predictive Theta"); plt.show()
        plt.figure(); plt.hist(predictive_y, bins=30); plt.title("Predictive Y"); plt.show()

    results = {
        'df': df,
        'pooled_interval': pooled_interval,
        'pooled_sim': pooled_sim,
        'theta_samples': theta_samples,
        'mu_samples': mu_samples,
        'sigma2_samples': sigma2_samples,
        'tau2_samples': tau2_samples,
        'predictive_theta': predictive_theta,
        'predictive_y': predictive_y,
    }

    print("Pipeline complete")
    return results


if __name__ == "__main__":
    _ = main(plot=False)
