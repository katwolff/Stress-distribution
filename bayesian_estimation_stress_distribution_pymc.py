# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:11:32 2023

@author: wolff_k1
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from datetime import date

today = str(date.today())

az.style.use("arviz-darkgrid")
rng = np.random.default_rng(1234)

# Data: 500 parts with 10 failures
nparts = 500
nfail = 10
# observed failures (only 1 trial at each stress level so only 0 or 1 failures)
y = np.concatenate((np.repeat(1,nfail), np.repeat(0,nparts - nfail)))
# bundle data into dataframe
data = pd.DataFrame({"y": y})


coords = {"parts": data.index.values}

with pm.Model(coords=coords) as stress_distribution_model:
    # priors, further re-parametrization necessary
    sigma_stress = pm.TruncatedNormal("sigma_stress", mu=1, sigma=2, lower=0.1, upper=10)
    q90_stress = pm.TruncatedNormal("q90_stress", mu=1, sigma=2, lower=0.1, upper=2)
    mu_stress = pm.Deterministic("mu_stress", np.log(q90_stress) - 1.64*sigma_stress)
    # lognormal strain distribution
    stress = pm.LogNormal("stress", mu = mu_stress, sigma = sigma_stress, dims="parts")
    # pfailure is given by invlogit
    x = -5 + 4 * stress
    pfailure = pm.Deterministic("pfailure", pm.math.invlogit(x), dims="parts")
    # likelihood
    failure = pm.Bernoulli("failure", p=pfailure, dims="parts")
    nfailure = pm.Deterministic("nfailure", pm.math.sum(failure))
    # observe only number of failures, not individual parts
    # pm.Potential and pm.DiracDelta do not accept "observed" parameter
    # therefore use normal distribution with small sigma (nfailure is actually discrete)
    pm.Normal("observed", mu = nfailure, sigma = 0.001, observed = nfail)
    
    
pm.model_to_graphviz(stress_distribution_model)
gv = pm.model_to_graphviz(stress_distribution_model)
gv.render(f"Figures/{today}_stress_distribution_model", format='jpg', cleanup=True)


# Prior distribution checks
    
with stress_distribution_model:
    idata2 = pm.sample_prior_predictive(2000)

fig, ax = plt.subplots(1, 3, figsize=(9, 3))

sigma_stress_prior = np.concatenate(idata2.prior.sigma_stress.values)
mu_stress_prior = np.concatenate(idata2.prior.mu_stress.values)
scale_stress_prior = np.exp(mu_stress_prior)
q90_stress_prior = np.concatenate(idata2.prior.q90_stress.values)

az.plot_kde(
    values = sigma_stress_prior,
    values2 = q90_stress_prior,
    ax = ax[0]
)
ax[0].set(xlabel="sigma_stress", ylabel="q90_stress", title="Prior distribution")
az.plot_kde(
    values = sigma_stress_prior,
    values2 = mu_stress_prior,
    ax = ax[1]
)
ax[1].set(xlabel="sigma_stress", ylabel="mu_stress", title="Prior distribution")
az.plot_kde(
    values = sigma_stress_prior,
    values2 = scale_stress_prior,
    ax = ax[2]
)
ax[2].set(xlabel="sigma_stress", ylabel="scale_stress", title="Prior distribution")
plt.show()
fig.savefig(f"Figures/{today}_prior_distribution_pymc_model_parameters.jpg")


idno=1
# Run model
with stress_distribution_model:
    nchains=4
    ndraws=2000
    ntune=2000
    tacc=0.8
    idata = pm.sample(draws = ndraws, tune=ntune, chains = nchains, 
                      target_accept=tacc)
    
az.plot_trace(idata, var_names=["q90_stress", "sigma_stress", "mu_stress"]);
plt.savefig(f"Figures/{today}_chains{nchains}_tune{ntune}_draws{ndraws}_no{idno}_traces.jpg")

fig, ax = plt.subplots(1, 3, figsize=(9, 3))

sigma_stress_posterior = np.concatenate(idata.posterior.sigma_stress.values)
mu_stress_posterior = np.concatenate(idata.posterior.mu_stress.values)
scale_stress_posterior = np.exp(mu_stress_posterior)
q90_stress_posterior = np.concatenate(idata.posterior.q90_stress.values)

az.plot_kde(
    values = sigma_stress_posterior,
    values2 = q90_stress_posterior,
    ax = ax[0]
)
ax[0].set(xlabel="sigma_stress", ylabel="q90_stress", title="Posterior distribution")
az.plot_kde(
    values = sigma_stress_posterior,
    values2 = mu_stress_posterior,
    ax = ax[1]
)
ax[1].set(xlabel="sigma_stress", ylabel="mu_stress", title="Posterior distribution")
az.plot_kde(
    values = sigma_stress_posterior,
    values2 = scale_stress_posterior,
    ax = ax[2]
)
ax[2].set(xlabel="sigma_stress", ylabel="scale_stress", title="Posterior distribution")
plt.show()
fig.savefig(f"Figures/{today}_posterior_distribution_chains{nchains}_tune{ntune}_draws{ndraws}_no{idno}_pymc_model_parameters.jpg")


# Sample from posterior model

# var_names are important! They define at which point of the model samples
# are drawn. If no var_names are specified only the last distribution gets
# sampled!
thinned_idata = idata.sel(draw=slice(None, None, 5))
with stress_distribution_model:
    thinned_idata.extend(pm.sample_posterior_predictive(thinned_idata,
                                            var_names=['stress', 'pfailure',
                                                       'nfailure', 'observed'],
                                            random_seed=5))

thinned_idata.posterior_predictive.nfailure[0].mean()

az.plot_dist(thinned_idata.posterior_predictive.nfailure, 
             hist_kwargs={'bins': np.arange(0,25)})
az.plot_dist(thinned_idata.posterior_predictive.observed)

#idata.to_netcdf(f"Figures/{today}_chains{nchains}_tune{ntune}_draws{ndraws}_no{idno}_idata.nc")

# diagnostics
#az.summary(idata)
az.plot_energy(idata)
plt.savefig(f"Figures/{today}_chains{nchains}_tune{ntune}_draws{ndraws}_no{idno}_energy_plot.jpg")
az.bfmi(idata)
az.rhat(idata)
#az.hdi(idata)
az.plot_autocorr(idata, var_names=['q90_stress', 'sigma_stress', 'mu_stress'] )
az.plot_ess(idata, var_names=['q90_stress', 'sigma_stress', 'mu_stress'], kind='evolution' )
az.plot_parallel(idata, var_names=['q90_stress', 'sigma_stress'])
#az.plot_bpv(thinned_idata, kind="p_value", var_names=['nfailure'])
az.plot_posterior(idata, var_names=['q90_stress', 'sigma_stress'])
#az.plot_ppc(thinned_idata)
az.plot_rank(idata, var_names=['q90_stress', 'sigma_stress'])





























