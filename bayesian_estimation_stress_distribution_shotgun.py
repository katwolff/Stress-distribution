# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:09:40 2023

@author: wolff_k1
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import date
from scipy.stats import lognorm, halfnorm, uniform
#import os
import time

#os.getcwd()

np.random.seed(286)
today = str(date.today())

#loc=0, s=0.1...3, scale=0.05...5 to produce "reasonably looking" f(stress)
rv_s_prior = halfnorm(loc=0.001, scale=1.5)
rv_scale_prior = halfnorm(loc=0.001, scale=2.5)

# "resolution" for parameter distribution
nsample_prior = 1000
# "precision" for likelihood
nsample_likelihood = 1000
prior_distribution = pd.DataFrame()
prior_distribution["s_stress"] = rv_s_prior.rvs(size=nsample_prior)
prior_distribution["scale_stress"] = rv_scale_prior.rvs(size=nsample_prior)
prior_distribution["mu_stress"] = np.log(prior_distribution.scale_stress)
prior_distribution["q90_stress"] = np.exp(prior_distribution.mu_stress +\
    1.64 * prior_distribution.s_stress)

fig, ax = plt.subplots(1, 3, figsize=(9, 3))
ax[2].hexbin(prior_distribution.s_stress, prior_distribution.scale_stress)
ax[2].set_title("Prior distribution")
ax[2].set_xlabel("s_stress")
ax[2].set_ylabel("scale_stress")
ax[1].hexbin(prior_distribution.s_stress, prior_distribution.mu_stress)
ax[1].set_title("Prior distribution")
ax[1].set_xlabel("s_stress")
ax[1].set_ylabel("mu_stress")
ax[0].hexbin(prior_distribution.s_stress, prior_distribution.q90_stress)
ax[0].set_title("Prior distribution")
ax[0].set_xlabel("s_stress")
ax[0].set_ylabel("q90_stress")
ax[0].set_ylim(0,1000)
plt.show()
fig.savefig(f"Figures/{today}_prior_distribution_nsample{nsample_prior}.jpg")


# observed 10 failures in 500 parts
nparts = 500
nfailures_obs = 10

posterior_list = []

start = time.time()
for index, row in prior_distribution.iterrows():
    print(index)
    #print(row["s_stress"], row["scale_stress"])
    likelihood = 0
    for i in range(nsample_likelihood):
        stress_sample = lognorm.rvs(s = row["s_stress"], scale = row["scale_stress"], 
                          size = nparts)
        rand_unif = uniform.rvs(size = nparts)
        x = -5 + 4 * stress_sample
        pfailure = 1 / (1 + np.exp(-x))
        nfailures = (rand_unif < pfailure).sum()
 #       print(broken_parts)
        if nfailures == nfailures_obs:
            likelihood = likelihood + 1
    #print(likelihood)
    if likelihood > 0:
        posterior_list = posterior_list + [[row["s_stress"], row["scale_stress"], 
                                            row["mu_stress"], row["q90_stress"],
                                           likelihood]]

end = time.time()
mins = (end-start)/60
print(f"Computing likelihood took {mins:.1f} minutess")


posterior_df = pd.DataFrame(posterior_list, columns=["s_stress", "scale_stress", 
                                                     "mu_stress", "q90_stress",
                                                     "likelihood"])
posterior_df.to_excel(f"{today}_posterior_distribution_nsample_prior{nsample_prior}_nsample_lik{nsample_likelihood}_nparts{nparts}.xlsx", 
                      index=False)
#posterior_df = pd.read_excel("2023-06-08_posterior_distribution.xlsx")

posterior_df2 = posterior_df.loc[posterior_df.index.repeat(posterior_df.likelihood)]

fig, ax = plt.subplots(1, 3, figsize=(9, 3))
ax[2].hexbin(posterior_df2.s_stress, posterior_df2.scale_stress)
ax[2].tricontour(posterior_df.s_stress, posterior_df.scale_stress, 
              posterior_df.likelihood)
ax[2].set_title("Posterior distribution")
ax[2].set_xlabel("s_stress")
ax[2].set_ylabel("scale_stress")
ax[1].hexbin(posterior_df2.s_stress, posterior_df2.mu_stress)
ax[1].tricontour(posterior_df.s_stress, posterior_df.mu_stress, 
              posterior_df.likelihood)
ax[1].set_title("Posterior distribution")
ax[1].set_xlabel("s_stress")
ax[1].set_ylabel("mu_stress")
ax[0].hexbin(posterior_df2.s_stress, posterior_df2.q90_stress)
ax[0].tricontour(posterior_df.s_stress, posterior_df.q90_stress, 
              posterior_df.likelihood)
ax[0].set_title("Posterior distribution")
ax[0].set_xlabel("s_stress")
ax[0].set_ylabel("q90_stress")
plt.show()
fig.savefig(f"Figures/{today}_posterior_distribution_nsample_prior{nsample_prior}_nsample_lik{nsample_likelihood}_nparts{nparts}.jpg")










