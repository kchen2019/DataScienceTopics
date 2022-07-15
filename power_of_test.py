## this code is for simulating coin toss experiment

# import packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

# set a random state so the output can be repeated
np.random.seed(0)

# ------------------------------------------------------------------
# Toss a fair coin
# ------------------------------------------------------------------

# function to toss a fair coin based on sample size and iteration
def coin_toss(sample_size, iteration):

    p_head = []
    for i in range(iteration):
        p_curr = sum(np.random.binomial(1, 0.5, sample_size)) / sample_size
        p_head.append(p_curr)

    return(p_head)

# toss coin 10 times, and repeat the experiment for many iterations
sample_size_N = 10
iteration_runs = [1, 5, 10, 100, 200, 300, 1000, 2000]
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8,4))
axes=axes.ravel()
for i, ax in enumerate(axes):
    print(i)
    iteration = iteration_runs[i]
    ax.hist(coin_toss(sample_size_N,iteration))
    ax.set_title('Repeat {} times'.format(iteration))
    ax.set_xlim([0, 1])
fig.tight_layout()
plt.show()

# toss coin 50 times, and repeat the experiment for many iterations
sample_size_N = 50
iteration_runs = [1, 5, 10, 100, 200, 300, 1000, 2000]
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8,4))
axes=axes.ravel()
for i, ax in enumerate(axes):
    print(i)
    iteration = iteration_runs[i]
    ax.hist(coin_toss(sample_size_N,iteration))
    ax.set_title('Repeat {} times'.format(iteration))
    ax.set_xlim([0, 1])
fig.tight_layout()
plt.show()

# toss coin 100 times, and repeat the experiment for many iterations
sample_size_N = 100
iteration_runs = [1, 5, 10, 100, 200, 300, 1000, 2000]
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8,4))
axes=axes.ravel()
for i, ax in enumerate(axes):
    print(i)
    iteration = iteration_runs[i]
    ax.hist(coin_toss(sample_size_N,iteration))
    ax.set_title('Repeat {} times'.format(iteration))
    ax.set_xlim([0, 1])
fig.tight_layout()
plt.show()

# ------------------------------------------------------------------
# Toss an unknown coin
# ------------------------------------------------------------------
# set sample size of 10
sample_size = 10
# plot the null hypothesis normal distribution
u = 0.5
sigma = np.sqrt(u*(1-u)/sample_size)
# randomly sample 1000 along x-axis from this distribution
x = np.sort(np.random.normal(u, sigma, 1000))
# compute likelihood of the observation along x-axis
l = [scipy.stats.norm(u, sigma).pdf(i) for i in x]
# plot x and l
plt.plot(x, l, color = 'red')
# this is observed p(Head)
plt.axvline(0.4, linewidth = 2)
# this is the vertical line at 0.05 significant level
plt.axvline(scipy.stats.norm(u, sigma).ppf(0.05), linewidth = 2, color = 'black')
# this is vertical line for P(Head) under H0
plt.axvline(0.5, linestyle = '--')
# bound the x-axis to [0, 1] since x-axis is probability
plt.xlim([0,1])
plt.show()


# set sample size of 100
sample_size = 100
# plot the null hypothesis normal distribution
u = 0.5
sigma = np.sqrt(u*(1-u)/sample_size)
# randomly sample 1000 along x-axis from this distribution
x = np.sort(np.random.normal(u, sigma, 1000))
# compute likelihood of the observation along x-axis
l = [scipy.stats.norm(u, sigma).pdf(i) for i in x]
# plot x and l
plt.plot(x, l, color = 'red')
# this is observed p(Head)
plt.axvline(0.4, linewidth = 2)
# this is the vertical line at 0.05 significant level
plt.axvline(scipy.stats.norm(u, sigma).ppf(0.05), linewidth = 2, color = 'black')
# this is vertical line for P(Head) under H0
plt.axvline(0.5, linestyle = '--')
# bound the x-axis to [0, 1] since x-axis is probability
plt.xlim([0,1])
plt.show()


# ------------------------------------------------------------------
# Factors that impact on power
# ------------------------------------------------------------------

# sample size
# sample size = 10
sample_size = 10
# plot the null hypothesis normal distribution
u0 = 0.5
sigma0 = np.sqrt(u0*(1-u0)/sample_size)
# randomly sample 1000 observations along x-axis from this distribution
x0 = np.sort(np.random.normal(u0, sigma0, 1000))
# compute likelihood of the x
l0 = [scipy.stats.norm(u0, sigma0).pdf(i) for i in x0]
# plot the alternative hypothesis normal distribution
u1 = 0.4
sigma1 = np.sqrt(u1*(1-u1)/sample_size)
# randomly sample 1000 observations along x-axis from this distribution
x1 = np.sort(np.random.normal(u1, sigma1, 1000))
# compute likelihood of the x1
l1 = [scipy.stats.norm(u1, sigma1).pdf(i) for i in x1]
# plot null hypothesis normal distribution
plt.plot(x0, l0, color = 'red')
# plot alternative hypothesis normal distribution
plt.plot(x1, l1, color = 'green')
# this is the vertical line at 0.05 significant level
plt.axvline(scipy.stats.norm(u0, sigma0).ppf(0.05), linewidth = 2, color = 'black')
# this is vertical line for P(Head) under H0
plt.axvline(0.5, linestyle = '--', color = 'red')
# this is vertical line for P(Head) under Ha
plt.axvline(0.4, linestyle = '--', color = 'green')
# bound the x-axis to [0, 1] since x-axis is probability
plt.xlim([0,1])
plt.show()

# sample size = 100
sample_size = 100
# plot the null hypothesis normal distribution
u0 = 0.5
sigma0 = np.sqrt(u0*(1-u0)/sample_size)
# randomly sample 1000 observations along x-axis from this distribution
x0 = np.sort(np.random.normal(u0, sigma0, 1000))
# compute likelihood of the x
l0 = [scipy.stats.norm(u0, sigma0).pdf(i) for i in x0]
# plot the alternative hypothesis normal distribution
u1 = 0.4
sigma1 = np.sqrt(u1*(1-u1)/sample_size)
# randomly sample 1000 observations along x-axis from this distribution
x1 = np.sort(np.random.normal(u1, sigma1, 1000))
# compute likelihood of the x1
l1 = [scipy.stats.norm(u1, sigma1).pdf(i) for i in x1]
# plot null hypothesis normal distribution
plt.plot(x0, l0, color = 'red')
# plot alternative hypothesis normal distribution
plt.plot(x1, l1, color = 'green')
# this is the vertical line at 0.05 significant level
plt.axvline(scipy.stats.norm(u0, sigma0).ppf(0.05), linewidth = 2, color = 'black')
# this is vertical line for P(Head) under H0
plt.axvline(u0, linestyle = '--', color = 'red')
# this is vertical line for P(Head) under Ha
plt.axvline(u1, linestyle = '--', color = 'green')
# bound the x-axis to [0, 1] since x-axis is probability
plt.xlim([0,1])
plt.show()


# effect size
# sample size = 10
sample_size = 10
# plot the null hypothesis normal distribution
u0 = 0.5
sigma0 = np.sqrt(u0*(1-u0)/sample_size)
# randomly sample 1000 observations along x-axis from this distribution
x0 = np.sort(np.random.normal(u0, sigma0, 1000))
# compute likelihood of the x
l0 = [scipy.stats.norm(u0, sigma0).pdf(i) for i in x0]
# plot the alternative hypothesis normal distribution
u1 = 0.1
sigma1 = np.sqrt(u1*(1-u1)/sample_size)
# randomly sample 1000 observations along x-axis from this distribution
x1 = np.sort(np.random.normal(u1, sigma1, 1000))
# compute likelihood of the x1
l1 = [scipy.stats.norm(u1, sigma1).pdf(i) for i in x1]
# plot null hypothesis normal distribution
plt.plot(x0, l0, color = 'red')
# plot alternative hypothesis normal distribution
plt.plot(x1, l1, color = 'green')
# this is the vertical line at 0.05 significant level
plt.axvline(scipy.stats.norm(u0, sigma0).ppf(0.05), linewidth = 2, color = 'black')
# this is vertical line for P(Head) under H0
plt.axvline(u0, linestyle = '--', color = 'red')
# this is vertical line for P(Head) under Ha
plt.axvline(u1, linestyle = '--', color = 'green')
# bound the x-axis to [-1, 1] for displaying the alternative hypothesis distribution
plt.xlim([-1,1])
plt.show()

# alpha level
# sample size = 10
sample_size = 10
# plot the null hypothesis normal distribution
u0 = 0.5
sigma0 = np.sqrt(u0*(1-u0)/sample_size)
# randomly sample 1000 observations along x-axis from this distribution
x0 = np.sort(np.random.normal(u0, sigma0, 1000))
# compute likelihood of the x
l0 = [scipy.stats.norm(u0, sigma0).pdf(i) for i in x0]
# plot the alternative hypothesis normal distribution
u1 = 0.4
sigma1 = np.sqrt(u1*(1-u1)/sample_size)
# randomly sample 1000 observations along x-axis from this distribution
x1 = np.sort(np.random.normal(u1, sigma1, 1000))
# compute likelihood of the x1
l1 = [scipy.stats.norm(u1, sigma1).pdf(i) for i in x1]
# plot null hypothesis normal distribution
plt.plot(x0, l0, color = 'red')
# plot alternative hypothesis normal distribution
plt.plot(x1, l1, color = 'green')
# this is the vertical line at 0.1 significant level
plt.axvline(scipy.stats.norm(u0, sigma0).ppf(0.1), linewidth = 2, color = 'black')
# this is vertical line for P(Head) under H0
plt.axvline(u0, linestyle = '--', color = 'red')
# this is vertical line for P(Head) under Ha
plt.axvline(u1, linestyle = '--', color = 'green')
# bound the x-axis to [0, 1] since x-axis is probability
plt.xlim([0,1])
plt.show()


# ------------------------------------------------------------------
# The End
# ------------------------------------------------------------------





