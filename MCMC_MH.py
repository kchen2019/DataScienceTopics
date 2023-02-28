import numpy as np
import matplotlib.pyplot as plt

# # Monte Carlo
# samples = np.random.normal(0, 1, 10000)
# figure, ax = plt.subplots(1,2)
# ax[0].hist(samples)
# ax[0].set_title('Histogram of 10k samples from N(0,1)')
# ax[1].plot(samples)
# ax[1].set_title('10k Random samples')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# # Markov Chain Monte Carlo
# sample_t1 = np.random.normal(0, 1, 1)
# samples = [sample_t1]
# for t in range(1,10000):
#     samples_t = np.random.normal(samples[t-1], 1, 1)
#     samples = np.append(samples, samples_t)
#
# figure, ax = plt.subplots(1,2)
# ax[0].hist(samples)
# ax[0].set_title('Histogram of 10k samples')
# ax[1].plot(samples)
# ax[1].set_title('10k Random samples')
# plt.show()

# MH
import scipy
import random
import numpy as np
#  initial
theta0 = np.random.beta(1,1,1)
prior0 = scipy.stats.beta.pdf(theta0, 1, 1)
head_counts_0 = sum(np.random.randint(2, size=10))
likelihood_0 = scipy.stats.binom.pmf(head_counts_0, 10, theta0)

# collect
theta_accept = theta0
prior_accept = prior0
likelihood_accept = likelihood_0

i = 0
while len(theta_accept) <= 1000:
    theta = np.random.beta(1,1,1)
    prior = scipy.stats.beta.pdf(theta, 1, 1)
    head_counts = sum(np.random.randint(2, size=10))
    likelihood = scipy.stats.binom.pmf(head_counts, 10, theta)
    ratio = (prior*likelihood) / (prior_accept[i] * likelihood_accept[i])
    if ratio >=1:
        theta_accept = np.append(theta_accept, theta)
        prior_accept = np.append(prior_accept, prior)
        likelihood_accept = np.append(likelihood_accept, likelihood)
    else:
        u = np.random.beta(1,1,1)
        if ratio >= u:
            theta_accept = np.append(theta_accept, theta)
            prior_accept = np.append(prior_accept, prior)
            likelihood_accept = np.append(likelihood_accept, likelihood)
        else:
            print('ratio is: ', ratio)
            print('u is: ', u)

# drop the first theta, prior, likelihood
theta_accept = theta_accept[1:]
prior_accept = prior_accept[1:]
likelihood_accept = likelihood_accept[1:]

# t test
mean = np.mean(theta_accept)
print('mean of theta, ', mean)
sd = np.std(theta_accept)
print('std of theta, ', sd)
df = 10
se = sd / np.sqrt(df)
print ('standard error of theta, ', se)
t = np.abs((mean-0.5) / se)
print('t = ', t)

p = (1-scipy.stats.t.cdf(t, 10))*2
print('p value from t test is, ', p)

# sort theta
theta_accept_sorted = np.sort(theta_accept)
theta_25_quantile = theta_accept_sorted[25]
theta_975_quantile = theta_accept_sorted[975]

print('the 2.5% quantile is: ', theta_25_quantile)
print('the 97.5% quantile is: ', theta_975_quantile)

# plot
theta_prior = np.random.beta(1,1,1000)
likelihood = likelihood_accept
theta_posterior = theta_accept

plt.hist(theta_prior, histtype='step', edgecolor='blue', linewidth=2)
plt.hist(likelihood, histtype='step', edgecolor='orange', linewidth=2)
plt.hist(theta_posterior, histtype='step', edgecolor='green', linewidth=2)
plt.legend(['prior distribution', 'likelihood distribution', 'posterior distribution'])
plt.show()