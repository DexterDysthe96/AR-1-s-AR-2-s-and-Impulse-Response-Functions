# Dexter Dysthe
# Dr. Geert Bekaert
# B9325: Financial Econometrics, Time Series
# 7 April 2022

import math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.despine()
np.random.seed(51796)


# -------------------------------------------------- Question 1 -------------------------------------------------- #

def std_locscale_model_sim(N_sim=1000, T=100, mu=0.01, sigma=0.05):
    y_0 = np.random.normal(mu, sigma)

    # The list means will store the N means for the simulated y_1,...,y_T
    # The list sds will store the N standard deviations for the simulated y_1,...,y_T
    means = list()
    sds = list()
    for nn in range(N_sim):
        # Simulate T-many y_t's
        y = list()
        y.append(y_0)
        y.extend(np.random.normal(mu, sigma, T-1))

        # Calculate the mean of the simulated y_t's
        mean_y = np.mean(y)
        means.append(mean_y)

        # Calculate the sd of the simulated y_t's
        sd_y = np.std(y, ddof=1)
        sds.append(sd_y)

    return means, sds


def AR_one_with_drift(N_sim=1000, T=100, mu=0.01, sigma=0.05, rho=0.9):
    x_0 = np.random.normal(mu, sigma)

    # In order for the unconditional mean and variance of the AR(1) w/ drift to match the
    # unconditional mean and variance of the standard location model, we set mu_x and sigma_x
    # equal to the following
    mu_x = mu * (1 - rho)
    sigma_x = sigma * np.sqrt(1 - rho**2)

    # The list means will store the N means for the simulated y_1,...,y_T
    # The list sds will store the N standard deviations for the simulated y_1,...,y_T
    means = list()
    sds = list()
    for nn in range(N_sim):
        # Simulate T-many x_t's
        x = list()
        x.append(x_0)
        [x.append(mu_x + rho*x[ii] + np.random.normal(0, sigma_x)) for ii in range(T-1)]

        # Calculate the mean of the simulated x_t's
        mean_x = np.mean(x)
        means.append(mean_x)

        # Calculate the sd of the simulated x_t's
        sd_x = np.std(x, ddof=1)
        sds.append(sd_x)

    return means, sds


def sample_stats_and_plotting(tup, model_type):
    # tup should be a tuple of lists, the first list corresponding to a list of means and the
    # second list corresponding to a list of standard deviations
    means, sds = tup
    N_sim = len(means)

    print('-------------------------- {} --------------------------'.format(model_type))
    # Sample statistics and plotting for means
    print('Standard Deviation of Sample Means: {}'.format(np.std(means, ddof=1)))
    print('Skewness of Sample Means: {}'.format(stats.skew(np.asarray(means))))
    print('Kurtosis of Sample Means: {}'.format(stats.kurtosis(np.asarray(means))))
    print('\n')
    plt.hist(means, edgecolor='black', bins=math.ceil(np.sqrt(N_sim)))
    plt.title('Distribution of Sample Mean Estimates ({})'.format(model_type))
    plt.show()

    # Sample statistics and plotting for standard deviations
    print('Standard Deviation of Sample Standard Deviations: {}'.format(np.std(sds, ddof=1)))
    print('Skewness of Sample Standard Deviations: {}'.format(stats.skew(np.asarray(sds))))
    print('Kurtosis of Sample Standard Deviations: {}'.format(stats.kurtosis(np.asarray(sds))))
    print('\n')
    plt.hist(sds, edgecolor='black', bins=math.ceil(np.sqrt(N_sim)))
    plt.title('Distribution of Sample Standard Deviation Estimates ({})'.format(model_type))
    plt.show()


# SD of Sample Means:
#     - Model 1: 0.00478
#     - Model 2: 0.02013
# Skewness of Sample Means:
#     - Model 1: 0.0109
#     - Model 2: 0.09495
# Kurtosis of Sample Means:
#     - Model 1: 0.1459
#     - Model 2: -0.1294
# SD of Sample SDs:
#     - Model 1: 0.003569
#     - Model 2: 0.009034
# Skewness of Sample SDs:
#     - Model 1: 0.03056
#     - Model 2: 0.68306
# Kurtosis of Sample SDs:
#     - Model 1: -0.2264
#     - Model 2:  0.7418
#
# As can be seen from the SD of the sample means and the sample SDs, model 1 has considerably
# smaller sampling uncertainty. The 3rd and 4th sampling moments are also easily distinguished
# from each other. This indicates that sample moments are an effective tool for testing whether
# a given time series of data follows model 1 or model 2. Since many of the sample moments differ
# substantially from each other in the 2 considered models, one can see that a robust way of testing
# one model specification vs. the other is via a moment test.

print('\n')
sample_stats_and_plotting(std_locscale_model_sim(), 'Standard Loc/Scale Model')

print('\n')
sample_stats_and_plotting(AR_one_with_drift(), 'AR(1) with Drift')


# -------------------------------------------------- Question 2 -------------------------------------------------- #
# Answers for parts 2 through 4 can be found in the write-up included in our submitted zip file.

def ACF_AR_two(num_lags=20, phi_1=1.1, phi_2=-0.25):
    lags = [lag for lag in range(num_lags+1)]

    # Initialize first two values for the ACF
    acf = list()
    acf.extend([1, phi_1 / (1 - phi_2)])

    # Use the difference equations which generate the ACF for an AR(2) as shown on page 26
    # of the Lecture 1 notes.
    [acf.append(phi_1*acf[ii-1] + phi_2*acf[ii-2]) for ii in range(2, num_lags+1)]

    # Plot the ACF for the specified number of lags
    plt.plot(range(num_lags+1), acf)
    plt.xticks(range(num_lags+1), lags)
    plt.xlabel('Lags')
    plt.ylabel('ACF')
    plt.title('Autocorrelation Function of an AR(2)')
    plt.show()


ACF_AR_two()


def impulse_response(phi_1=1.1, phi_2=-0.25, j_max=60):
    # We are considering an AR(2) w/o drift, i.e. x_t = phi_1 x_{t-1} + phi_2 x_{t-2} + eps_t. As
    # instructed in the problem statement, we set x_{t-1} = x_{t-2} = 0 and eps_t = 1, eps_{t+j} = 0
    # for all positive integers j. Thus, x_t = 1 and x_{t+1} = phi_1.
    x = list()
    x.extend([0, 1, phi_1])
    [x.append(phi_1*x[ii-1] + phi_2*x[ii-2]) for ii in range(3, j_max+2)]

    plt.plot(range(-1, j_max+1), x)
    j_ticks = [-1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    plt.xticks([ii for ii in j_ticks], [ii for ii in j_ticks])
    plt.xlabel('j')
    plt.title('Impulse-Response (phi_1 = {}, phi_2 = {})'.format(phi_1, phi_2))
    plt.show()


impulse_response()


# -------------------------------------------------- Question 3 -------------------------------------------------- #

def AR_one_monte_carlo(N_sim=10000, T=100, mu=0.01, sigma=0.05, rho=0.9):
    expec_ar_one = mu / (1 - rho)
    var_ar_one = sigma / (1 - rho**2)
    # As discussed in Q1 in the homework pdf, we draw r_0 from a normal
    ar_one_0 = np.random.normal(expec_ar_one, np.sqrt(var_ar_one))

    all_walks = list()
    for nn in range(N_sim):
        ar_one = list()
        ar_one.append(ar_one_0)
        [ar_one.append(mu + rho * ar_one[ii] + np.random.normal(0, sigma)) for ii in range(T - 1)]
        all_walks.append(ar_one)

    # Take transpose for plotting purposes
    return np.array(all_walks).transpose(), np.linspace(0, T, num=T)


def plot_AR_one(N_sim, T, mu, sigma, rho):
    sample_paths100, t = AR_one_monte_carlo(N_sim, T, mu, sigma, rho)
    plt.plot(t, sample_paths100)
    plt.title('{} Sample Paths of an AR(1); sigma = {}%, rho = {}'.format(N_sim, 100*sigma, rho))
    plt.xlabel('Time')
    plt.show()


# Plot AR(1) for various values of sigma. As we increase sigma, the roughness of the paths increases.
# When sigma = 0.03, the AR(1) oscillates (approximately) between -0.2 and 0.3; when, on the other
# hand, sigma = 0.30, the AR(1) oscillates (approximately) between -2.5 and 3.
for vol in [0.03, 0.05, 0.08, 0.15, 0.30]:
    plot_AR_one(10, 1000, 0.01, vol, 0.9)
    plot_AR_one(10, 1000, 0.01, vol, -0.9)


def ACF_AR_one(N_sim, T, mu, sigma, rho, num_coefficients):
    # num_coefficients specifies how many ACF coefficients we wish to calculate

    # List of N_sim-many AR(1) sample paths of length T
    paths = AR_one_monte_carlo(N_sim, T, mu, sigma, rho)[0].transpose()

    # Get the sample means for each of the N_sim simulated sample paths
    r_bar = paths.mean(axis=1)

    # The jth entry of this list will contain a list of the N_sim-many sample autocorrelation coefficients
    # for the jth lag
    autocorr_coefficients = list()
    # The first sample autocorrelation coefficient is always equal to one by definition
    autocorr_coefficients.append(list(np.ones(N_sim)))

    # For each j calculate the jth ACF coefficient for each of the N_sim many sample paths.
    for j in range(1, num_coefficients+1):
        # Create a list to store rho_hat_j, the sample jth lag autocorrelation (from page 110 of Hamilton),
        # for each of the N_sim simulated sample paths.
        rho_hat_js = list()

        # Counter to keep track of which path we are on -- used in order to match up the correct r_bar value
        idx = 0
        for path in paths:
            x_t = path[j:]
            x_t_jlags = path[:-j]

            # Calculate the sample variance
            gamma_hat_0 = np.mean((path - r_bar[idx]) ** 2)

            # Using the formula from page 110 of Hamilton, calculate the jth sample autocovariance
            gamma_hat_j = 1 / T * (np.inner(x_t - r_bar[idx], x_t_jlags - r_bar[idx]))

            # Calculate the jth ACF coefficient and add it to the list rho_hat_js which will store
            # all of the jth ACF coefficients for each of the N_sim simulated sample paths
            rho_hat_j = gamma_hat_j / gamma_hat_0
            rho_hat_js.append(rho_hat_j)

            idx += 1

        # Add the list of N_sim many jth ACF coefficients to the list of the first num_coefficients
        # ACF coefficients
        autocorr_coefficients.append(rho_hat_js)

    return autocorr_coefficients


# ------------------ Plotting of the ACFs (T = 100, 1000, 10000) ------------------ #
ACF_coefficients1 = ACF_AR_one(10000, 100, 0.01, 0.05, 0.9, 99)
ACF_coefficients_arr1 = np.array(ACF_coefficients1)
plt.plot(range(100), ACF_coefficients_arr1)
x_ticks_vals1 = [9*tick for tick in range(12)]
plt.xticks([jj for jj in x_ticks_vals1], [jj for jj in x_ticks_vals1])
plt.xlabel('Lags')
plt.title('Sample Autocorrelation Functions (T = 100)')
plt.show()

# Only considering ~200 lags
ACF_coefficients2 = ACF_AR_one(10000, 1000, 0.01, 0.05, 0.9, 198)
ACF_coefficients_arr2 = np.array(ACF_coefficients2)
plt.plot(range(199), ACF_coefficients_arr2)
x_ticks_vals2 = [18*tick for tick in range(12)]
plt.xticks([jj for jj in x_ticks_vals2], [jj for jj in x_ticks_vals2])
plt.xlabel('Lags')
plt.title('Sample Autocorrelation Functions (T = 1000)')
plt.show()

# Only considering ~450 lags
ACF_coefficients3 = ACF_AR_one(10000, 10000, 0.01, 0.05, 0.9, 439)
ACF_coefficients_arr3 = np.array(ACF_coefficients3)
plt.plot(range(440), ACF_coefficients_arr3)
x_ticks_vals3 = [40*tick for tick in range(12)]
plt.xticks([jj for jj in x_ticks_vals3], [jj for jj in x_ticks_vals3])
plt.xlabel('Lags')
plt.title('Sample Autocorrelation Functions (T = 10000)')
plt.show()


# Function to calculate both finite sample and asymptotic standard errors
def standard_errors(N_sim, T, mu, sigma, rho, num_coefficients):
    # List of T-many lists where the jth list stores the N_sim-many sample autocorrelation coefficients
    # for the jth lag.
    sample_autocorr_coefficients = ACF_AR_one(N_sim, T, mu, sigma, rho, num_coefficients)

    # List to store the averages of the asymptotic standard errors for each lag from 0 up to T - 1
    mean_asym_ses = list()

    # List to store the finite sample standard errors for each lag from 0 up to T - 1
    finite_sample_ses = list()

    # gamma_hat_js is a list of N_sim-many sample autocorrelation coefficients
    for gamma_hat_js in sample_autocorr_coefficients:
        asym_ses = np.sqrt(1/T * (1 - np.asarray(gamma_hat_js)**2))
        mean_asym_se = np.mean(asym_ses)
        mean_asym_ses.append(mean_asym_se)

        # Calculate the standard deviation of the N_sim-many sample autocorrelation coefficients
        finite_sample_se = np.std(gamma_hat_js, ddof=1)
        finite_sample_ses.append(finite_sample_se)


    return [list(tup) for tup in zip(mean_asym_ses, finite_sample_ses)]


plt.plot(range(100), standard_errors(1000, 100, 0.01, 0.05, 0.9, 99))
plt.xticks([jj for jj in x_ticks_vals1], [jj for jj in x_ticks_vals1])
plt.title('Comparison of SEs (T=100)')
plt.legend(['Asymptotic SEs', 'Finite Sample SEs'])
plt.show()

plt.plot(range(199), standard_errors(1000, 1000, 0.01, 0.05, 0.9, 198))
plt.xticks([jj for jj in x_ticks_vals2], [jj for jj in x_ticks_vals2])
plt.title('Comparison of SEs (T=1000)')
plt.legend(['Asymptotic SEs', 'Finite Sample SEs'])
plt.show()

plt.plot(range(440), standard_errors(1000, 10000, 0.01, 0.05, 0.9, 439))
plt.xticks([jj for jj in x_ticks_vals3], [jj for jj in x_ticks_vals3])
plt.title('Comparison of SEs (T=10000)')
plt.legend(['Asymptotic SEs', 'Finite Sample SEs'])
plt.show()


def OLS_estimates_and_SEs(T, rho, N_sim=10000, mu=0.01, sigma=0.05):
    # List of N_sim-many AR(1) sample paths of length T
    paths = AR_one_monte_carlo(N_sim, T, mu, sigma, rho)[0].transpose()

    beta_ols_vals = list()
    se_beta_ols_vals = list()
    # Loop over each of the N time series
    for path in paths:
        # Extract the lagged regressor
        x = np.asarray(path[:-1])

        # Extract dependent variable
        y = np.asarray(path[1:])

        # Calculate sample means of independent and dependent variables
        x_bar = np.mean(x)
        y_bar = np.mean(y)

        # Calculate OLS estimate and standard error. Doing w/o statsmodels library
        # in order to decrease runtime complexity.
        beta_ols = np.inner(x - x_bar, y - y_bar) / np.sum((x - x_bar)**2)
        se_beta_ols = sigma / np.sqrt(np.sum((x - x_bar)**2))

        beta_ols_vals.append(beta_ols)
        se_beta_ols_vals.append(se_beta_ols)

    # Calculate the average of our N many OLS estimates and standard errors
    mean_of_beta_ols = np.mean(beta_ols_vals)
    mean_of_beta_ols_ses = np.mean(se_beta_ols_vals)

    return mean_of_beta_ols, mean_of_beta_ols_ses


ols_est_and_ses1 = list()
ols_est_and_ses2 = list()
ols_est_and_ses3 = list()
for T_val in [100, 1000, 10000]:
    ols_est_and_ses1.append(OLS_estimates_and_SEs(T_val, 0.9))
    ols_est_and_ses2.append(OLS_estimates_and_SEs(T_val, 0.95))
    ols_est_and_ses3.append(OLS_estimates_and_SEs(T_val, 0.999999))


# As you increase rho towards 1, the time series models "converge" to a non-stationary model in which
# case the OLS estimator is no longer consistent. In general, OLS estimators for AR(1) models are biased,
# however, for stationary AR(1)'s the OLS estimator is consistent; we lose consistency when we do not have
# stationarity. The OLS estimator is biased in the case of an AR(1) model because the expectation of the
# product of r_t and eps_t is nonzero.
ols_est_and_ses_dict1 = {'rho = 0.9': ols_est_and_ses1}
ols_est_and_ses_dict2 = {'rho = 0.95': ols_est_and_ses2}
ols_est_and_ses_dict3 = {'rho = 0.999999': ols_est_and_ses3}
ols_est_and_ses_df1 = pd.DataFrame(ols_est_and_ses_dict1, index=['T=100', 'T=1000', 'T=10000'])
ols_est_and_ses_df2 = pd.DataFrame(ols_est_and_ses_dict2, index=['T=100', 'T=1000', 'T=10000'])
ols_est_and_ses_df3 = pd.DataFrame(ols_est_and_ses_dict3, index=['T=100', 'T=1000', 'T=10000'])

# First entry in the tuple corresponds to the OLS estimate and the second entry corresponds to the standard
# error
print(ols_est_and_ses_df1, '\n')
print(ols_est_and_ses_df2, '\n')
print(ols_est_and_ses_df3, '\n')
