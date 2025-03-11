from helper_functions import choose_statistic, population_distribution, sampling_distribution
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import codecademylib3

# task 1: load in the spotify dataset
try:
    spotify_data = pd.read_csv('spotify_data.csv')
except FileNotFoundError:
    print('The file spotify_data.csv was not found.')

# task 2: preview the dataset
print(spotify_data.head())

# task 3: select the relevant column
song_tempos = spotify_data.tempo

# task 5: plot the population distribution with the mean labeled
population_distribution(song_tempos)

# task 6: sampling distribution of the sample mean
sampling_distribution(song_tempos, samp_size=30, stat='Mean')
# The sample mean is an unbiased estimator, because the mean is about the same as the population

# task 8: sampling distribution of the sample minimum
sampling_distribution(song_tempos, samp_size=30, stat='Minimum')
# The sample minimum is a biased estimator, because the population minimum differs from the sampling mean

# task 10: sampling distribution of the sample variance
sampling_distribution(song_tempos, samp_size=30, stat='Variance')
# The variance is an unbiased estimator, because the population variance and the sampling mean is almost the same
# Using the sample variance formula, we can observe that the sample variance is also an unbiased estimator

# task 13: calculate the population mean and standard deviation
# Calculate once and reuse
population_mean = np.mean(song_tempos)
population_std = np.std(song_tempos)
print(f'Population mean: {population_mean:.3f}')
print(f'Population standard deviation: {population_std:.3f}')

# task 14: calculate the standard error using the formula: standard deviation divided by the square root of the sample size
standard_error = population_std/(30**.5)
print(f'Standard error of the sampling distribution: {standard_error:.3f}')

# task 15: calculate the probability of observing an average tempo of 140bpm or lower from a sample of 30 songs using the cumulative distribution function (CDF)
print(f'The probabilty of observing an average tempo of 140 bpm or less is {stats.norm.cdf(140, population_mean, standard_error)*100:.2f}%')

# task 16: calculate the probability of observing an average tempo of 150bpm or higher from a sample of 30 songs
probability_above_150 = 1 - stats.norm.cdf(150, population_mean, standard_error)
print(f'The probability of observing an average tempo of 150 bpm or more is {probability_above_150*100:.2f}%')

# EXTRA

