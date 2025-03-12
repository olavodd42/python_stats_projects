# import libraries
import codecademylib3
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp, binom_test

def perform_ttest(data, popmean, alpha=0.05):
    _, pvalue = ttest_1samp(data, popmean)
    pvalue /= 2
    if pvalue < alpha:
        print(f'Reject the null hypothesis. p-value: {pvalue:.5f}')
    else:
        print(f'Fail to reject the null hypothesis. p-value: {pvalue:.5f}')


# load data
heart = pd.read_csv('heart_disease.csv')
yes_hd = heart[heart.heart_disease == 'presence']
no_hd = heart[heart.heart_disease == 'absence']

chol_hd = yes_hd.chol

mean_chol_hd = chol_hd.mean()
print(f'The mean cholesterol level for patients with heart disease is {mean_chol_hd:.2f}')

print('H0: People with heart disease have an average cholesterol level equal to 240 mg/dl')
print('H0: People with heart disease have an average cholesterol level that is greater than 240 mg/dl')

perform_ttest(chol_hd, 240)
# pvalue /= 2


# print(f'The p-value for this one-sided test is about {pvalue:.5f}')

# With the p-value = 0.00354 that is less than alpha=0.05, we reject the null hypothesis and conclude that the average cholesterol level for heart disease patients is significantly greater than 240 mg/dl.
print()
chol_no_hd = no_hd.chol

mean_chol_no_hd = chol_no_hd.mean()
print(f'The mean cholesterol level for patients without heart disease is {mean_chol_no_hd:.2f}')

print('H0: People without disease have an average cholesterol level equal to 240 mg/dl')
print('H1: People without heart disease have an average cholesterol level that is greater than 240 mg/dl')

perform_ttest(chol_no_hd, 240)

# print(f'The p-value for this one-sided test is about {pvalue:.5f}')

# With the p-value =  0.26397 that is more than alpha=0.05, which reinforce that the average cholesterol level for people without heart disease is 240mg/dl

num_patients = len(heart)
print()
print(f'There are {num_patients} patients in this dataset.')

fbs = heart.fbs
num_highfbs_patients = sum(fbs == 1)
print(f'There are {num_highfbs_patients} patients with fasting blood sugar greater than 120 mg/dl')

percentage_highfbs_patients = 100*num_highfbs_patients/num_patients
print(f'{percentage_highfbs_patients:.2f}% of the patients have high fasting blood sugar levels.')
# 14% is different from the 8% measured in 1988

pvalue = binom_test(num_highfbs_patients, n=num_patients, p=.08, alternative='greater')
print('H0: This sample was drawn from a population where 8% of people have fasting blood sugar > 120 mg/dl')
print('H1: This sample was drawn from a population where more than 8% of people have fasting blood sugar > 120 mg/dl')

print(f'The p-value for this hypotesis test is {pvalue:.5f}')

# The p-value of 0.00005 is less than alpha=.05, we can conclude that the population proportion of high fbs sugar (fbs > 120) is probably greater than 8%