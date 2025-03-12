# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import codecademylib3

# Load the data and check for missing values
codecademy = pd.read_csv('codecademy.csv')
if codecademy.isnull().values.any():
    print('Warning: Missing data detected!')

# Print the first five rows
print(codecademy.head())

# Create a scatter plot of score vs completed
plt.scatter(codecademy.completed, codecademy.score)
plt.title('Number of other content completed vs. Student score')
plt.xlabel('Number of other content completed prior to this quiz')
plt.ylabel('Student score on a quiz')
plt.show()
plt.clf()

# The scatter plot suggests a linear relationship between 'completed' and 'score'.
# Further analysis with the regression line will confirm this observation.

# Fit a linear regression model to predict 'score' using 'completed' as the predictor
# This will help us understand how the number of completed lessons affects the quiz score
model = sm.OLS.from_formula('score ~ completed', data=codecademy).fit()
print(f'Coefficients predicted: {model.params}')

# The intercept indicates that when the student has completed 0 content before tha quiz, he will get in average 13.21 points on the quiz.
# Meanwhile, the slope of 1.31 indicates that after complete every additional content, the user will get an additional 1.31 in the quiz in average.

# Plot the scatter plot with the line on top
plt.scatter(codecademy.completed, codecademy.score)
plt.plot(codecademy.completed, model.predict(codecademy.completed), color='r')
plt.title('Number of other content completed vs. Student score')
plt.xlabel('Number of other content completed prior to this quiz')
plt.ylabel('Student score on a quiz')
plt.show()
plt.clf()


# Predict score for learner who has completed 20 prior lessons
print(f'For a learner who has completed 20 prior lessons, the predicted quiz score is {model.predict({"completed":[20]}).iloc[0]:.2f} points')

# Calculate fitted values
fitted_values = model.predict(codecademy.completed)

# Calculate residuals
residuals = codecademy.score - fitted_values

# Plot a histogram of residuals to check the normality assumption of the linear regression model
plt.hist(residuals)
plt.title('Residuals on the regression')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
plt.clf()
# It appears to be approximately normal.

# Check homoscedasticity assumption
plt.scatter(fitted_values, residuals)
plt.xlabel('Predicted score values')
plt.ylabel('Residuals')
plt.title('Predicted score vs. Residuals')
plt.show()
plt.clf()
# According to the scatter plot, there is homoscedasticity on the linear regression

# Create a boxplot of score vs lesson
sns.boxplot(x='lesson', y='score', data=codecademy)
plt.show()
plt.clf()

# The learnes who had completed lesson A appears to have a better score in general than those who completed lesson B

# Fit a linear regression to predict score based on which lesson they took
model = sm.OLS.from_formula('score ~ lesson', data=codecademy).fit()
print(f'Coefficients: {model.params}')

# Calculate and print group means and mean difference in one step
grouped_means = codecademy.groupby('lesson').score.mean()
mean_difference = grouped_means.max() - grouped_means.min()
print(f'Means for learners who took each lesson: {grouped_means}\nDifference: {mean_difference:.2f}')
mean_difference = grouped_means.max() - grouped_means.min()
print(f'The difference between the mean for students who took lesson A and the mean for students who took lesson B on quiz is {mean_difference:.2f}')
# The mean for lesson A is the intercept of the regression line, meaning that x=0 indicates students who took lesson A.
# While the mean difference is equal to the slope, what means that lesson B is at x=1 (lesson A score + mean difference)
# Use `sns.lmplot()` to plot `score` vs. `completed` colored by `lesson`
sns.lmplot(y='score', x='completed', hue='lesson', data=codecademy)
plt.show()
plt.clf()