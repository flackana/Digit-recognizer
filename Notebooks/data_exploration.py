""" In this notebook we explore the data."""
#%%
import pandas as pd
from src.functions import plot_a_digit
import seaborn as sns
import matplotlib.pyplot as plt
#%%
train = pd.read_csv('../Data/Data/Raw/train.csv')
x_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]
x_test = pd.read_csv('../Data/Data/Raw/test.csv')
print("x train data shape is: " + str(x_train.shape))
print("x_test data shape is: " + str(x_test.shape))
x_train.to_csv('../Data/Data/Clean/x_train_clean.csv', index=False)
y_train.to_csv('../Data/Data/Clean/Y_train_clean.csv', index = False)
x_test.to_csv('../Data/Data/Clean/x_test_clean.csv', index = False)
# %% Plot a digit to see for example
plot_a_digit(x_train, 13)
# How balanced is the train dataset?
counts = pd.DataFrame(y_train).groupby(['label'])['label'].count()
percentages = [round(counts[i]/sum(counts), 3) for i in range(len(counts))]
counts = pd.DataFrame(counts)

counts.rename(columns = {"label" : "#"})
counts['%'] = percentages

# Plot a pie chart showing the percentages of each digit
data = counts.iloc[:,0].values
labels = [str(i) for i in counts.index]
colors = sns.color_palette('pastel')[0:5]

#create pie chart
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.title('Ratios between digits in the train set')
plt.legend()
plt.savefig('../Figures/Pie_chart_ratios_digits.png')

