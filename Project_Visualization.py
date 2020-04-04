import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

dataset = pd.read_csv('churn_data.csv')

## Cleaning Data

# Removing NaN
'''
dataset.isna().any()
dataset.isna().sum()
'''
dataset = dataset[pd.notnull(dataset['age'])]
dataset = dataset.drop(columns = ['credit_score', 'rewards_earned'])

## Visualization
# Histogram
dataset2 = dataset.drop(columns = ['user', 'churn'])
Histogram_Figure = plt.figure(figsize = (15, 12))
plt.suptitle('Histogram of Numerical Columns', fontsize = 20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(6, 5, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i-1], fontsize = 8)

    vals = np.size(dataset2.iloc[:, i-1].unique())

    plt.hist(dataset2.iloc[:, i -1], bins = vals, color = '#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
Histogram_Figure.show()

# Pie Chart
dataset2 = dataset[['housing', 'is_referred', 'app_downloaded',
                    'web_user', 'app_web_user', 'ios_user',
                    'android_user', 'registered_phones', 'payment_type',
                    'waiting_4_loan', 'cancelled_loan', 'rejected_loan',
                    'received_loan', 'zodiac_sign', 'left_for_two_month_plus', 'left_for_one_month',
                    'is_referred']]
Pie_Figure = plt.figure(figsize = (15, 12))
plt.suptitle('Pie Chart Distributions', fontsize = 20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(6, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i-1], fontsize = 8)
    values = dataset2.iloc[:, i-1].value_counts(normalize = True).values
    index = dataset2.iloc[:, i-1].value_counts(normalize = True).index
    plt.pie(values, labels = index, autopct = '%1.1f%%')
    plt.axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
Pie_Figure.show()

#Exploring Uneven Features
'''
dataset[dataset2.waiting_4_loan == 1].churn.value_counts()
dataset[dataset2.cancelled_loan == 1].churn.value_counts()
dataset[dataset2.received_4_loan == 1].churn.value_counts()
dataset[dataset2.rejected_loan == 1].churn.value_counts()
dataset[dataset2.left_for_one_month == 1].churn.value_counts()
'''

#Correlation Plot
Correlation_Figure = plt.figure()
dataset.drop(columns = ['churn', 'user', 'housing',
                        'payment_type', 'zodiac_sign']).corrwith(dataset.churn).plot.bar(
                            figsize = (40, 30), title = 'Correlation with the Response Variable', fontsize = 10,
                            rot = 45, grid = True)

Correlation_Figure.show()

# Correlation Matrix
sn.set(style = 'white')

# Compute correlation matrix
corr = dataset.drop(columns = ['user', 'churn']).corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up matplotlib figure
Correlation_Matrix, ax = plt.subplots(figsize=(18,15))
Correlation_Matrix.suptitle("Correlation Matrix", fontsize = 20)

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220,10, as_cmap= True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
           square=True, linewidths=1, cbar_kws={"shrink": .5})

Correlation_Matrix.show()

