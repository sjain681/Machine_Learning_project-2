import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

import math

data = pd.read_csv("ps2_public.csv")
df1 = pd.DataFrame(data)

# the independent variables set 
X = df1[['treatment', 'age', 'blood_pressure', 'TestA', 'TestB', 'GeneD', 'GeneE', 'GeneF']]
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = X.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))]
print(vif_data)

plt.figure()
corr = df1.corr()# plot the heatmap
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
plt.title("Correlation Heatmap")

Q1 = df1['blood_pressure'].quantile(0.25)
Q3 = df1['blood_pressure'].quantile(0.75)
IQR = Q3 - Q1
fil_ter = (df1['blood_pressure'] >= Q1 - 1.5*IQR) & (df1['blood_pressure'] <= Q3 + 1.5*IQR)
df1 = df1.loc[fil_ter]

df1.plot(kind="scatter", x="blood_pressure", y="age")
plt.title("blood_pressure vs. age")

df1['family_history'] = df1.family_history.fillna(np.random.choice([True, False], p=[0.1, 0.9]))

data_final = pd.get_dummies(df1)

X = data_final.loc[:, data_final.columns != 'treatment']
y = data_final.loc[:, data_final.columns == 'treatment']

plt.figure()
df1["age"].hist(grid = False) # bins = 10, grid = False
plt.title("age Histogram")

plt.figure()
df1["blood_pressure"].hist(bins = 10, grid = False)
plt.title("blood_pressure Histogram")

plt.figure()
df1["TestA"].hist(bins = 10, grid = False)
plt.title("TestA Histogram")

df1.plot(kind="scatter", x="TestA", y="age")
plt.title("age vs TestA")

attributes = ['age', 'blood_pressure', 'TestA', 'TestB']
scatter_matrix(df1[attributes])

plt.show()