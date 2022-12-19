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
df1['family_history'] = df1.family_history.fillna(np.random.choice([True, False], p=[0.1, 0.9]))

# the independent variables set
X = df1[['treatment', 'age', 'blood_pressure', 'TestA', 'TestB', 'GeneD', 'GeneE', 'GeneF']]
# VIF dataframe 
vif_data = pd.DataFrame() 
vif_data["feature"] = X.columns
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))]
print(vif_data)

Q1 = df1['blood_pressure'].quantile(0.25)
Q3 = df1['blood_pressure'].quantile(0.75)
IQR = Q3 - Q1
fil_ter = (df1['blood_pressure'] >= Q1 - 1.5*IQR) & (df1['blood_pressure'] <= Q3 + 1.5*IQR)
df1 = df1.loc[fil_ter]

df1['family_history'] = df1.family_history.fillna(np.random.choice([True, False], p=[0.1, 0.9]))

data_final = pd.get_dummies(df1)

cols = [l for l in list(data_final.columns) if "treatment" not in l and "TestA" not in l and "TestB" not in l]

X = data_final[cols]
y = data_final['treatment']
selector = SelectKBest(chi2, k=5)
selector.fit_transform(X, y)
cols = selector.get_support(indices=True)
X_new = X.iloc[:,cols]
print(X_new.columns)