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
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

data = pd.read_csv("ps2_public.csv")
df1 = pd.DataFrame(data)
print(df1.family_history.value_counts())

"""
Q1 = df1['blood_pressure'].quantile(0.25)
Q3 = df1['blood_pressure'].quantile(0.75)
IQR = Q3 - Q1
fil_ter = (df1['blood_pressure'] >= Q1 - 1.5*IQR) & (df1['blood_pressure'] <= Q3 + 1.5*IQR)
df1 = df1.loc[fil_ter]

df1['family_history'] = df1.family_history.fillna(np.random.choice([True, False], p=[0.1, 0.9]))

data_final = pd.get_dummies(df1)
print(data_final.columns)
print(data_final.info())

"""