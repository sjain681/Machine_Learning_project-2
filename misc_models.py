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
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from scipy import stats

data = pd.read_csv("ps2_public.csv")
df1 = pd.DataFrame(data)

Q1 = df1['blood_pressure'].quantile(0.25)
Q3 = df1['blood_pressure'].quantile(0.75)
IQR = Q3 - Q1
fil_ter = (df1['blood_pressure'] >= Q1 - 1.5*IQR) & (df1['blood_pressure'] <= Q3 + 1.5*IQR)
df1 = df1.loc[fil_ter]

df1['family_history'] = df1.family_history.fillna(np.random.choice([True, False], p=[0.1, 0.9]))
data_final = pd.get_dummies(df1)

# ==============>

X = data_final.loc[:, data_final.columns != 'treatment']
y = data_final.loc[:, data_final.columns == 'treatment']
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)  # apply scaling on training data
Pipeline(steps=[('standardscaler', StandardScaler()),
                ('logisticregression', LogisticRegression())])

print(pipe.score(X_test, y_test))  # apply scaling on testing data, without leaking training data.

# ==============>

clf = LogisticRegression(solver="liblinear", random_state=0).fit(X_train, y_train)
roc_auc_score(y_test, clf.decision_function(X_test))

# ==============>

from sklearn.linear_model import LogisticRegressionCV

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

C = [10, 1, .1, .001]
logreg = LogisticRegressionCV(penalty='l1', solver='liblinear', Cs=C, refit=True)
logreg = logreg.fit(X_train, y_train)
print('Coefficient of each feature:', logreg.coef_)
print('Training accuracy:', logreg.score(X_train, y_train))
print('Validation accuracy:', logreg.score(X_val, y_val))
print('Test accuracy:', logreg.score(X_test, y_test))

# ==============>

# evaluate a model with a given number of repeats
def evaluate_model(X, y, repeats):
    # prepare the cross-validation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=repeats, random_state=1)
    # create model
    model = LogisticRegression()
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

# configurations to test
repeats = range(1,16)
results = list()
for r in repeats:
    # evaluate using a given number of repeats
    scores = evaluate_model(X_train, y_train, r)
    # summarize
    print('>%d mean=%.4f se=%.3f' % (r, np.mean(scores), stats.sem(scores)))
    # store
    results.append(scores)
# plot the results
plt.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
plt.show()