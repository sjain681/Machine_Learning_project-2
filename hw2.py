import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

Q1 = df1['blood_pressure'].quantile(0.25)
Q3 = df1['blood_pressure'].quantile(0.75)
IQR = Q3 - Q1
fil_ter = (df1['blood_pressure'] >= Q1 - 1.5*IQR) & (df1['blood_pressure'] <= Q3 + 1.5*IQR)
df1 = df1.loc[fil_ter]

df1['family_history'] = df1.family_history.fillna(np.random.choice([True, False], p=[0.1, 0.9]))

data_final = pd.get_dummies(df1)

X = data_final.loc[:, data_final.columns != 'treatment']
y = data_final.loc[:, data_final.columns == 'treatment']
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred = logistic_regression.predict(X_test)
threshold = 0.5
y_val_pred = logistic_regression.predict(X_val)
logistic_regression.predict_proba(X_test)[:,1]
(logistic_regression.predict_proba(X_test)[:,1] > threshold).astype(int)
score = logistic_regression.score(X_test, y_test)
print(score)

d = confusion_matrix(y_test, y_pred)
print(d)

plt.figure()
sns.heatmap(d, annot=True)
print(metrics.accuracy_score(y_test, y_pred))

C = [10, 1, .1, .001]
logreg = LogisticRegressionCV(penalty='l1', solver='liblinear', Cs=C, refit=True)
logreg = logreg.fit(X_train, y_train)
print('Training accuracy:', logreg.score(X_train, y_train))
print('Validation accuracy:', logreg.score(X_val, y_val))
print('Test accuracy:', logreg.score(X_test, y_test))

# get importance
importance = logreg.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
arr = [x for x in range(len(importance))] #X[0]))]
# print(arr)
plt.figure()
plt.bar(arr, importance)

# # generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = logreg.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.figure()
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()

# predict class values
yhat = logreg.predict(X_test)
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.figure()
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()