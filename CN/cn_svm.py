import sklearn
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, precision_score, recall_score, accuracy_score, cohen_kappa_score, f1_score, auc, \
    average_precision_score
from sklearn.model_selection import GridSearchCV, cross_val_predict, KFold, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
import numpy as np
from sklearn.pipeline import Pipeline
import pickle
import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

dat = pd.read_csv('data/cn_progress.csv')

dat['last_DX'].value_counts()
dat['last_DX'] = dat.last_DX.eq('MCI_AD').mul(1)
dat['last_DX'].value_counts()

dat.head()

X = dat.drop(['last_DX'], axis=1)
y = dat['last_DX']

grid = {'svm__C': list(range(1, 11, 1))}

ros = RandomOverSampler()
knnImp = KNNImputer(n_neighbors=5, add_indicator=True)
svm = SVC(kernel='linear', probability=True)

pipe = Pipeline([('imputer', knnImp), ('ros', ros), ('svm', svm)])

inner_cv = KFold(n_splits=5, shuffle=True)
outer_cv = KFold(n_splits=5, shuffle=True)

clf = RandomizedSearchCV(estimator=pipe, param_distributions=grid, cv=inner_cv, scoring='roc_auc', n_iter=10)

nested_score = cross_val_predict(clf, X=X, y=y, cv=outer_cv, method='predict_proba', n_jobs=-1)
nested_pred = cross_val_predict(clf, X=X, y=y, cv=outer_cv, method='predict', n_jobs=-1)

y_probs = nested_score[:, 1]
fpr, tpr, thresholds = roc_curve(y, y_probs)

youdenJ = tpr - fpr
index = np.argmax(youdenJ)
thresholdOpt = round(thresholds[index], ndigits=4)

y_pred = (nested_score[:, 1] >= thresholdOpt).astype(bool)

precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
accuracy = accuracy_score(y, y_pred)
kappa = cohen_kappa_score(y, y_pred)
f1 = f1_score(y, y_pred)
auc_score = auc(fpr, tpr)
prc = average_precision_score(y, y_pred)

metrics = pd.DataFrame([[precision, recall, accuracy, kappa, f1, auc_score, prc]],
                       columns=['precision', 'recall', 'accuracy', 'kappa', 'f1', 'auc_score', 'prc'])

metrics.to_csv('data/cn_svm_metrics.csv')
