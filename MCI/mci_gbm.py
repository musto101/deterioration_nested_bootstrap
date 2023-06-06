import sklearn
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, precision_score, recall_score, accuracy_score, cohen_kappa_score, f1_score, auc, \
    average_precision_score
from sklearn.model_selection import GridSearchCV, cross_val_predict, KFold, RandomizedSearchCV
import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
#from imblearn.pipeline import make_pipeline
import pickle

dat = pd.read_csv('data/mci_progress.csv')

dat['last_DX'].value_counts()
dat['last_DX'] = dat.last_DX.eq('Dementia').mul(1)
dat['last_DX'].value_counts()

dat.head()

X = dat.drop(['last_DX'], axis=1)
y = dat['last_DX']

grid = {'gbm__learning_rate': [0.001, 0.0001],
        'gbm__n_estimators': [10, 100, 500, 1000], 'gbm__min_samples_leaf': [5, 10, 15, 20, 30],
        'gbm__max_depth': [5, 10, 15, 20, 30]}

ros = RandomOverSampler()
knnImp = KNNImputer(n_neighbors=5, add_indicator=True)
gbm = GradientBoostingClassifier(validation_fraction=0.1, n_iter_no_change=10, tol=0.01)

#pipe = make_pipeline(knnImp,  ros, gbm)
#pipe = Pipeline([('imputer', knnImp), ('ros', ros), ('gbm', gbm)])

pipe = Pipeline([('imputer', knnImp), ('ros', ros), ('gbm', gbm)])
inner_cv = KFold(n_splits=5, shuffle=True)
outer_cv = KFold(n_splits=5, shuffle=True)
## hhh
clf = RandomizedSearchCV(estimator=pipe, param_distributions=grid, cv=inner_cv, scoring='roc_auc', n_iter=100)

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

metrics.to_csv('data/gbm_metrics.csv')
