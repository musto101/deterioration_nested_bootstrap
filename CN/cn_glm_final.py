import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, precision_score, recall_score, accuracy_score, cohen_kappa_score, f1_score, auc, \
    average_precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_predict, StratifiedKFold, RandomizedSearchCV
import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import resample
import joblib
import os
from google.cloud import storage
dat = pd.read_csv('data/cn_progress.csv')

dat['last_DX'].value_counts()
dat['last_DX'] = dat.last_DX.eq('MCI_AD').mul(1)
dat['last_DX'].value_counts()

dat.head()

X = dat.drop(['last_DX'], axis=1)
y = dat['last_DX']
mcRep = 1
mcMetrics = pd.DataFrame([])

grid = {'elasticnet__alpha': np.arange(0.01, 1, 0.01), 'elasticnet__l1_ratio': np.arange(0.01, 1, 0.01)}

knnImp = KNNImputer(n_neighbors=5)
scale = StandardScaler()
rs = resample(replace=True, n_samples=500)
ros = RandomOverSampler(random_state=1)
# X_res, y_res = ros.fit_resample(X, y)
elasticnet = SGDClassifier(loss='log', penalty='elasticnet', max_iter=10000, tol=1e-3, n_jobs=-1)

pipe = Pipeline([('imputer', knnImp), ('scale', scale), ('ros', ros), ('rs', rs), ('elasticnet', elasticnet)])

inner_cv = StratifiedKFold(n_splits=5, shuffle=True)
# outer_cv = StratifiedKFold(n_splits=5, shuffle=True)

for i in range(mcRep):
    clf = GridSearchCV(estimator=pipe, param_grid=grid, cv=inner_cv, scoring='roc_auc')
    clf.fit(X, y)

    nested_score = clf.predict_proba(X)
    nested_pred = clf.predict(X)

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
    cm = confusion_matrix(y, y_pred)
    metrics = pd.DataFrame([[i, precision, recall, accuracy, kappa, f1, auc_score, prc]],
                           columns=['iter', 'precision', 'recall', 'accuracy', 'kappa', 'f1', 'auc_score', 'prc'])
    mcMetrics = mcMetrics.append(metrics)

final_model = clf.best_estimator_
filename = 'final_models/cn_glm_final.joblib'
joblib.dump(final_model, open(filename, 'wb'))

local_path = 'final_models/cn_glm_final.joblib'

# Upload model artifact to Cloud Storage
model_directory = 'deter_nested_cn_glm'
storage_path = os.path.join(model_directory, local_path)
blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
blob.upload_from_filename(local_path)

# mcMetrics.to_csv('data/cn_glm_metrics.csv')
