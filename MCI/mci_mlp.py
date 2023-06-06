import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, precision_score, recall_score, accuracy_score, cohen_kappa_score, f1_score, auc, \
    average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_predict, KFold, RandomizedSearchCV
import numpy as np
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import resample

dat = pd.read_csv('data/mci_progress.csv')

dat['last_DX'].value_counts()
dat['last_DX'] = dat.last_DX.eq('Dementia').mul(1)
dat['last_DX'].value_counts()

dat.head()

X = dat.drop(['last_DX'], axis=1)
y = dat['last_DX']
mcRep = 10
mcMetrics = pd.DataFrame([])

grid = {'mlp__hidden_layer_sizes': [(10,), (50, ), (100, ), (10, 10), (50, 50), (100, 100), (10, 10, 10), (50, 50, 50),
                                    (100, 100, 100)], 'mlp__max_iter': [10000],
        'mlp__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5], 'mlp__learning_rate': ['constant', 'invscaling', 'adaptive'],
        'mlp__learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
        'mlp__activation': ['identity', 'logistic', 'tanh', 'relu'], 'mlp__solver': ['adam', 'sgd', 'lbfgs']}

knnImp = KNNImputer(n_neighbors=5, add_indicator=True)
scale = StandardScaler()
rs = resample(replace=True, n_samples=500)
ros = RandomOverSampler()
mlp = MLPClassifier(early_stopping=True, tol=0.01, n_iter_no_change=20, validation_fraction=0.1)

pipe = Pipeline([('imputer', knnImp), ('scale', scale), ('rs', rs), ('ros', ros), ('mlp', mlp)])

inner_cv = KFold(n_splits=5, shuffle=True)
outer_cv = KFold(n_splits=5, shuffle=True)

for i in range(mcRep):
    clf = RandomizedSearchCV(estimator=pipe, param_distributions=grid, cv=inner_cv, scoring='roc_auc', n_iter=10000)

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

    metrics = pd.DataFrame([[i, precision, recall, accuracy, kappa, f1, auc_score, prc]],
                           columns=['iter', 'precision', 'recall', 'accuracy', 'kappa', 'f1', 'auc_score', 'prc'])
    mcMetrics = mcMetrics.append(metrics)

mcMetrics.to_csv('data/mlp_metrics.csv')
