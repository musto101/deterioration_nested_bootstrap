from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_curve, precision_score, recall_score, accuracy_score, cohen_kappa_score, f1_score, auc, \
    average_precision_score
from keras.callbacks import Callback
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.initializers import he_normal
from keras import backend as K

dat = pd.read_csv('data/mci_progress.csv')

dat['last_DX'].value_counts()
dat['last_DX'] = dat.last_DX.eq('Dementia').mul(1)
dat['last_DX'].value_counts()

dat.head()

X = dat.drop(['last_DX'], axis=1)
y = dat['last_DX']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shape of train and test sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import he_normal
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
import numpy as np

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply KNN imputation to training and test sets
imputer = KNNImputer(n_neighbors=5)
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {
    'batch_size': [16, 32, 64, 128],
    'epochs': [50, 100, 200],
    'optimizer': ['adam', 'rmsprop'],
   # 'learning_rate': [0.1, 0.01, 0.001],
   # 'momentum': [0.9, 0.95, 0.99],
    'dropout_rate': [0.0, 0.1, 0.2],
    'l1_penalty': [0.0, 0.001, 0.01],
    'l2_penalty': [0.0, 0.001, 0.01],
    'activation': ['relu', 'tanh', 'selu', 'elu'],
    'hidden_units': [(32,), (64,), (128,), (256,), (512,), (32, 16), (64, 32), (128, 64)]
}
# }
# # Define the neural network model
# model = Sequential()
# model.add(Dense(units=64, activation='elu', kernel_initializer=he_normal(seed=42), input_dim=X_train.shape[1]))
# model.add(Dense(units=32, activation='elu', kernel_initializer=he_normal(seed=42)))
# model.add(Dense(units=1, activation='sigmoid', kernel_initializer=he_normal(seed=42)))
#
# # Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
#
#


# Define the OneCycleLR callback
class OneCycleLR(Callback):
    def __init__(self, max_lr, epochs, batch_size, samples, verbose=0):
        self.max_lr = max_lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.samples = samples
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        self.current_epoch = 0
        self.total_iterations = self.epochs * (self.samples // self.batch_size)
        self.max_iterations = self.epochs * (self.samples // self.batch_size)
        self.step_size = self.max_iterations // 2

    def on_batch_end(self, batch, logs=None):
        if self.current_iteration == self.step_size:
            self.current_iteration = 0
            self.step_size = self.max_iterations - self.step_size
            lr = self.max_lr / 10
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print("Learning rate changed to: ", lr)

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        self.current_iteration = 0


# Define the OneCycleLR callback with your desired hyperparameters
one_cycle_lr = OneCycleLR(max_lr=0.01, epochs=100, batch_size=32, samples=X_train.shape[0], verbose=1)
#
# # Compile the model
# model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='binary_crossentropy', metrics=['AUC'])
#
# # Train the model with the OneCycleLR callback and early stopping
# model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_split=0.1, callbacks=[one_cycle_lr, early_stopping])

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.activations import relu, tanh, selu
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import he_uniform


def create_model(hidden_units=(64,), activation='relu', dropout_rate=0.0, l1_penalty=0.0, l2_penalty=0.0, optimizer='adam', learning_rate=0.001):
    model = Sequential()
    model.add(Dense(hidden_units[0], input_shape=(X_train.shape[1],), activation=activation, kernel_initializer=he_uniform(), kernel_regularizer=l1_l2(l1=l1_penalty, l2=l2_penalty)))
    for units in hidden_units[1:]:
        model.add(Dense(units, activation=activation, kernel_initializer=he_uniform(), kernel_regularizer=l1_l2(l1=l1_penalty, l2=l2_penalty)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['AUC'])
    return model


model = KerasClassifier(build_fn=create_model, verbose=0)
random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=100, cv=5, verbose=1, random_state=42)
random_search.fit(X_train, y_train, callbacks=[one_cycle_lr, early_stopping], validation_split=0.1)
# Get the best model from the random search
best_model = random_search.best_estimator_.model

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# Calculate the true positive rate and false positive rate for different threshold values
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Calculate the Youden index (J) for each threshold value
J_values = tpr - fpr
best_J_index = np.argmax(J_values)
best_threshold = thresholds[best_J_index]

# Apply the best threshold to the predictions to get binary class labels
y_pred_binary = (y_pred >= best_threshold).astype(int)

# Calculate performance metrics for the model on the test set
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
kappa = cohen_kappa_score(y_test, y_pred_binary)
auc = roc_auc_score(y_test, y_pred)

# Create a dictionary of the results and convert it to a pandas dataframe
results_dict = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'Kappa': kappa, 'AUC': auc}
results_df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['Value'])

# Save the results as a CSV file
results_df.to_csv('data/AnnModelResults.csv', index=True)