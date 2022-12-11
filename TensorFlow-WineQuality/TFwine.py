import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score

""" Lukasz Cettler s20168, Wojciech Mierzejewski s21617
    Tensor Flow Classification Modelling 
    Dataset: wine quality dataset from KAGGLE
"""

df = pd.read_csv('data/winequalityN.csv')
"""Data input from CSV file"""

df = df.dropna()
"""Dataset missing values removal"""

df['is_good_wine'] = [1 if quality >= 6 else 0 for quality in df['quality']]
"""Binary classification of wine quality, good for quality >=6, poor for less than 6"""
df.drop('quality', axis=1, inplace=True)
"""Quality key removal"""

X = df.drop('is_good_wine', axis=1)
"""Data split for training and testing sets 80/20, features will be everything but target variable"""
y = df['is_good_wine']
"""Target variable"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""4 sets data split, size 80/20"""

print(X_train.shape)
print(X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
""" Data scaling, some features have next to 0 values, while some are at +150, this may confuse neural network, considering some
    features of higher scale a more influential
"""

print(X_train_scaled[:3])
"""Standard scikit scaling applied, data valleys are much more shallow ones now"""
"""Model training"""
"""Output layer structure - single neuron activated with sigmoid probability function with easy good or bad wine assigment"""
"""Loss function binary cross entropy"""
"""Class balance - do we have same amount of good and bad wines in set"""
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
""" Sequential model use example 12 features to 128 neurons, activated with Rectified linear unit activation function.
    One layer neuron activated with sigmoid function for a better probabilistic assigment
"""

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(lr=0.03),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)
"""Model compilation with binary cross section entropy loss function, accuracy, precision and recall as metrics, Adam function optimizer"""
history = model.fit(X_train_scaled, y_train, epochs=100)
"""Model training, results saved, epochs set to 100"""

predictions = model.predict(X_test_scaled)
"""Run model on X_test_scaled"""
prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(predictions)]
"""Prediction classes set up to 0 and 1"""
loss, accuracy, precision, recall = model.evaluate(X_test_scaled, y_test)

print(confusion_matrix(y_test, prediction_classes))
"""Confusion matrix"""
"""
        true positive  | false positive 
        false negatives| true negative

"""

print(f'Accuracy:  {accuracy_score(y_test, prediction_classes):.2f}')
print(f'Precision: {precision_score(y_test, prediction_classes):.2f}')
print(f'Recall:    {recall_score(y_test, prediction_classes):.2f}')
""" Accuracy is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations
    Precision is the ratio of correctly predicted positive observations to the total predicted positive observations
    Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes
"""