import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

""" Lukasz Cettler s20168, Wojciech Mierzejewski s21617
    Tensor Flow based animal recognition system 
    Dataset: CIFAR-10 contains 60 000 32x32 colour images in 10 classes, with 6 000 images per class
    Excersise aim is to teach our nn to properly recognize animal pictures
"""

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
"""Loading CIFAR-10 dataset, data is split into train and test sets, independet and dependent variables """

y_train = y_train.reshape(-1, )
"""Numpy array shape change to vector"""
y_test = y_test.reshape(-1, )
"""Numpy array shape change to vector"""

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
"""Defining classes names as an array"""

X_train = X_train / 255.0
X_test = X_test / 255.0
"""Data preprocessing, standarizing input data, 255 RGB scale"""

ann = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')
])
print(ann.output_shape)
"""prints (None, 10)"""
""" Keras (API for TF), sequential model, applied with
    layers.Flatten - matrix is left with one dimension only
    layers.Dense -  layers of 3000 neurons, 1000 neurons, outputs 10 neurons, activation function: relu and softmax
"""
ann.compile(optimizer='SGD',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
""" Model ann compilation, optimizer used to set weights and learning rate to reduce losses
    loss function - minimized quantity to be sought during learning for our model
    metrics - This metric creates two local variables, total and count that are used to compute the frequency with which 
    predicted matches true. 
 """
ann.fit(X_train, y_train, epochs=5)
""" Mode fit - Trains the model for a fixed number of epochs (dataset iterations)."""
y_pred = ann.predict(X_test)
"""Generates output predictions for the input samples."""
y_pred_classes = [np.argmax(element) for element in y_pred]
"""Returns the indices of the maximum values along y_pred"""

print("Classification Report: \n", classification_report(y_test, y_pred_classes))

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
"""Second model convolutional, mainly used for image recognition"""

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
"""Model compilation"""

cnn.fit(X_train, y_train, epochs=10)
""" Mode fit - Trains the model for a fixed number of epochs (dataset iterations)."""
cnn.evaluate(X_test, y_test)
"""This model.evaluate to evaluate the model and it will output the loss and the accuracy."""

y_pred = cnn.predict(X_test)
"""Generates output predictions for the input samples."""
y_classes = [np.argmax(element) for element in y_pred]
"""Returns the indices of the maximum values along y_pred"""

classesIdx = range(0, 10)
classLabels = dict(zip(classesIdx, classes))
batch = X_test[1000: 1009]
labels = np.argmax(y_test[1000: 1009], axis=(-1))
yPredictions = ann.predict(X_test)
yPredictedClasses = [np.argmax(probability) for probability in yPredictions]
"""Matching predicted items against class array"""

def ifPredictionCorrect(index):
    """Function to determine right or wrong recognition"""
    print("Actual class of the object:", classes[y_test[index]])
    print("Predicted class of the object:", classes[yPredictedClasses[index]])

    if classes[yPredictedClasses[index]] == classes[y_test[index]]:
        print("Correct Prediction!")
    else:
        print("Incorrect Prediction!")

    print("")


print("-----------------------------------------------------------------------")
ifPredictionCorrect(2988)
ifPredictionCorrect(2989)
ifPredictionCorrect(2990)
ifPredictionCorrect(2991)
ifPredictionCorrect(9536)
print("-----------------------------------------------------------------------")
"""Generating samples for prediction, prediction check and printing output"""