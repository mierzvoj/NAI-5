import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

y_train = y_train.reshape(-1, )

y_test = y_test.reshape(-1, )

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def plot_sample(X, y, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


plot_sample(X_train, y_train, 0)

plot_sample(X_train, y_train, 1)

X_train = X_train / 255.0
X_test = X_test / 255.0

ann = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')
])

ann.compile(optimizer='SGD',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)

y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

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

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=10)

cnn.evaluate(X_test, y_test)

y_pred = cnn.predict(X_test)

y_classes = [np.argmax(element) for element in y_pred]

plot_sample(X_test, y_test, 3)

classesIdx = range(0, 10)
classLabels = dict(zip(classesIdx, classes))
batch = X_test[1000: 1009]
labels = np.argmax(y_test[1000: 1009], axis=(-1))

yPredictions = ann.predict(batch)

print("-----------------------------------------------------------------------")
print("The predictions arrays are:-\n", yPredictions)
print("-----------------------------------------------------------------------")
yPredictions = ann.predict(X_test)
yPredictedClasses = [np.argmax(probability) for probability in yPredictions]


def plotImage(x, y, index):
    plt.figure(figsize=(15, 1.5))
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])


# Function to check if the prediction was correct
def ifPredictionCorrect(index):
    # Plotting the image first
    plotImage(X_test, y_test, index)

    print("Actual class of the object:", classes[y_test[index]])
    print("Predicted class of the object:", classes[yPredictedClasses[index]])

    if classes[yPredictedClasses[index]] == classes[y_test[index]]:
        print("Correct Prediction!")
    else:
        print("Incorrect Prediction!")

    print("")


# end function ifPredictionCorrect()
print("-----------------------------------------------------------------------")
ifPredictionCorrect(2988)
ifPredictionCorrect(2989)
ifPredictionCorrect(2990)
ifPredictionCorrect(2991)
ifPredictionCorrect(9536)
print("-----------------------------------------------------------------------")
