from IPython import get_ipython
from keras.datasets import fashion_mnist
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

print('Training Data Shape :', train_X.shape, train_Y.shape)
print('Test Data Sample: ', test_X.shape, test_Y.shape)

classes = np.unique(train_Y)
nclasses = len(classes)
print("Number of classes=", nclasses)
print("Classes are ", classes)

train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)
train_X.shape, test_X.shape

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

train_X = train_X / 255
test_X = test_X / 255

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)
print("Categorical data", train_Y[0])
print("After one_coding", train_Y_one_hot[0])

train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
train_X.shape, valid_X.shape, train_label.shape, valid_label.shape

batch_size = 64
num_classes = 10
epochs = 3

fashion_model = Sequential()

fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28, 28, 1), padding='same'))
# fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2), padding='same'))
fashion_model.add(Dropout(0.25))

fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
# fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
fashion_model.add(Dropout(0.25))

fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
# fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
fashion_model.add(Dropout(0.25))

fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
# fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dropout(0.25))

fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

fashion_model.summary()

fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1,
                                  validation_data=(valid_X, valid_label))

test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)

print("Test Loss", test_eval[0])
print("Test_Accuracy", test_eval[1])

# --------------------------------------------------#
classesIdx = range(0, 10)
classLabels = dict(zip(classesIdx, classes))
batch = test_X[1000: 1009]
labels = np.argmax(test_Y[1000: 1009], axis=(-1))
yPredictions = fashion_model.predict(batch)
yPredictions = fashion_model.predict(test_X)
yPredictedClasses = [np.argmax(probability) for probability in yPredictions]


def ifPredictionCorrect(index):
    print("Actual class of the object:", classes[test_Y[index]])
    print("Predicted class of the object:", classes[yPredictedClasses[index]])

    if classes[yPredictedClasses[index]] == classes[test_Y[index]]:
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
