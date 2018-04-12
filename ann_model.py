from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# print(X_train[0].shape)
# print(y_train.shape)
# print(y_train)

#plotting mnist
# plt.subplot(221)
# plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
# plt.subplot(222)
# plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
# plt.subplot(223)
# plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
# plt.subplot(224)
# plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# plt.show()

# setting the seed
seed = 7
numpy.random.seed(seed)

# flatteing the image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# normalizing the grayscale image (0-255) --> (0-1)
X_train /= 255
X_test /= 255

# OneHotEncoding the classes of train and test data
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(y_train.shape)
num_classes = y_test.shape[1]


# Simple ANN model
def ANN_classifier():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = ANN_classifier()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy {}'.format(scores*100))


