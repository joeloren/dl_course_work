import numpy as np
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Conv2D, Dense, Dropout, Flatten, InputLayer, MaxPool2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from keras.initializers import Constant
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

import keras.backend as K
from keras import regularizers
import keras
from sklearn.model_selection import train_test_split


def normalize(X_train, X_test):
    """
    This function normalize inputs for zero mean and unit variance
    it is used when training a model.
    Input: training set and test set
    Output: normalized training set and test set according to the
    training set statistics.
    """
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test


def create_datagen(X_train, X_test):

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(X_train)

    return datagen


def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate


def build_cifar_10_model(weight_decay=1e-4):
    model = Sequential()

    #    model.add(MaxPooling2D(pool_size=(2,2), padding="valid"))
    model.add(Conv2D(64, (2, 2), padding='valid',
                     input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))

    model.add(BatchNormalization())
    model.add(
        Conv2D(32, (2, 2), padding='same', input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
    model.add(Dropout(0.3))

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(
        Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3, 3), padding='valid',
                     input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(
        Conv2D(32, (3, 3), padding='valid', input_shape=(32, 32, 3), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
    model.add(Dropout(0.5))

    model.add(Flatten())
    #     model.add(Dense(128,kernel_regularizer=regularizers.l2(weight_decay)))
    #     model.add(Activation('relu'))
    #     model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    num_classes = 10
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    X_train = K.cast_to_floatx(X_train) / 255
    X_test = K.cast_to_floatx(X_test) / 255

    X_train, X_test = normalize(X_train, X_test)

    cp_cb = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5',
                            monitor='val_acc',
                            save_best_only=False,
                            save_weights_only=True,
                            period=5)

    model = build_cifar_10_model(weight_decay=1e-4)
    model.summary()

    batch_size = 32
    sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
    rmsprop = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
    adam = keras.optimizers.adam(lr=.001, decay=1e-6)
    num_epochs = 250

    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

    datagen = create_datagen(X_train, X_test)
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=X_train.shape[0] // batch_size, epochs=num_epochs,
                        validation_data=(X_test, y_test), callbacks=[LearningRateScheduler(lr_schedule), cp_cb])
