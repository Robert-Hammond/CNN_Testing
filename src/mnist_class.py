from mlxtend.data import loadlocal_mnist
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import shutil
import random
import glob
import warnings
import os
from hilbert_flatten import *

train_path = 'data/mnist/train'
test_path = 'data/mnist/test'
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# ONLY RUN THESE IF HAVENT ALREADY CREATED IMAGES
if not os.path.isdir(train_path):
    train_image_data, train_labels = loadlocal_mnist(
        images_path='data/mnist/train-images.idx3-ubyte',
        labels_path='data/mnist/train-labels.idx1-ubyte')

    assert len(train_image_data) == 60000
    assert len(train_image_data[0]) == 784

    os.chdir('data/mnist')
    os.makedirs('train')
    for i in range(10):
        os.makedirs('train/'+str(i))

    for i in range(1000):
        img = Image.new('L', (28, 28))
        img.putdata(train_image_data[i])
        img.save('train/'+str(train_labels[i]) +
                 '/'+str(train_labels[i])+'.'+str(i)+'.jpg')

    os.chdir('../..')

if not os.path.isdir(test_path):
    test_image_data, test_labels = loadlocal_mnist(
        images_path='data/mnist/t10k-images.idx3-ubyte',
        labels_path='data/mnist/t10k-labels.idx1-ubyte')

    assert len(test_image_data) == 10000
    assert len(test_image_data[0]) == 784

    os.chdir('data/mnist')
    os.makedirs('test')
    for i in range(10):
        os.makedirs('test/'+str(i))

    for i in range(100):
        img = Image.new('L', (28, 28))
        img.putdata(test_image_data[i])
        img.save('test/'+str(test_labels[i])+'/' +
                 str(test_labels[i])+'.'+str(i)+'.jpg')

    os.chdir('../..')


datagen = ImageDataGenerator(rescale=1./255)

train_batches = datagen.flow_from_directory(directory=train_path,
                                            target_size=(28, 28),
                                            color_mode="grayscale",
                                            classes=classes,
                                            batch_size=10)
test_batches = datagen.flow_from_directory(directory=test_path,
                                           target_size=(28, 28),
                                           color_mode="grayscale",
                                           classes=classes,
                                           batch_size=10,
                                           shuffle=False)

assert train_batches.n > 0
assert test_batches.n > 0
assert train_batches.num_classes == test_batches.num_classes == 10

if not os.path.isfile("models/mnist.h5") or True:
    # https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist
    model = Sequential([
        Conv2D(24, kernel_size=5, padding='same', activation='relu',
               input_shape=(28, 28, 1)),
        MaxPool2D(),
        Conv2D(48, kernel_size=3, padding='same', activation='relu'),
        MaxPool2D(),
        HilbertFlatten(),
        Dense(64, activation='relu'),
        Dense(train_batches.num_classes, activation='softmax'),
    ])

    model.summary()
    print("compiling model")
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    print("training model")
    model.fit(x=train_batches, validation_data=test_batches,
              epochs=10, verbose=2)
    model.save("models/mnist.h5")
else:
    print("loading model")
    model = load_model("models/mnist.h5")
