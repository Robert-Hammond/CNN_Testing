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
import sys
from tqdm import tqdm
from include import *

# mixed
# chinese_123
# misc
# chinese_456
# sides
# circles

category = 'circles'
classifications = ['1', '2', '3']
total_iterations = 50
iterations = 50
batch_size = 3
test_split = 0.2
learning_rate = 0.001

def measure_model(model, iteration, accuracy_points, confidence_points):
    predictions = model.predict(x=test_batches, verbose=0)
    predicted_classes = np.argmax(predictions, axis=-1)

    num_correct = 0
    confidence = 0
    for i in range(len(predictions)):
        if test_batches.classes[i] == predicted_classes[i]:
            confidence += max(predictions[i])
            num_correct += 1
    accuracy = num_correct / len(predictions)
    confidence = confidence / num_correct
    # confidence = (confidence - 1./len(classifications)) * (len(classifications)/(len(classifications) - 1))
    # record the results
    accuracy_points[iteration].append(accuracy)
    confidence_points[iteration].append(confidence)

def write_results(accuracy_points, confidence_points):
    for i in range(iterations + 1):
        acc_mean = np.average(accuracy_points[i])
        acc_sd = np.std(accuracy_points[i])
        con_mean = np.average(confidence_points[i])
        con_sd = np.std(confidence_points[i])

        results.write(f'\"{i}\",\"{acc_mean}\",\"{acc_sd}\",\"{con_mean}\",\"{con_sd}\"\n')
    

datagen = ImageDataGenerator(rescale=1./255, validation_split=test_split)

data_path = 'data/'+category+'/images'

train_batches = None
test_batches = None

def load_data():
    global train_batches
    global test_batches
    train_batches = datagen.flow_from_directory(directory=data_path,
                            target_size=(28,28),
                            color_mode="grayscale",
                            classes=classifications,
                            batch_size=batch_size,
                            subset='training',
                            shuffle=True)
    test_batches = datagen.flow_from_directory(directory=data_path,
                            target_size=(28,28),
                            color_mode="grayscale",
                            classes=classifications,
                            batch_size=1,
                            subset='validation',
                            shuffle=False)
    assert train_batches.n > 0
    assert test_batches.n > 0
    assert train_batches.num_classes == test_batches.num_classes == 3



accuracy_points = [[] for i in range(iterations + 1)]
confidence_points = [[] for i in range(iterations + 1)]

for x in range(total_iterations):
    sys.stdout = open(os.devnull, "w")
    load_data()
    sys.stdout = sys.__stdout__
    
    # https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist
    model = Sequential([
        Conv2D(24, kernel_size=5, padding='same', activation='relu',
                input_shape=(28, 28, 1)),
        MaxPool2D(),
        Conv2D(48, kernel_size=3, padding='same', activation='relu'),
        MaxPool2D(),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(len(classifications), activation='softmax'),
    ])
    model.summary()
    # print("compiling model")
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                    loss='categorical_crossentropy', metrics=['accuracy'])

    print(f"Trial {x+1} / {total_iterations}")

    for i in tqdm(range(iterations)):
        measure_model(model, i, accuracy_points, confidence_points)
        model.fit(x=train_batches, steps_per_epoch=1, epochs=1, verbose=0, validation_data=test_batches)
        
    measure_model(model, iterations, accuracy_points, confidence_points)

results = open(category + "_results.csv", "w")
write_results(accuracy_points, confidence_points)
results.close()