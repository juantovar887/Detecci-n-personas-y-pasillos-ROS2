import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, pathlib, PIL

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns

from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers import Flatten, Dense
from keras.models import Model

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import NumeralTickFormatter
from bokeh.layouts import row

from keras.applications.xception import Xception
import cv2
import random

# Initialize Bokeh output
output_file("/home/tombae/training_results.html")

# Updated constants
IMAGE_SIZE = [400, 400]
BATCH_SIZE = 128
NUM_EPOCHS = 36
CLASSES = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9']  # Update with your new class names
NUM_CLASSES = len(CLASSES)

# Paths
train_dir = '/home/tombae/model/percepcion/classes'
test_dir = '/home/tombae/model/percepcion/test'
val_dir = '/home/tombae/model/percepcion/val'

def print_directory_structure(directory):
    data_dir = pathlib.Path(directory)
    folder = list(data_dir.glob('*'))
    images = list(data_dir.glob('*/*.jpg'))
    print('Folder Structure:')
    for f in folder:
        print(f)
    print('\nNumber of images: ', len(images))

print_directory_structure(train_dir)
print_directory_structure(test_dir)
print_directory_structure(val_dir)

# Data Augmentation and Loading
def get_data_generators(train_dir, val_dir, test_dir, image_size, batch_size, classes):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        validation_split=0.15)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
            directory=train_dir,
            target_size=image_size,
            color_mode='rgb',
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            classes=classes,
            seed=42,
            shuffle=False)

    valid_gen = train_datagen.flow_from_directory(
            directory=val_dir,
            target_size=image_size,
            color_mode='rgb',
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            classes=classes,
            seed=42,
            shuffle=False)

    test_gen = test_datagen.flow_from_directory(
            directory=test_dir,
            target_size=image_size,
            color_mode='rgb',
            batch_size=batch_size,
            class_mode='categorical',
            classes=classes,
            seed=42,
            shuffle=False)

    return train_gen, valid_gen, test_gen

train_gen, valid_gen, test_gen = get_data_generators(train_dir, val_dir, test_dir, IMAGE_SIZE, BATCH_SIZE, CLASSES)

# Model Creation and Compilation
def create_model(image_size, num_classes):
    base_model = Xception(input_shape=image_size + [3], weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    prediction = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=prediction)
    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model

model = create_model(IMAGE_SIZE, NUM_CLASSES)
model.summary()

# Model Training
def train_model(model, train_gen, valid_gen, num_epochs):
    checkpointer = ModelCheckpoint(filepath='/home/tombae/model/percepcion/model.weights.best.keras', verbose=1, save_best_only=True)
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=num_epochs,
        steps_per_epoch=len(train_gen),
        validation_steps=len(valid_gen),
        callbacks=[checkpointer])
    return history

history = train_model(model, train_gen, valid_gen, NUM_EPOCHS)

predicted_classes = model.predict(test_gen)
predicted_classes = np.argmax(predicted_classes,axis=1)
print(predicted_classes)

true_labels = test_gen.classes
print(true_labels[:])

Clases = train_gen.class_indices
fig, ax = plt.subplots(figsize=(30,15),ncols=1,nrows=1)
fig.tight_layout(pad=22.0)
clf_report = classification_report(true_labels, predicted_classes, digits = 4, target_names=[*Clases], output_dict=True, zero_division=0)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True,ax=ax, cbar=False, cmap='YlOrRd',fmt=".1%", cbar_kws={"ticks":[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]},
            vmin = 0, vmax=1)
plt.show()

# Model Evaluation
def evaluate_model(model_path, test_gen):
    model = load_model(model_path)
    test_eval = model.evaluate(test_gen)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])
    return test_eval

evaluate_model('/home/tombae/model/percepcion/model.weights.best.keras', test_gen)
