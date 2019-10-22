import argparse

from keras.models import Sequential
#Import from keras_preprocessing not from keras.preprocessing
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import regularizers, optimizers

from Visualize import draw_stats, show_image_data_sample

import pandas as pd
import numpy as np
import pickle
import sys

# Command line parsing
parser = argparse.ArgumentParser()
parser.add_argument("epochs", nargs="?", type=int, default=1, metavar="<epochs>")
parser.add_argument("initial_epochs", nargs="?", type=int, default=1, metavar="<initial epochs>")
parser.add_argument("-V", "--visual", help="Shows visuals on screen", action="store_true")
args = parser.parse_args()

epochs = args.epochs
initial_epochs = args.initial_epochs

print("Initial Epochs: ", initial_epochs, ", Epochs: ", epochs);

traindf = pd.read_csv("./postimerkit2014.csv", dtype=str, sep=';')

datagen = ImageDataGenerator(rescale=1./255.)  # how about data augmentation

train_generator = datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="./images/",
    x_col="id",
    y_col="label",
    seed=42,
    batch_size=40,  # so full set  is dividable with this 2320/40 = 58
    shuffle=True,
    class_mode="categorical",
    target_size=(256,256))  # small resize smaller from 260*260

if args.visual:
    show_image_data_sample(train_generator)

(X_batch, Y_batch) = train_generator.next()   # just to get size of training set
classes= Y_batch.shape[1]

#save labels for later usage, mappings back to info
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
pickle_out = open("Models/Mappings.pickle","wb")
pickle.dump(labels, pickle_out)
pickle_out.close()

# build the model, not so deep needed !

initializer='glorot_uniform'  # probably the best and default
activation='relu' # try other activations too like 'tanh'

model = Sequential()
model.add(Conv2D(12, (2, 2), padding='same', kernel_initializer=initializer, activation='relu',
                 input_shape=(256,256,3)))
model.add(Conv2D(16, (3, 3), kernel_initializer=initializer, activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(20, (3, 3), kernel_initializer=initializer))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(401, activation='relu'))
model.add(Dense(classes, activation='softmax', kernel_initializer=initializer))
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

# real training here, not modified images so those start to converge fast
history_callback = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=initial_epochs  # initial training
)

model.save("./Models/postim_weights.h5")

if args.visual:
    draw_stats(history_callback)

# second set with modified images
datagen=ImageDataGenerator(rescale=1./255.,
    rotation_range=3,   # think those, as augmented features ..
    width_shift_range=0.03,
    height_shift_range=0.03
)  # how about data augmentation)  # how about data augmentation

train_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="./images/",
    x_col="id",
    y_col="label",
    seed=42,
    batch_size=40,  # so full set  is dividable with this 2320/40 = 58
    shuffle=True,
    class_mode="categorical",
    target_size=(256,256))  # small resize smaller from 260*260

if args.visual:
    show_image_data_sample(train_generator)

history_callback = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=epochs
)

model.save("./Models/postim_weights.h5")

if args.visual:
    draw_stats(history_callback)
