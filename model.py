import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import argparse
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

import keras.layers as kl
from keras.layers import  LSTM, Bidirectional

from keras.layers import Input
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D, Dense, MaxPooling2D, Dropout, InputLayer
from keras import Model
from keras.optimizers import RMSprop
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping, TensorBoard
from input_pipeline import get_dataset, img_width, img_height

class_names = np.array(
    ['Boredom', 'Engagement', 'Confusion', 'Frustration']
)

img_width = 299
img_height = 299


numpy_dir = "numpyoutput"

def get_model():
    """
    Creates custom model for engagement
    recognition.

    Returns:
        model for the training.
    """
    
    inputs = Input(shape=(299,299,3))
    
    # Define the Model Architecture.
    ########################################################################################################################
   
    x = Conv2D(16, (3, 3), padding='same',activation = 'relu')(inputs)
    x = (MaxPooling2D((4, 4)))(x)

    
    x = (Conv2D(32, (3, 3), padding='same',activation = 'relu'))(x)
    x = (MaxPooling2D((4, 4)))(x)

    
    x = (Conv2D(64, (3, 3), padding='same',activation = 'relu'))(x)
    x = (MaxPooling2D((2, 2)))(x)

    
    x = (Conv2D(64, (3, 3), padding='same',activation = 'relu'))(x)
    x = (MaxPooling2D((2, 2)))(x)
    
                                      
    x = (Flatten())(x)
                                      
    
                                 
    boredom = Dense(4, name="y1")(x)
    engagement = Dense(4, name="y2")(x)
    confusion = Dense(4, name="y3")(x)
    frustration = Dense(4, name="y4")(x)
    model = Model(inputs,outputs=[boredom, engagement, confusion, frustration])

    # Return the constructed CNN model.
    return model

train_ds = get_dataset("Train")
validation_ds = get_dataset("Validation")
test_ds = get_dataset("Test")

RCNN_model = get_model()
print("Model Created Successfully!")

RCNN_model.compile(optimizer=RMSprop(learning_rate=0.0001),
                  loss={"y1": SparseCategoricalCrossentropy(from_logits=True),
                        "y2": SparseCategoricalCrossentropy(from_logits=True),
                        "y3": SparseCategoricalCrossentropy(from_logits=True),
                        "y4": SparseCategoricalCrossentropy(from_logits=True)},
                  metrics={"y1": "sparse_categorical_accuracy",
                           "y2": "sparse_categorical_accuracy",
                           "y3": "sparse_categorical_accuracy",
                           "y4": "sparse_categorical_accuracy"})
print(RCNN_model.summary())

callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=1e-2,
                      patience=2, verbose=1),
        TensorBoard(log_dir=str("logdir"))
    ]


history = RCNN_model.fit(train_ds,
                        epochs=10,
                        callbacks=callbacks,
                        validation_data=validation_ds)

RCNN_model.save("modelpath")