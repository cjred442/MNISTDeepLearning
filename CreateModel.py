import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical 
from keras.callbacks import EarlyStopping

inputData = pd.read_csv('mnist.csv')

X = inputData.drop(inputData.columns[0], axis=1).values
y = to_categorical(inputData[inputData.columns[0]])

# Initiate Early Stopping Monitor
earlyStoppingMonitor = EarlyStopping(patience=2)

# Create the model: model
model = Sequential()

# Add hidden layers
model.add(Dense(125, activation='relu', input_shape = (784,)))
model.add(Dense(125, activation='relu')) 
model.add(Dense(125, activation='relu')) 
model.add(Dense(125, activation='relu'))
model.add(Dense(125, activation='relu'))
model.add(Dense(125, activation='relu'))

# Add the output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X, y, validation_split=0.3, epochs=15, callbacks = [earlyStoppingMonitor])

# Save model (commented out model with 100% accuracy achieved)
# model.save('MNIST_Model.h5')
