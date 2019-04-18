import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import os
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical 
from keras.models import load_model

# Set logging level to low to remove deprecated messages
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # or [0, 1, 2] 
tf.logging.set_verbosity(tf.logging.ERROR)

inputData = pd.read_csv('mnist.csv')

X = inputData.drop(inputData.columns[0], axis=1).values
y = to_categorical(inputData[inputData.columns[0]])

model = load_model('MNIST_Model.h5')

# Manual evaluation
imageIndex = 1458
# Print expected result
print(inputData[inputData.columns[0]][imageIndex])
# Draw graphical representation of the predicted number
plt.imshow(X[imageIndex].reshape(28, 28), cmap='Greys')
plt.savefig('figure.png')
plt.show(block=False)
plt.pause(3)

# Predict using a 1 x 784 matrix
pred = model.predict(X[imageIndex].reshape(1, 784))
print("Prediction: " + str(pred.argmax()))

