#Imports

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

#Data loading and preprocessing

def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return (X_train, y_train), (X_test, y_test)

def preprocess_data(X_train, y_train, X_test, y_test):
    #Reshape (60000, 28, 28) -> (60000, 28, 28, 1)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    #Normalization [0, 255] -> [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    #One-hot encoding
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    return (X_train, y_train), (X_test, y_test)
