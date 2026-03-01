#Imports

import tensorflow as tf
from tensorflow.keras import layers, models

#Model architecture

def build_model():
    model = models.Sequential([

        #First convolution
        layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (28, 28, 1)),
        layers.BatchNormalization(), #Normalizational layer to stabilize training, allows higher learning rate.
        layers.MaxPooling2D((2, 2)), #Downsamples the feature maps, helps to capture dominant features.
        layers.Dropout(0.25), #Regularization technique to prevent overfitting, randomly sets 25% of the inputs to zero during training.

        #Second convolution
        layers.Conv2D(64, (3, 3), activation = "relu"), #Increases the number of filters to capture more complex features.
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        #Classification head
        layers.Flatten(), #2D -> 1D
        layers.Dense(128, activation = "relu"),
        layers.Dropout(0.5), #The fully connected layer has more parameters and is more prone to overfitting.
        layers.Dense(10, activation = "softmax")
    ])

    model.compile(
        optimizer = "adam",
        loss      = "categorical_crossentropy", #Multi-class classification problem, labels are one-hot encoded.
        metrics   = [
            "accuracy",
            tf.keras.metrics.Precision(name = "precision"),
            tf.keras.metrics.Recall(name = "recall"),
            tf.keras.metrics.AUC(name = "auc")
        ]
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()