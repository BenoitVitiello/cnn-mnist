#Imports

import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from data_loader import load_data, preprocess_data
from model import build_model

#Training function

def train():
    #Loading and preprocessing
    (X_train, y_train), (X_test, y_test) = load_data()
    (X_train, y_train), (X_test, y_test) = preprocess_data(X_train, y_train, X_test, y_test)

    #Model building
    model = build_model()

    #Saving the best model
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    os.makedirs(os.path.join(ROOT_DIR, "outputs", "models"), exist_ok = True)

    checkpoint = ModelCheckpoint(
        filepath = os.path.join(ROOT_DIR, "outputs", "models", "best_model.keras"),
        monitor = "val_accuracy",
        save_best_only = True,
        verbose = 1
    )

    early_stopping = EarlyStopping( #Stops training if validation accuracy doesn't improve for 5 consecutive epochs.
        monitor = "val_accuracy", #Prevents overfitting.
        patience= 5,
        verbose = 1,
        restore_best_weights = True
    )

    reduce_lr = ReduceLROnPlateau( #Reduces learning rate if validation loss doesn't improve for 3 consecutive epochs, that helps to fine-tune the model.
        monitor = "val_loss",
        factor = 0.5,
        patience = 3,
        verbose = 1
    )

    #Training
    history = model.fit(
        X_train, y_train,
        epochs = 20,
        batch_size = 64,
        validation_split = 0.1,
        callbacks = [checkpoint, early_stopping, reduce_lr]
    )

    return model, history


if __name__ == "__main__":
    model, history = train()