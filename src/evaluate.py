# Imports
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.models import load_model
from data_loader import load_data, preprocess_data

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate(model):
    #Loading and preprocessing
    (X_train, y_train), (X_test, y_test) = load_data()
    (X_train, y_train), (X_test, y_test) = preprocess_data(X_train, y_train, X_test, y_test)

    #Evaluating on the test set
    results = model.evaluate(X_test, y_test, verbose = 1)
    test_loss, test_accuracy, test_precision, test_recall, test_auc = results

    print(f"Test loss : {test_loss:.4f}")
    print(f"Test accuracy : {test_accuracy:.4f}")
    print(f"Test precision : {test_precision:.4f}")
    print(f"Test recall : {test_recall:.4f}")
    print(f"Test AUC : {test_auc:.4f}")

    #Predictions
    y_pred     = model.predict(X_test)
    y_pred_cls = np.argmax(y_pred, axis = 1)
    y_true_cls = np.argmax(y_test, axis = 1)

    #F1 scores
    f1_macro    = f1_score(y_true_cls, y_pred_cls, average = "macro")
    f1_weighted = f1_score(y_true_cls, y_pred_cls, average = "weighted")
    print(f"F1 Score (macro) : {f1_macro:.4f}")
    print(f"F1 Score (weighted) : {f1_weighted:.4f}")

    #Classification report
    print("\nClassification Report :")
    print(classification_report(y_true_cls, y_pred_cls))

    #Confusion matrix
    plot_confusion_matrix(y_true_cls, y_pred_cls)

    return y_pred, y_pred_cls, y_true_cls


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize = (10, 8))
    sns.heatmap(
        cm,
        annot = True,
        fmt = "d",
        cmap = "Blues",
        xticklabels = range(10),
        yticklabels = range(10)
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    os.makedirs(os.path.join(ROOT_DIR, "outputs", "figures"), exist_ok = True)
    plt.savefig(os.path.join(ROOT_DIR, "outputs", "figures", "confusion_matrix.png"), dpi = 150)
    plt.show()


def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize = (14, 10))

    #Accuracy
    axes[0, 0].plot(history.history["accuracy"], label = "Train")
    axes[0, 0].plot(history.history["val_accuracy"], label = "Validation")
    axes[0, 0].set_title("Accuracy over epochs")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()

    #Loss
    axes[0, 1].plot(history.history["loss"], label = "Train")
    axes[0, 1].plot(history.history["val_loss"], label = "Validation")
    axes[0, 1].set_title("Loss over epochs")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()

    #Precision & Recall
    axes[1, 0].plot(history.history["precision"], label = "Train Precision")
    axes[1, 0].plot(history.history["val_precision"], label = "Val Precision")
    axes[1, 0].plot(history.history["recall"], label = "Train Recall")
    axes[1, 0].plot(history.history["val_recall"], label = "Val Recall")
    axes[1, 0].set_title("Precision & Recall over epochs")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend()

    #AUC
    axes[1, 1].plot(history.history["auc"], label = "Train")
    axes[1, 1].plot(history.history["val_auc"], label = "Validation")
    axes[1, 1].set_title("AUC over epochs")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("AUC")
    axes[1, 1].legend()

    plt.tight_layout()
    os.makedirs(os.path.join(ROOT_DIR, "outputs", "figures"), exist_ok = True)
    plt.savefig(os.path.join(ROOT_DIR, "outputs", "figures", "training_history.png"), dpi = 150)
    plt.show()