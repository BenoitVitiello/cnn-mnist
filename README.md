# cnn-mnist

A Convolutional Neural Network (CNN) built with TensorFlow to classify handwritten digits from the MNIST dataset. This project covers the full machine learning pipeline : data exploration, model architecture, training with callbacks, and evaluation with multiple metrics.


## Results
 
- Accuracy ~99.3%
- Precisio ~99.4%
- Recall ~99.4%
- F1 Score ~99.3%
- AUC ~99.9%



## Project Structure

```
cnn-mnist/
│
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                    # Main entry point — runs the full pipeline
│
├── notebooks/
│   ├── 01_exploration.ipynb   # Dataset exploration and visualization
│   ├── 02_training.ipynb      # Model training and loss/accuracy curves
│   └── 03_results.ipynb       # Evaluation, confusion matrix, error analysis
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── model.py               # CNN architecture
│   ├── train.py               # Training loop with callbacks
│   └── evaluate.py            # Evaluation metrics and visualizations
│
└── outputs/
    ├── models/                # Saved model checkpoints (not tracked by git)
    └── figures/               # Generated plots (not tracked by git)
```

## Model Architecture

The CNN uses two convolutional blocks followed by a fully connected classifier.

```
Input (28x28x1)
    │
    ├── Conv2D(32, 3x3, relu)
    ├── BatchNormalization
    ├── MaxPooling2D(2x2)
    ├── Dropout(0.25)
    │
    ├── Conv2D(64, 3x3, relu)
    ├── BatchNormalization
    ├── MaxPooling2D(2x2)
    ├── Dropout(0.25)
    │
    ├── Flatten
    ├── Dense(128, relu)
    ├── Dropout(0.5)
    │
    └── Dense(10, softmax)
```


**Training configuration :**
- Optimizer : Adam
- Loss : Categorical Crossentropy
- Epochs : 20 (with Early Stopping, though it usually goes up to 20.)
- Batch size : 64
- Validation split : 10%

**Techniques :**
ModelCheckpoint : saves the best model based on validation accuracy
EarlyStopping : stops training after 5 epochs without improvement
ReduceLROnPlateau : halves the learning rate after 3 epochs without improvement


## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cnn-mnist.git
cd cnn-mnist
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline

```bash
python main.py
```

This will train the model, evaluate it on the test set, and save all outputs to the `outputs/` folder.

## Requirements

- Python 3.11+
- TensorFlow 2.20+
- See `requirements.txt` for the full list

---

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) — 70,000 grayscale images of handwritten digits (0–9), split into 60,000 training and 10,000 test samples. Each image is 28x28 pixels. The dataset is automatically downloaded by Keras on first run.
