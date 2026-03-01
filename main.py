import sys 
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_data, preprocess_data
from model import build_model
from train import train
from evaluate import evaluate, plot_training_history

def main():
    #Ouput directories
    os.makedirs("outputs/models",  exist_ok = True)
    os.makedirs("outputs/figures", exist_ok = True)

    #Train
    print("Model training")
    model, history = train()

    # --- Evaluate ---
    print("Model evaluating")
    evaluate(model)
    plot_training_history(history)

if __name__ == "__main__":
    main()