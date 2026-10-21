import torch
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import glob
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

import random

#def r2_score(y_true, y_pred):
#    """Computes R² (coefficient of determination)."""
#    y_true = y_true.cpu()
#    y_pred = y_pred.cpu()
#    ss_res = ((y_true - y_pred) ** 2).sum()
#    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
#    return 1 - ss_res / ss_tot

class ProteinDataset(Dataset):
    def __init__(self, pt_files):
        self.embeddings = []
        self.temperatures = []

        if isinstance(pt_files, str):
            pt_files = [pt_files]

        for file in pt_files:
            print(f"Loading {file}...")
            data = torch.load(file)
            self.embeddings.append(data['embeddings'])         # shape: (N, D)
            self.temperatures.append(data['temperatures'])     # shape: (N,)

        self.embeddings = torch.cat(self.embeddings, dim=0)    # shape: (Total_N, D)
        self.temperatures = torch.cat(self.temperatures, dim=0)# shape: (Total_N,)
        print(f"Total records: {len(self.embeddings)}")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.temperatures[idx]


class TemperaturePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # Output is scalar (temperature)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)  # Squeeze for shape (batch,) instead of (batch, 1)

def load_model_from_checkpoint(model_class, checkpoint_path, device, **model_kwargs):
    """
    Load a PyTorch model from a checkpoint file.
    
    Args:
        model_class: Class of the model (not an instance).
        checkpoint_path: Path to the .pth file.
        device: 'cpu' or 'cuda'.
        model_kwargs: Any arguments needed to instantiate the model.

    Returns:
        model: The loaded PyTorch model in eval mode on the given device.
    """
    # Initialize a new instance of the model
    model = model_class(**model_kwargs).to("cuda")

    # Load the state dict
    state_dict = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(state_dict)

    model.eval()  # Set to eval mode by default
    return model

def evaluate_model_on_bootstrap(model, dataset, device, n_bootstrap=100, batch_size=64):
    model.eval()
    metrics = []
    all_bootstrap_preds = []
    all_bootstrap_targets = []

    for i in range(n_bootstrap):
        # Sample indices with replacement
        indices = np.random.choice(len(dataset), len(dataset), replace=True)
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

        preds, targets = [], []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds.append(outputs.cpu().numpy())
                targets.append(labels.cpu().numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

        all_bootstrap_preds.append(preds)
        all_bootstrap_targets.append(targets)

        mse = mean_squared_error(targets, preds)
        r2 = r2_score(targets, preds)
        metrics.append((mse, r2))

        print(f"Bootstrap {i+1}/{n_bootstrap}: MSE = {mse:.4f}, R² = {r2:.4f}")

    return {
        "metrics": metrics,
        "predictions": all_bootstrap_preds,
        "targets": all_bootstrap_targets
    }

def evaluate_model_on_bootstrap(model, dataset, device, n_bootstrap=100, batch_size=64):
    model.eval()
    metrics = []
    all_bootstrap_preds = []
    all_bootstrap_targets = []

    for i in range(n_bootstrap):
        # Sample indices with replacement
        indices = np.random.choice(len(dataset), len(dataset), replace=True)
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

        preds, targets = [], []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds.append(outputs.cpu().numpy())
                targets.append(labels.cpu().numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

        all_bootstrap_preds.append(preds)
        all_bootstrap_targets.append(targets)

        mse = mean_squared_error(targets, preds)
        r2 = r2_score(targets, preds)
        metrics.append((mse, r2))

        print(f"Bootstrap {i+1}/{n_bootstrap}: MSE = {mse:.4f}, R² = {r2:.4f}")

    return {
        "metrics": metrics,
        "predictions": all_bootstrap_preds,
        "targets": all_bootstrap_targets
    }

def main():

    model_dir = "/ThermalGAN/weights/Classifier/OGT_IMG_DATA_EMBEDDINGS"
    # Load dataset
    # All .pt files in the directory (you can filter/sort if needed)
    #train_files = sorted(glob.glob("/ThermalGAN/data/embeddings/train/*.pt"))
    val_files   = sorted(glob.glob("/ThermalGAN/data/embeddings/val/*.pt"))
    test_files  = sorted(glob.glob("/ThermalGAN/data/embeddings/test/*.pt"))
    
    #train_dataset = ProteinDataset(train_files)
    val_dataset = ProteinDataset(val_files)
    test_dataset = ProteinDataset(test_files)
    
    
    
    # load model
    input_dim = val_dataset.embeddings.shape[1]
    model_kwargs = {"input_dim": input_dim}
    model = load_model_from_checkpoint(model_class = TemperaturePredictor, checkpoint_path = os.path.join(model_dir, "best_model.pth"), device = "cuda", **model_kwargs)
    
    evaluations_val  = evaluate_model_on_bootstrap(model=model, dataset=val_dataset, device="cuda", n_bootstrap=100, batch_size=256)
    evaluations_test = evaluate_model_on_bootstrap(model=model, dataset=test_dataset, device="cuda", n_bootstrap=100, batch_size=256)

    ## Save evaluations
    pd.DataFrame(evaluations_val).to_csv(os.path.join(model_dir, "evaluation_val.csv"))
    pd.DataFrame(evaluations_test).to_csv(os.path.join(model_dir, "evaluation_test.csv"))

if __name__ == "__main__":
    main()





