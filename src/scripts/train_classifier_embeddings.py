import torch
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import glob
import pandas as pd
import argparse 
import os
import h5py
parser = argparse.ArgumentParser()

parser.add_argument("-o", "--output", type=str, default="./",
                   help="Output directory")
parser.add_argument("--max_epochs", type=int, default=1000)

def r2_score(y_true, y_pred):
    """Computes R² (coefficient of determination)."""
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0, mode='min', verbose=True):
        """
        Args:
            patience (int): How long to wait after last improvement.
            delta (float): Minimum change to qualify as an improvement.
            mode (str): 'min' for loss, 'max' for score (e.g., R²).
            verbose (bool): Print messages when stopping.
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.early_stop = False

        self.compare = (
            (lambda new, best: new < best - delta)
            if mode == 'min'
            else (lambda new, best: new > best + delta)
        )

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

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

class ProteinDataset_stream(Dataset):
    def __init__(self, pt_files):
        if isinstance(pt_files, str):
            pt_files = [pt_files]

        self.pt_files = pt_files
        self.index_map = []  # List of tuples: (file_index, local_index)
        self._last_file = None
        self._last_data = None
        # Preprocess: build index mapping from global idx to (file_idx, local_idx)

        self.file_sample_counts = []
        for i, file in enumerate(pt_files):
            print(f"Indexing {file}...")
            data = torch.load(file, map_location='cpu')
            num_samples = data['embeddings'].shape[0]
            self.file_sample_counts.append(num_samples)
            self.index_map.extend([(i, j) for j in range(num_samples)])

        print(f"Total records: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, local_idx = self.index_map[idx]
        file_path = self.pt_files[file_idx]
        
        if self._last_file != file_path:
            self._last_data = torch.load(file_path, map_location='cpu')
            self._last_file = file_path
        
        data = self._last_data
        return data['embeddings'][local_idx], data['temperatures'][local_idx]
class HDF5ProteinDataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        self._h5 = None  # will be opened lazily in __getitem__

        # Load metadata (length only) without keeping file open
        with h5py.File(self.hdf5_file, 'r') as f:
            self.length = f['embeddings'].shape[0]
            self.input_dim = f["embeddings"].shape[1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open file if not yet opened (important for DataLoader workers)
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_file, 'r')

        embedding = self._h5['embeddings'][idx]
        temperature = self._h5['temperatures'][idx]

        return torch.from_numpy(embedding), torch.tensor(temperature) #torch.tensor(embedding), torch.tensor(temperature)

    def __del__(self):
        # Cleanly close the file when the object is destroyed
        if self._h5 is not None:
            self._h5.close()

class TemperaturePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)  # Output is scalar (temperature)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)  # Squeeze for shape (batch,) instead of (batch, 1)

def main(args):
    # Load dataset
    # All .pt files in the directory (you can filter/sort if needed)
    train_files = sorted(glob.glob("/ThermalGAN/data/embeddings/train/*.pt"))
    val_files   = sorted(glob.glob("/ThermalGAN/data/embeddings/val/*.pt"))
    test_files  = sorted(glob.glob("/ThermalGAN/data/embeddings/test/*.pt"))
    
#    train_dataset = ProteinDataset(train_files)
#    val_dataset = ProteinDataset(val_files)
#    test_dataset = ProteinDataset(test_files)
    train_dataset = HDF5ProteinDataset("/ThermalGAN/data/HDF5/train.hdf5")
    val_dataset = HDF5ProteinDataset("/ThermalGAN/data/HDF5/val.hdf5")
    test_dataset = HDF5ProteinDataset("/ThermalGAN/data/HDF5/test.hdf5")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8192,shuffle=True, pin_memory=True, num_workers=16,prefetch_factor=20)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8192, pin_memory=True, num_workers=16,prefetch_factor=20)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8192, pin_memory=True, num_workers=16,prefetch_factor=20)
    
    
    # Initialize model
    print(f"Input dim is: {train_dataset.input_dim}")
    input_dim = train_dataset.input_dim
    model = TemperaturePredictor(input_dim).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    early_stopper = EarlyStopping(patience=50, mode='min')
    
    # Training loop
    device = next(model.parameters()).device
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output,"best_model.pth")
    
    history = []
    
    epochs = args.max_epochs
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device) 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item() * inputs.size(0)

        print("Model is on:", next(model.parameters()).device)
        print("Inputs are on:", inputs.device)
        print("Targets are on:", targets.device)
        avg_train_loss = running_loss / len(train_dataset)
    
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
    
                all_preds.append(outputs)
                all_targets.append(targets)
    
        avg_val_loss = val_loss / len(val_dataset)
    
        scheduler.step(avg_val_loss)
        early_stopper.step(avg_val_loss)  # or r2 if using mode='max'
    
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
    
        # Checkpoint if this is the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with val_loss = {avg_val_loss:.4f}")
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        r2 = r2_score(all_targets, all_preds)
    
        
    
        test_loss = 0.0
        all_preds = []
        all_targets = []
    
        # Secondary Validation
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
    
                all_preds.append(outputs)
                all_targets.append(targets)
    
        avg_test_loss = test_loss / len(test_dataset)
        
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        r2_test = r2_score(all_targets, all_preds)
    
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, R² = {r2:.4f}, Test Loss = {avg_test_loss:.4f}, R² = {r2_test:.4f}")
        
        history.append({
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'r2': r2.item() if isinstance(r2, torch.Tensor) else r2,
        'test_loss': avg_test_loss,
        'r2_test': r2_test.item() if isinstance(r2_test, torch.Tensor) else r2_test})
    
    df_history = pd.DataFrame(history)
    df_history.to_csv(os.path.join(args.output,"history.csv"), index=False)
    print(f"Saved training history to {os.path.join(args.output,'history.csv')}")

if __name__ == "__main__":
    args = parser.parse_args()

    print("CUDA available:", torch.cuda.is_available())
    print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

    if not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)
    main(args)

