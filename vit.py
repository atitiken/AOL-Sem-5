import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, auc, classification_report
)
from sklearn.model_selection import train_test_split, StratifiedKFold
# from imblearn.combine import SMOTEENN # <-- Keep if using SMOTE-ENN
from imblearn.over_sampling import SMOTE # <-- Use SMOTE only for now
# from imblearn.under_sampling import EditedNearestNeighbours # <-- Keep if using SMOTE-ENN
from PIL import Image, UnidentifiedImageError, ImageFile
from tqdm import tqdm
from timm import create_model
import time # Added for unique filenames
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Allow loading of truncated images (Use with caution, better to fix/remove them)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Get the absolute path of the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
# Note: Setting deterministic=True can impact performance. Set to False if speed is critical.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configuration
class Config:
    # Data paths
    train_csv = os.path.join(script_dir, "train_ratio_1in4.csv")
    test_csv = os.path.join(script_dir, "test_ratio_1in4.csv")
    train_dir = os.path.join(script_dir, "dataset_ratio_1in4", "train")
    test_dir = os.path.join(script_dir, "dataset_ratio_1in4", "test")

    # Model parameters
    model_name = "vit_tiny_patch16_224"  # ViT-Ti/16 model
    pretrained = True
    num_classes = 1  # Binary classification (fracture or not)

    # Training parameters
    batch_size = 64 # Adjust based on GPU memory
    num_epochs = 50 # Can be increased if early stopping isn't triggering too early
    learning_rate = 5e-5 # *** ADJUSTED: Lower initial LR ***
    weight_decay = 1e-5
    validation_split = 0.1 # Use 10% of training data for validation (used in final training only)

    # Image preprocessing
    img_size = 224  # ViT input size

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save paths
    model_dir = os.path.join(script_dir, "models", "tuned_v2") # Changed save directory
    checkpoint_dir = os.path.join(model_dir, "checkpoint")
    # Path for the best model found during k-fold (used for final testing)
    best_kfold_model_path = os.path.join(model_dir, "vit_ti_best_kfold_model.pth")
    # Path for the model trained on the full dataset (optional, might not be the best)
    final_trained_model_path = os.path.join(model_dir, "vit_ti_final_trained_model.pth")
    plot_dir = os.path.join(script_dir, "plots", "tuned_v2") # Changed plot directory

    # K-fold cross-validation
    n_folds = 10
    random_state = SEED

    # Class imbalance parameters
    use_smote = True # *** ADDED: Flag to easily enable/disable SMOTE ***
    smote_ratio = 1.0  # Aim for minority class to be 50% of majority class size (Can be tuned: 0.3, 0.7, 1.0)
    use_weighted_loss = True # *** ADDED: Flag to easily enable/disable weighted loss ***

    # Mixed precision
    use_amp = True

    # SWA parameters
    use_swa = False # *** ADJUSTED: Disabled SWA for now to isolate other issues ***
    swa_start = 10 # Start SWA after this many epochs (if use_swa is True)
    swa_lr = 5e-5 # *** ADJUSTED: Drastically reduced SWA LR (if use_swa is True) ***

    # Learning rate scheduler
    warmup_epochs = 5
    min_lr = 1e-6

    # Early stopping
    patience = 10 # *** ADJUSTED: Increased patience ***
    min_delta = 1e-4 # Minimum change to qualify as an improvement

    # Class weights for loss function (will be calculated dynamically if use_weighted_loss is True)
    pos_weight = None

config = Config()
print(f"Using device: {config.device}")

# Create directories if they don't exist
os.makedirs(config.plot_dir, exist_ok=True)
os.makedirs(config.model_dir, exist_ok=True)
os.makedirs(config.checkpoint_dir, exist_ok=True) # Create checkpoint directory

# Create custom dataset class
class FractureDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, indices=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            indices (list, optional): List of indices to use from the csv file.
        """
        try:
            self.all_data = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_file}")
            raise
        except Exception as e:
            print(f"Error reading CSV file {csv_file}: {e}")
            raise

        if indices is not None:
            # Ensure indices are within the bounds of the loaded data
            valid_indices = [i for i in indices if i < len(self.all_data)]
            if len(valid_indices) != len(indices):
                print(f"Warning: Some provided indices were out of bounds for {csv_file}.")
            self.data = self.all_data.iloc[valid_indices].reset_index(drop=True)
        else:
            self.data = self.all_data

        self.img_dir = img_dir
        self.transform = transform

        if len(self.data) == 0:
            print(f"Warning: No data loaded for dataset from {csv_file} with provided indices.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx >= len(self.data):
             print(f"Warning: Index {idx} out of bounds for dataset size {len(self.data)}.")
             return None # Should not happen with standard samplers but good practice

        try:
            img_name = self.data.iloc[idx, 0]
            img_path = os.path.join(self.img_dir, img_name)

            if not os.path.exists(img_path):
                # print(f"Warning: Image file not found at {img_path}, skipping.")
                return None # Return None if image doesn't exist

            # Open image and convert to RGB
            image = Image.open(img_path).convert('RGB')

            label = self.data.iloc[idx, 1]

            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(label, dtype=torch.float32)

        except FileNotFoundError:
            # This case should be caught by os.path.exists, but included for safety
            # print(f"Warning: Image file not found at {img_path} during loading, skipping.")
            return None
        except UnidentifiedImageError:
            # print(f"Warning: Could not identify image file {img_path}, skipping.")
            return None
        except Image.DecompressionBombError:
            # print(f"Warning: Image file {img_path} is too large (Decompression Bomb), skipping.")
            return None
        except OSError as e:
            # Catch other OS errors like file truncation if ImageFile.LOAD_TRUNCATED_IMAGES is False
            # print(f"Warning: OS error loading image {img_path}: {e}, skipping.")
            return None
        except Exception as e:
            # Catch any other unexpected errors during loading
            # print(f"Warning: Unexpected error loading image {img_path}: {e}, skipping.")
            return None

# Define data transformations
train_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # Added saturation/hue
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # Slight translation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0), # Added Random Erasing
])

val_test_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom collate function to filter out None samples
def collate_fn_skip_none(batch):
    """Collates batch data, skipping None items."""
    original_len = len(batch)
    batch = [item for item in batch if item is not None]
    filtered_len = len(batch)
    if original_len > filtered_len:
        # print(f"Skipped {original_len - filtered_len} samples due to loading errors.")
        pass # Reduce verbose logging
    if not batch:
        # print("Warning: Entire batch was filtered out.")
        return None # Return None if the whole batch is invalid
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e:
        print(f"Error during collation: {e}")
        # Potentially inspect the batch items here if errors persist
        # for i, item in enumerate(batch):
        #     print(f"Item {i}: Image type {type(item[0])}, Label type {type(item[1])}")
        #     if isinstance(item[0], torch.Tensor):
        #         print(f" Image shape: {item[0].shape}")
        return None # Return None if collation fails


# Learning rate scheduler with warmup
class WarmupCosineScheduler:
    """Cosine LR scheduler with linear warmup."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        # Ensure base_lr is correctly fetched even with multiple param groups
        self.base_lr = max(pg['lr'] for pg in optimizer.param_groups)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing phase
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            # Ensure progress doesn't go beyond 1 for cosine calculation
            progress = min(progress, 1.0)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))

        # Apply the calculated LR to all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# Early stopping class
class EarlyStopping:
    """Early stops the training if validation score doesn't improve after a given patience."""
    def __init__(self, patience=7, min_delta=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation score improved.
                            Default: 7
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                               Default: 0
            verbose (bool): If True, prints a message for each validation score improvement.
                            Default: False
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        """Checks if training should stop based on the validation score."""
        score = val_score

        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(f"EarlyStopping: Best score set to {score:.6f}")
            return False # Don't stop yet
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                 print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True # Stop training
        else:
            if self.verbose and score > self.best_score:
                print(f"EarlyStopping: Validation score improved ({self.best_score:.6f} --> {score:.6f}). Resetting counter.")
            self.best_score = score
            self.counter = 0
            return False # Don't stop yet

        return self.early_stop # Return current stop status

# Define the model
def create_fracture_model():
    """Creates the ViT model."""
    model = create_model(
        config.model_name,
        pretrained=config.pretrained,
        num_classes=config.num_classes
    )
    return model

# Apply SMOTE to training data (Modified to use only SMOTE)
def apply_smote(X, y):
    """Apply SMOTE (only) to address class imbalance"""
    print("Applying SMOTE to balance the dataset...")
    print(f"Original class distribution: {Counter(y)}")

    # Create SMOTE object
    smote = SMOTE(
        sampling_strategy=config.smote_ratio,
        random_state=config.random_state
    )

    # Apply SMOTE
    # Reshape X to be 2D for imblearn compatibility
    X_np = np.array(X).reshape(-1, 1)
    X_resampled_np, y_resampled = smote.fit_resample(X_np, y)

    # Flatten X_resampled back to a list of indices
    X_resampled = X_resampled_np.flatten().tolist()

    print(f"Resampled class distribution: {Counter(y_resampled)}")
    return X_resampled, y_resampled

# --- Training and Evaluation Functions ---
def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = [] # Store probabilities for AUC calculation

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    num_processed_samples = 0 # Keep track of actual samples processed

    for batch_data in progress_bar:
        # Skip if batch is None (due to collation errors)
        if batch_data is None:
            continue
        images, labels = batch_data
        # Skip if tensors are empty (can happen if batch size is 1 and it fails)
        if images is None or labels is None or images.nelement() == 0:
            continue

        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Add channel dimension for BCE loss
        batch_size = images.size(0)
        num_processed_samples += batch_size

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        if config.use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            # Optional: Gradient clipping can sometimes help stabilize training
            # scaler.unscale_(optimizer) # Unscale gradients before clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training without mixed precision
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Statistics
        running_loss += loss.item() * batch_size

        # Store predictions and labels
        # Use detach() before converting to numpy
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = probs > 0.5
        all_probs.extend(probs.flatten().tolist()) # Use extend with list
        all_preds.extend(preds.flatten().tolist())
        all_labels.extend(labels.detach().cpu().numpy().flatten().tolist())

        # Update progress bar
        progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    # Handle case where no samples were processed
    if num_processed_samples == 0:
        print("Warning: No valid samples processed in this training epoch.")
        return { 'loss': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0.5 }

    epoch_loss = running_loss / num_processed_samples
    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)
    all_probs = np.array(all_probs)

    # Calculate metrics safely
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    # Handle cases with only one class present in the batch/epoch
    auc_score = 0.5
    if len(np.unique(all_labels)) > 1:
       try:
           auc_score = roc_auc_score(all_labels, all_probs)
       except ValueError as e:
           print(f"Warning: Could not calculate AUC for training: {e}")

    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score
    }
    return metrics

def evaluate(model, dataloader, criterion, device, desc="Evaluating"):
    """Evaluates the model on a given dataset."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []  # For ROC-AUC
    num_processed_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=desc, leave=False)
        for batch_data in progress_bar:
            # Skip if batch is None
            if batch_data is None:
                continue
            images, labels = batch_data
            # Skip if tensors are empty
            if images is None or labels is None or images.nelement() == 0:
                continue

            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # Add channel dimension for BCE loss
            batch_size = images.size(0)
            num_processed_samples += batch_size

            # Forward pass (no autocast needed for evaluation unless specifically desired)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * batch_size

            # Store predictions and labels
            probs = torch.sigmoid(outputs).cpu().numpy() # No detach needed in no_grad context
            preds = probs > 0.5
            all_probs.extend(probs.flatten().tolist())
            all_preds.extend(preds.flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())

    # Handle case where no samples were processed
    if num_processed_samples == 0:
        print(f"Warning: No valid samples processed during {desc}.")
        return { 'loss': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0.5, 'labels': [], 'probs': [] }

    epoch_loss = running_loss / num_processed_samples
    all_preds = np.array(all_preds).astype(int)
    all_labels = np.array(all_labels).astype(int)
    all_probs = np.array(all_probs)

    # Calculate metrics safely
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    # Handle cases with only one class present
    auc_score = 0.5
    if len(np.unique(all_labels)) > 1:
        try:
            auc_score = roc_auc_score(all_labels, all_probs)
        except ValueError as e:
            print(f"Warning: Could not calculate AUC for {desc}: {e}")


    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'labels': all_labels, # Return labels and probs for detailed plots
        'probs': all_probs
    }
    return metrics
# --- End Training and Evaluation Functions ---

# --- Plotting Functions ---
def plot_training_history(history, plot_dir, fold_num=None):
    """Plots training and validation metrics over epochs."""
    epochs = range(1, len(history['train_loss']) + 1)
    timestamp = time.strftime("%Y%m%d-%H%M%S") # Unique timestamp for filenames
    fold_suffix = f"_fold_{fold_num}" if fold_num is not None else "_final"

    plt.figure(figsize=(12, 10))

    # Plot Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title(f'Loss (Fold {fold_num})' if fold_num else 'Loss (Final Training)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'ro-', label='Validation Accuracy')
    plt.title(f'Accuracy (Fold {fold_num})' if fold_num else 'Accuracy (Final Training)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_f1'], 'bo-', label='Training F1 Score')
    plt.plot(epochs, history['val_f1'], 'ro-', label='Validation F1 Score')
    plt.title(f'F1 Score (Fold {fold_num})' if fold_num else 'F1 Score (Final Training)')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    # Plot AUC
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['train_auc'], 'bo-', label='Training AUC')
    plt.plot(epochs, history['val_auc'], 'ro-', label='Validation AUC')
    plt.title(f'AUC (Fold {fold_num})' if fold_num else 'AUC (Final Training)')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_filename = os.path.join(plot_dir, f'training_metrics{fold_suffix}_{timestamp}.png')
    plt.savefig(plot_filename)
    print(f"Saved training history plot to {plot_filename}")
    plt.close() # Close the plot to free memory

def plot_roc_curve(labels, probs, plot_dir, dataset_name="Test"):
    """Plots the ROC curve."""
    # Ensure there are both classes present
    if len(np.unique(labels)) < 2:
        print(f"Skipping ROC curve plot for {dataset_name}: Only one class present.")
        return
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {dataset_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plot_filename = os.path.join(plot_dir, f'{dataset_name.lower()}_roc_curve_{timestamp}.png')
    plt.savefig(plot_filename)
    print(f"Saved ROC curve plot to {plot_filename}")
    plt.close()

def plot_precision_recall_curve(labels, probs, plot_dir, dataset_name="Test"):
    """Plots the Precision-Recall curve."""
     # Ensure there are both classes present
    if len(np.unique(labels)) < 2:
        print(f"Skipping Precision-Recall curve plot for {dataset_name}: Only one class present.")
        return
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision) # Note: order is recall, precision for auc()
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
    # Calculate no-skill line (ratio of positives)
    no_skill = np.sum(labels == 1) / len(labels) if len(labels) > 0 else 0
    plt.plot([0, 1], [no_skill, no_skill], color='navy', lw=2, linestyle='--', label=f'No Skill (AP={no_skill:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_name}')
    plt.legend(loc="lower left")
    plt.grid(True)
    plot_filename = os.path.join(plot_dir, f'{dataset_name.lower()}_pr_curve_{timestamp}.png')
    plt.savefig(plot_filename)
    print(f"Saved Precision-Recall curve plot to {plot_filename}")
    plt.close()

def plot_fold_metrics(fold_metrics, plot_dir):
    """Plot metrics across folds."""
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    num_metrics = len(metrics_to_plot)
    folds = range(1, len(fold_metrics) + 1)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Determine grid size for subplots
    ncols = 3
    nrows = (num_metrics + ncols - 1) // ncols

    plt.figure(figsize=(5 * ncols, 4 * nrows))
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(nrows, ncols, i + 1)
        # Extract metric values, handling potential missing keys or None values
        values = [fm.get(metric, 0) if fm is not None else 0 for fm in fold_metrics]
        mean_value = np.mean(values)
        std_dev = np.std(values)

        bars = plt.bar(folds, values, yerr=std_dev, capsize=5, alpha=0.7)
        plt.axhline(y=mean_value, color='r', linestyle='-', label=f'Mean: {mean_value:.3f}')

        # Add value labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center', fontsize=9) # va='bottom' places text above bar

        plt.title(f'{metric.capitalize()} Across Folds')
        plt.xlabel('Fold')
        plt.ylabel(metric.capitalize())
        plt.xticks(folds)
        plt.ylim(0, 1.05) # Extend y-limit slightly for labels
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.6) # Grid lines for y-axis only

    plt.tight_layout()
    plot_filename = os.path.join(plot_dir, f'fold_metrics_{timestamp}.png')
    plt.savefig(plot_filename)
    print(f"Saved fold metrics plot to {plot_filename}")
    plt.close()
# --- End Plotting Functions ---

# --- K-Fold Cross Validation Function ---
def train_with_kfold():
    """Perform k-fold cross-validation."""
    print("\nStarting K-fold Cross Validation...")

    # Initialize K-fold
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)

    # Get all data and labels
    try:
        all_data = pd.read_csv(config.train_csv)
    except FileNotFoundError:
        print(f"Error: Training CSV {config.train_csv} not found. Aborting k-fold.")
        return [], None # Return empty results
    X_indices = list(range(len(all_data)))  # Original indices
    y_labels = all_data['label'].values

    # Initialize lists to store metrics and model states for each fold
    fold_val_metrics = [] # Store validation metrics for each fold's best epoch
    fold_best_model_states = [] # Store the state_dict of the best model for each fold

    # Calculate positive class weight for weighted loss if enabled
    current_pos_weight = None
    if config.use_weighted_loss:
        neg_samples = np.sum(y_labels == 0)
        pos_samples = np.sum(y_labels == 1)
        if pos_samples > 0:
            current_pos_weight = torch.tensor(neg_samples / pos_samples).to(config.device)
            print(f"Class distribution - Negative: {neg_samples}, Positive: {pos_samples}")
            print(f"Using positive weight of {current_pos_weight.item():.4f} for weighted loss")
        else:
            print("Warning: No positive samples found in the dataset. Weighted loss disabled.")
            config.use_weighted_loss = False # Disable if no positive samples

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_indices, y_labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{config.n_folds}")
        print(f"{'='*50}")

        fold_train_indices = np.array(X_indices)[train_idx]
        fold_train_labels = y_labels[train_idx]

        # Apply SMOTE if enabled
        if config.use_smote:
            resampled_train_indices, _ = apply_smote(fold_train_indices, fold_train_labels)
        else:
            resampled_train_indices = fold_train_indices.tolist()
            print("SMOTE disabled for this fold.")

        # Create datasets for this fold
        train_dataset = FractureDataset(
            csv_file=config.train_csv,
            img_dir=config.train_dir,
            transform=train_transform,
            indices=resampled_train_indices # Use resampled indices
        )

        val_dataset = FractureDataset(
            csv_file=config.train_csv,
            img_dir=config.train_dir,
            transform=val_test_transform,
            indices=val_idx # Use original validation indices
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, collate_fn=collate_fn_skip_none,
            drop_last=True # Consider dropping last incomplete batch for stability
        )

        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=0, pin_memory=True, collate_fn=collate_fn_skip_none
        )

        # Check if loaders are empty
        if len(train_loader) == 0 or len(val_loader) == 0:
             print(f"Warning: DataLoader empty for Fold {fold + 1}. Skipping fold.")
             fold_val_metrics.append(None) # Placeholder for skipped fold
             fold_best_model_states.append(None)
             continue

        # Initialize model
        model = create_fracture_model().to(config.device)
        swa_model = None # Initialize SWA model as None

        # Initialize criterion with positive weight if enabled
        criterion = nn.BCEWithLogitsLoss(pos_weight=current_pos_weight if config.use_weighted_loss else None)

        # Initialize optimizer
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        # Initialize LR scheduler
        lr_scheduler = WarmupCosineScheduler(optimizer, config.warmup_epochs, config.num_epochs, config.min_lr)

        # Initialize SWA specifics only if SWA is enabled
        swa_scheduler = None
        if config.use_swa:
            swa_model = AveragedModel(model)
            swa_scheduler = SWALR(optimizer, swa_lr=config.swa_lr, anneal_epochs=1, anneal_strategy='cos') # Use adjusted swa_lr

        # Initialize gradient scaler for mixed precision
        scaler = GradScaler() if config.use_amp else None

        # Initialize early stopping for this fold
        early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta, verbose=True)

        # History dictionary for this fold
        history = {
            'train_loss': [], 'train_accuracy': [], 'train_precision': [], 'train_recall': [], 'train_f1': [], 'train_auc': [],
            'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_auc': []
        }

        best_val_f1_fold = -1.0 # Initialize with a value lower than any possible F1
        best_model_state_fold = None
        stopped_epoch = config.num_epochs

        # Training loop for this fold
        for epoch in range(config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
            print("-" * 30)

            # Update learning rate
            # Use SWA scheduler only if SWA is enabled and after swa_start
            if config.use_swa and epoch >= config.swa_start:
                if swa_scheduler: swa_scheduler.step()
                current_lr_display = config.swa_lr
                print(f"Using SWA learning rate: {current_lr_display:.6f}")
            else:
                current_lr_display = lr_scheduler.step(epoch)
                print(f"Current learning rate: {current_lr_display:.6f}")

            # Train
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, config.device, scaler)

            # Update SWA model if enabled and appropriate epoch
            if config.use_swa and swa_model is not None and epoch >= config.swa_start:
                swa_model.update_parameters(model)

            # Evaluate on validation set using the *standard* model for checkpointing and early stopping
            val_metrics = evaluate(model, val_loader, criterion, config.device, desc="Validating")

            # Store metrics in history
            for key in train_metrics: history[f'train_{key}'].append(train_metrics[key])
            for key in val_metrics:
                if key not in ['labels', 'probs']: history[f'val_{key}'].append(val_metrics[key])

            # Print metrics
            print(f"Train | Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['auc']:.4f}")
            print(f"Val   | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")

            # Save best model state based on validation F1
            if val_metrics['f1'] > best_val_f1_fold:
                best_val_f1_fold = val_metrics['f1']
                # Save state dict directly to avoid potential issues with deepcopy
                best_model_state_fold = {k: v.cpu() for k, v in model.state_dict().items()}
                print(f"*** New best model for Fold {fold + 1} saved with Val F1: {best_val_f1_fold:.4f} at epoch {epoch + 1} ***")

            # Check for early stopping based on standard model's validation F1
            if early_stopping(val_metrics['f1']):
                stopped_epoch = epoch + 1
                print(f"Early stopping triggered at epoch {stopped_epoch} for Fold {fold+1}")
                break # Exit epoch loop for this fold

        # --- Post-Epoch Loop for the Fold ---

        # Evaluate the SWA model *after* the loop finishes (or stops early) if SWA was used
        final_fold_metrics = None
        final_model_state_to_use = best_model_state_fold # Default to best non-SWA model

        if config.use_swa and swa_model is not None and stopped_epoch > config.swa_start : # Check if SWA ran
            print("Updating batch normalization for SWA model...")
            try:
                # Ensure train_loader is available and not empty for BN update
                if len(train_loader) > 0:
                     torch.optim.swa_utils.update_bn(train_loader, swa_model, device=config.device)
                     # Evaluate SWA model
                     swa_val_metrics = evaluate(swa_model, val_loader, criterion, config.device, desc="SWA Validating")
                     print(f"SWA Val | Loss: {swa_val_metrics['loss']:.4f} | Acc: {swa_val_metrics['accuracy']:.4f} | F1: {swa_val_metrics['f1']:.4f} | AUC: {swa_val_metrics['auc']:.4f}")

                     # Compare SWA F1 with the best standard model F1 for this fold
                     if swa_val_metrics['f1'] > best_val_f1_fold:
                         print(f"SWA model is better for Fold {fold+1}! Val F1: {swa_val_metrics['f1']:.4f}")
                         final_fold_metrics = swa_val_metrics
                         # Save SWA model state
                         final_model_state_to_use = {k: v.cpu() for k, v in swa_model.state_dict().items()}
                     else:
                         print(f"Best standard model (F1: {best_val_f1_fold:.4f}) was better than SWA model (F1: {swa_val_metrics['f1']:.4f}) for Fold {fold+1}.")
                         # Need to evaluate the best standard model again to get its full metrics dict
                         temp_model = create_fracture_model().to(config.device)
                         temp_model.load_state_dict({k: v.to(config.device) for k, v in best_model_state_fold.items()})
                         final_fold_metrics = evaluate(temp_model, val_loader, criterion, config.device, desc="Best Standard Model Re-eval")
                         del temp_model # Clean up
                else:
                    print("Skipping SWA BN update and evaluation as train_loader is empty.")
                    # Fallback to best standard model metrics if SWA eval fails
                    temp_model = create_fracture_model().to(config.device)
                    temp_model.load_state_dict({k: v.to(config.device) for k, v in best_model_state_fold.items()})
                    final_fold_metrics = evaluate(temp_model, val_loader, criterion, config.device, desc="Best Standard Model Re-eval")
                    del temp_model # Clean up

            except Exception as e:
                 print(f"Error during SWA BN update or evaluation: {e}. Using best standard model state.")
                 # Fallback to best standard model metrics if SWA eval fails
                 temp_model = create_fracture_model().to(config.device)
                 temp_model.load_state_dict({k: v.to(config.device) for k, v in best_model_state_fold.items()})
                 final_fold_metrics = evaluate(temp_model, val_loader, criterion, config.device, desc="Best Standard Model Re-eval")
                 del temp_model # Clean up

        elif best_model_state_fold is not None:
             # If SWA wasn't used or didn't improve, get metrics from the best standard epoch
             print(f"Using best standard model state from Fold {fold+1} (Val F1: {best_val_f1_fold:.4f})")
             temp_model = create_fracture_model().to(config.device)
             temp_model.load_state_dict({k: v.to(config.device) for k, v in best_model_state_fold.items()})
             final_fold_metrics = evaluate(temp_model, val_loader, criterion, config.device, desc="Best Standard Model Re-eval")
             del temp_model # Clean up
        else:
             print(f"Warning: No best model state found for Fold {fold+1}. Skipping metrics.")
             final_fold_metrics = None # Indicate failure for this fold


        # Store the best metrics and state dict for this fold
        fold_val_metrics.append(final_fold_metrics)
        fold_best_model_states.append(final_model_state_to_use)

        # Plot fold-specific training curves
        plot_training_history(history, config.plot_dir, fold_num=fold + 1)

        # Clean up memory
        del model, swa_model, optimizer, lr_scheduler, swa_scheduler, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache()
        print("-" * 50)
        # --- End of Fold Loop ---

    # --- After All Folds ---
    # Filter out None results from skipped folds
    valid_fold_indices = [i for i, fm in enumerate(fold_val_metrics) if fm is not None]
    if not valid_fold_indices:
        print("\nError: No folds completed successfully. Cannot determine best model or average metrics.")
        return [], None

    valid_fold_metrics = [fold_val_metrics[i] for i in valid_fold_indices]
    valid_model_states = [fold_best_model_states[i] for i in valid_fold_indices]

    # Plot metrics across completed folds
    if valid_fold_metrics:
        plot_fold_metrics(valid_fold_metrics, config.plot_dir)

        # Calculate average metrics across completed folds
        avg_metrics = {
            metric: np.mean([fold[metric] for fold in valid_fold_metrics])
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']
        }

        print("\nAverage Validation metrics across completed folds:")
        print(f"Accuracy:  {avg_metrics['accuracy']:.4f}")
        print(f"Precision: {avg_metrics['precision']:.4f}")
        print(f"Recall:    {avg_metrics['recall']:.4f}")
        print(f"F1 Score:  {avg_metrics['f1']:.4f}")
        print(f"AUC-ROC:   {avg_metrics['auc']:.4f}")

        # Find the best model state based on the highest validation F1 across completed folds
        best_fold_idx_overall = np.argmax([fm['f1'] for fm in valid_fold_metrics])
        best_state_dict = valid_model_states[best_fold_idx_overall]
        best_f1_overall = valid_fold_metrics[best_fold_idx_overall]['f1']
        original_fold_num = valid_fold_indices[best_fold_idx_overall] + 1 # Get the original fold number

        print(f"\nBest model overall from Fold {original_fold_num} with Validation F1: {best_f1_overall:.4f}")

        # Save the state_dict of the overall best model from k-fold
        torch.save(best_state_dict, config.best_kfold_model_path)
        print(f"Saved best k-fold model state from Fold {original_fold_num} to {config.best_kfold_model_path}")

        return valid_fold_metrics, best_state_dict # Return metrics and the best state dict
    else:
        print("\nNo valid fold metrics to analyze.")
        return [], None


# --- Training Loop for Full Dataset (Optional - K-Fold best is usually preferred) ---
# This function remains largely the same but is now optional.
# The main script logic will prioritize testing the best k-fold model.
def train_final_model():
    print("\nStarting Final Training on Full Dataset (Optional Step)...")
    # Load all training data
    full_train_data_info = pd.read_csv(config.train_csv)
    labels = full_train_data_info['label'].values
    indices = list(range(len(full_train_data_info)))

    # Split indices into training and validation sets (stratified)
    train_indices, val_indices = train_test_split(
        indices,
        test_size=config.validation_split,
        random_state=config.random_state,
        stratify=labels # Ensure similar class distribution
    )

    print(f"Original Train samples: {len(indices)}")
    print(f"Final Train split samples: {len(train_indices)}")
    print(f"Final Validation split samples: {len(val_indices)}")

    train_labels_split = labels[train_indices]

    # Apply SMOTE if enabled
    if config.use_smote:
        resampled_train_indices, _ = apply_smote(train_indices, train_labels_split)
    else:
        resampled_train_indices = train_indices.tolist()
        print("SMOTE disabled for final training.")

    # Create datasets
    train_dataset = FractureDataset(
        csv_file=config.train_csv, img_dir=config.train_dir,
        transform=train_transform, indices=resampled_train_indices
    )
    val_dataset = FractureDataset(
        csv_file=config.train_csv, img_dir=config.train_dir,
        transform=val_test_transform, indices=val_indices
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_fn_skip_none, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn_skip_none
    )

    if len(train_loader) == 0 or len(val_loader) == 0:
        print("Warning: DataLoader empty for final training. Skipping.")
        return None, -1.0 # Indicate failure

    # Initialize model
    model = create_fracture_model().to(config.device)
    swa_model = None

    # Calculate positive weight for loss if enabled
    current_pos_weight = None
    if config.use_weighted_loss:
        neg_samples = np.sum(labels == 0) # Use full dataset labels for weight calc
        pos_samples = np.sum(labels == 1)
        if pos_samples > 0:
            current_pos_weight = torch.tensor(neg_samples / pos_samples).to(config.device)
            print(f"Using positive weight of {current_pos_weight.item():.4f} for weighted loss")
        else:
            print("Warning: No positive samples found. Weighted loss disabled.")
            config.use_weighted_loss = False

    criterion = nn.BCEWithLogitsLoss(pos_weight=current_pos_weight if config.use_weighted_loss else None)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler = WarmupCosineScheduler(optimizer, config.warmup_epochs, config.num_epochs, config.min_lr)

    swa_scheduler = None
    if config.use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=config.swa_lr, anneal_epochs=1, anneal_strategy='cos')

    scaler = GradScaler() if config.use_amp else None
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta, verbose=True)

    history = {
        'train_loss': [], 'train_accuracy': [], 'train_precision': [], 'train_recall': [], 'train_f1': [], 'train_auc': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_auc': []
    }

    best_val_f1 = -1.0
    best_epoch = -1
    best_model_state = None
    stopped_epoch = config.num_epochs

    print("\nStarting Final Training Loop...")
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print("-" * 30)

        if config.use_swa and epoch >= config.swa_start:
            if swa_scheduler: swa_scheduler.step()
            current_lr_display = config.swa_lr
            print(f"Using SWA learning rate: {current_lr_display:.6f}")
        else:
            current_lr_display = lr_scheduler.step(epoch)
            print(f"Current learning rate: {current_lr_display:.6f}")

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, config.device, scaler)
        if config.use_swa and swa_model is not None and epoch >= config.swa_start:
            swa_model.update_parameters(model)
        val_metrics = evaluate(model, val_loader, criterion, config.device, desc="Validating")

        for key in train_metrics: history[f'train_{key}'].append(train_metrics[key])
        for key in val_metrics:
            if key not in ['labels', 'probs']: history[f'val_{key}'].append(val_metrics[key])

        print(f"Train | Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['auc']:.4f}")
        print(f"Valid | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"*** New best model for Final Training saved with Val F1: {best_val_f1:.4f} at epoch {epoch + 1} ***")

        if early_stopping(val_metrics['f1']):
            stopped_epoch = epoch + 1
            print(f"Early stopping triggered at epoch {stopped_epoch} for Final Training")
            break

    # Evaluate SWA model if used
    final_model_state_to_save = best_model_state
    final_best_f1 = best_val_f1

    if config.use_swa and swa_model is not None and stopped_epoch > config.swa_start:
        print("Updating batch normalization for SWA model (Final Training)...")
        try:
            if len(train_loader) > 0:
                torch.optim.swa_utils.update_bn(train_loader, swa_model, device=config.device)
                swa_val_metrics = evaluate(swa_model, val_loader, criterion, config.device, desc="SWA Validating (Final)")
                print(f"SWA Val | Loss: {swa_val_metrics['loss']:.4f} | Acc: {swa_val_metrics['accuracy']:.4f} | F1: {swa_val_metrics['f1']:.4f} | AUC: {swa_val_metrics['auc']:.4f}")
                if swa_val_metrics['f1'] > best_val_f1:
                    print(f"SWA model is better! Final Val F1: {swa_val_metrics['f1']:.4f}")
                    final_model_state_to_save = {k: v.cpu() for k, v in swa_model.state_dict().items()}
                    final_best_f1 = swa_val_metrics['f1']
                else:
                     print(f"Best standard model (F1: {best_val_f1:.4f}) was better than SWA model (F1: {swa_val_metrics['f1']:.4f}).")
            else:
                print("Skipping SWA BN update and evaluation as train_loader is empty.")
        except Exception as e:
            print(f"Error during SWA BN update or evaluation in final training: {e}. Using best standard model state.")

    # Save the best model state from this final training run
    if final_model_state_to_save is not None:
        torch.save(final_model_state_to_save, config.final_trained_model_path)
        print(f"\nFinal training complete. Best model from this run saved at epoch {best_epoch+1} with Validation F1: {final_best_f1:.4f} to {config.final_trained_model_path}")
    else:
        print("\nFinal training finished, but no best model state was saved.")

    # Plot training history
    plot_training_history(history, config.plot_dir, fold_num=None) # Pass None for final training plot

    return final_model_state_to_save, final_best_f1

# --- Testing Function ---
def test_model(model_state_dict):
    """Tests a model given its state dictionary."""
    print("\nLoading model state for final testing...")
    if model_state_dict is None:
        print("Error: No model state dictionary provided for testing. Aborting.")
        return False

    test_model = create_fracture_model()
    try:
        # Load state dict onto the correct device
        test_model.load_state_dict({k: v.to(config.device) for k, v in model_state_dict.items()})
        test_model = test_model.to(config.device)
        test_model.eval() # Set to evaluation mode
        print("Model loaded successfully for testing.")
    except Exception as e:
        print(f"Error loading model state_dict for testing: {e}. Aborting.")
        return False

    # Create test dataset and loader
    try:
        test_dataset = FractureDataset(
            csv_file=config.test_csv,
            img_dir=config.test_dir,
            transform=val_test_transform
        )
    except Exception as e:
        print(f"Error creating test dataset: {e}. Aborting testing.")
        return False

    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn_skip_none
    )

    if len(test_loader) == 0:
        print("Warning: Test DataLoader is empty. Cannot perform testing.")
        return False

    # Use standard BCE loss for evaluation (no weighting needed)
    criterion = nn.BCEWithLogitsLoss()

    # Evaluate on the test set
    print("Evaluating on the test set...")
    test_metrics = evaluate(test_model, test_loader, criterion, config.device, desc="Testing")

    print("\n--- Test Set Results ---")
    print(f"Loss:      {test_metrics['loss']:.4f}")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")
    print(f"AUC-ROC:   {test_metrics['auc']:.4f}")

    # Print detailed classification report if labels are available
    if len(test_metrics['labels']) > 0:
        y_pred = (test_metrics['probs'] > 0.5).astype(int)
        print("\nClassification Report:")
        try:
            print(classification_report(test_metrics['labels'], y_pred,
                                      target_names=['Non_Fractured', 'Fractured'], zero_division=0))
        except ValueError as e:
             print(f"Could not generate classification report: {e}") # Handle case where only one class is predicted/present

        # Plot ROC and Precision-Recall curves for the test set
        plot_roc_curve(test_metrics['labels'], test_metrics['probs'], config.plot_dir, dataset_name="Test")
        plot_precision_recall_curve(test_metrics['labels'], test_metrics['probs'], config.plot_dir, dataset_name="Test")
    else:
        print("No labels available for detailed test report or plots.")

    return True # Indicate testing was performed

# --- Main Execution ---
if __name__ == "__main__":
    # --- Step 1: K-Fold Cross-Validation ---
    print("\n" + "="*50)
    print("Step 1: Stratified K-fold Cross-Validation")
    print("="*50)
    fold_metrics, best_kfold_state_dict = train_with_kfold()

    # --- Step 2: Test the Best Model from K-Fold ---
    print("\n" + "="*50)
    print("Step 2: Testing Best Model from K-Fold")
    print("="*50)
    if best_kfold_state_dict:
        test_successful = test_model(best_kfold_state_dict)
        if test_successful:
            print("\nTesting of the best k-fold model completed.")
        else:
            print("\nTesting of the best k-fold model failed.")
    else:
        print("Skipping testing: No best model found from k-fold validation.")
        test_successful = False # Ensure final model isn't saved if k-fold failed

    # --- Step 3: (Optional) Train Final Model on Full Data ---
    # You might skip this if the k-fold best model is satisfactory
    print("\n" + "="*50)
    print("Step 3: (Optional) Training Final Model on Full Dataset")
    print("="*50)
    final_model_state, final_val_f1 = train_final_model()
    if final_model_state:
        print("\n--- Testing Final Trained Model ---")
        test_model(final_model_state)
    else:
        print("Skipping final model training or testing due to issues.")

    print("\nScript finished.")
