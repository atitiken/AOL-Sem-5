import os
import cv2
import pandas as pd
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import time


# Configuration matching vit.py
class Config:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(script_dir, "train_ratio_1in4.csv")
    test_csv = os.path.join(script_dir, "test_ratio_1in4.csv")
    train_dir = os.path.join(script_dir, "dataset_ratio_1in4", "train")
    test_dir = os.path.join(script_dir, "dataset_ratio_1in4", "test")
    img_size = 224  # Using same size as ViT for fair comparison
    plot_dir = os.path.join(script_dir, "plots", "hog_svm")
    cache_dir = os.path.join(script_dir, "cache", "hog_svm")
    model_dir = os.path.join(script_dir, "models", "hog_svm")


# Create directories
os.makedirs(Config.plot_dir, exist_ok=True)
os.makedirs(Config.cache_dir, exist_ok=True)
os.makedirs(Config.model_dir, exist_ok=True)


def load_data(csv_file, img_dir, desc="Loading Data", cache_name=None):
    """
    Loads images listed in the CSV, processes them, and extracts HOG features.
    Uses caching to speed up subsequent runs.
    """
    cache_path_X = None
    cache_path_y = None
    if cache_name:
        cache_path_X = os.path.join(Config.cache_dir, f"{cache_name}_X.joblib")
        cache_path_y = os.path.join(Config.cache_dir, f"{cache_name}_y.joblib")

        if os.path.exists(cache_path_X) and os.path.exists(cache_path_y):
            print(f"Loading cached features for {desc}...")
            return joblib.load(cache_path_X), joblib.load(cache_path_y)

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file}")
        return None, None

    X = []
    y = []

    print(f"Processing {len(df)} images from {desc}...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        img_name = row.iloc[0]  # Assuming first column is filename
        label = row.iloc[1]  # Assuming second column is label

        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        # Read image using cv2
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Resize to fixed size
        img = cv2.resize(img, (Config.img_size, Config.img_size))

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract HOG features
        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False,
            transform_sqrt=True,
        )

        X.append(features)
        y.append(label)

    X_arr = np.array(X)
    y_arr = np.array(y)

    if cache_name and "cache_path_X" in locals():
        print(f"Caching features for {desc}...")
        joblib.dump(X_arr, cache_path_X)
        joblib.dump(y_arr, cache_path_y)

    return X_arr, y_arr


def plot_confusion_matrix(y_true, y_pred, plot_dir):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["NonDemented", "VeryMildDemented"]
    )
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Confusion Matrix - HOG+SVM")
    plt.grid(False)
    plot_filename = os.path.join(plot_dir, f"svm_confusion_matrix_{timestamp}.png")
    plt.savefig(plot_filename)
    print(f"Saved confusion matrix to {plot_filename}")
    plt.close()


def plot_roc_curve(y_true, y_prob, plot_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.3f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - HOG+SVM")
    plt.legend(loc="lower right")
    plt.grid(True)
    plot_filename = os.path.join(plot_dir, f"svm_roc_curve_{timestamp}.png")
    plt.savefig(plot_filename)
    print(f"Saved ROC curve to {plot_filename}")
    plt.close()


def plot_pr_curve(y_true, y_prob, plot_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall, precision, color="blue", lw=2, label=f"PR curve (area = {pr_auc:.3f})"
    )
    no_skill = np.sum(y_true == 1) / len(y_true)
    plt.plot(
        [0, 1],
        [no_skill, no_skill],
        color="navy",
        lw=2,
        linestyle="--",
        label="No Skill",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - HOG+SVM")
    plt.legend(loc="lower left")
    plt.grid(True)
    plot_filename = os.path.join(plot_dir, f"svm_pr_curve_{timestamp}.png")
    plt.savefig(plot_filename)
    print(f"Saved PR curve to {plot_filename}")
    plt.close()


def train_and_evaluate():
    print("Step 1: Loading Training Data...")
    X_train, y_train = load_data(
        Config.train_csv, Config.train_dir, desc="Train Set", cache_name="train"
    )

    print("\nStep 2: Loading Test Data...")
    X_test, y_test = load_data(
        Config.test_csv, Config.test_dir, desc="Test Set", cache_name="test"
    )

    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: Could not load data.")
        return

    # Standardize features (important for SVM)
    print("\nStep 3: Scaling Features...")
    scaler_path = os.path.join(Config.model_dir, "svm_scaler.joblib")
    svm_model_path = os.path.join(Config.model_dir, "svm_model.joblib")

    if os.path.exists(scaler_path) and os.path.exists(svm_model_path):
        print("Loading saved scaler and model...")
        scaler = joblib.load(scaler_path)
        svm = joblib.load(svm_model_path)
        X_test_scaled = scaler.transform(X_test)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train SVM
        print("\nStep 4: Training SVM (this may take a while)...")
        # Using probability=True to get probabilities for AUC
        svm = SVC(
            kernel="rbf", probability=True, random_state=42, class_weight="balanced"
        )
        svm.fit(X_train_scaled, y_train)

        print("Saving model and scaler to models directory...")
        joblib.dump(scaler, scaler_path)
        joblib.dump(svm, svm_model_path)

    # Evaluate
    print("\nStep 5: Evaluating...")
    y_pred = svm.predict(X_test_scaled)
    y_prob = svm.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Use a faster approach for classification report if possible or just print
    print("\nHOG + SVM Results")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    auc_score = 0.5
    if len(np.unique(y_test)) > 1:
        auc_score = roc_auc_score(y_test, y_prob)
        print(f"AUC-ROC:   {auc_score:.4f}")

    print("-" * 50)

    # Simplified reporting to avoid timeout issues with classification_report on large output buffers?
    # Or simply print it. The previous timeout was odd.
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["NonDemented", "VeryMildDemented"],
            zero_division=0,
        )
    )

    # Generate Plots
    print("\nGenerating Plots...")
    plot_confusion_matrix(y_test, y_pred, Config.plot_dir)
    if len(np.unique(y_test)) > 1:
        plot_roc_curve(y_test, y_prob, Config.plot_dir)
        plot_pr_curve(y_test, y_prob, Config.plot_dir)


if __name__ == "__main__":
    train_and_evaluate()
