import torch
import os
import sys
import joblib
import pandas as pd
import numpy as np
import cv2
import time
from skimage.feature import hog
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
import matplotlib.pyplot as plt

# Ensure the current directory is in the path so we can import vit
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vit import config, test_model


def load_hog_data(csv_file, img_dir, img_size=224):
    """Loads images, processes them, and extracts HOG features."""
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file}")
        return None, None

    X = []
    y = []

    print(f"Processing {len(df)} images for HOG testing...")

    for idx, row in df.iterrows():
        img_name = row.iloc[0]
        label = row.iloc[1]
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (img_size, img_size))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    return np.array(X), np.array(y)


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


def test_svm():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "models", "hog_svm")
    plot_dir = os.path.join(script_dir, "plots", "hog_svm")
    os.makedirs(plot_dir, exist_ok=True)

    scaler_path = os.path.join(model_dir, "svm_scaler.joblib")
    svm_model_path = os.path.join(model_dir, "svm_model.joblib")
    test_csv = os.path.join(script_dir, "test_ratio_1in4.csv")
    test_dir = os.path.join(script_dir, "dataset_ratio_1in4", "test")

    if not os.path.exists(scaler_path) or not os.path.exists(svm_model_path):
        print(f"Error: SVM model or scaler not found in {model_dir}")
        return

    print(f"\nLoading SVM model from {svm_model_path}...")
    scaler = joblib.load(scaler_path)
    svm = joblib.load(svm_model_path)

    print("Loading test data for SVM...")
    X_test, y_test = load_hog_data(test_csv, test_dir)

    if X_test is None:
        return

    print("Scaling features...")
    X_test_scaled = scaler.transform(X_test)

    print("Evaluating SVM...")
    y_pred = svm.predict(X_test_scaled)
    y_prob = svm.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    auc_score = 0.5
    if len(np.unique(y_test)) > 1:
        auc_score = roc_auc_score(y_test, y_prob)

    print("\n--- HOG + SVM Test Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc_score:.4f}")

    print("\nClassification Report (SVM):")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["NonDemented", "VeryMildDemented"],
            zero_division=0,
        )
    )

    # Generate Plots
    print("\nGenerating Plots for SVM...")
    plot_confusion_matrix(y_test, y_pred, plot_dir)
    if len(np.unique(y_test)) > 1:
        plot_roc_curve(y_test, y_prob, plot_dir)
        plot_pr_curve(y_test, y_prob, plot_dir)


def main():
    print("=" * 50)
    print("Model Testing")
    print("=" * 50)

    # --- Test ViT ---
    print("\n1. Testing Vision Transformer (ViT)...")
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models/tuned_v2/vit_ti_best_kfold_model.pth",
    )

    if os.path.exists(model_path):
        print(f"Loading ViT model from {model_path}...")
        try:
            state_dict = torch.load(model_path, map_location=config.device)
            test_model(state_dict)
        except Exception as e:
            print(f"An error occurred testing ViT: {e}")
    else:
        print(f"Error: ViT model file not found at {model_path}")

    # --- Test SVM ---
    print("\n2. Testing HOG + SVM...")
    try:
        test_svm()
    except Exception as e:
        print(f"An error occurred testing SVM: {e}")


if __name__ == "__main__":
    main()
