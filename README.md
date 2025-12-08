# AOL-Sem-5: Alzheimer's MRI Classification

A comparison between a Vision Transformer (ViT) and a baseline HOG + SVM model for classifying MRI images (NonDemented vs. VeryMildDemented).

## Results

| Metric | ViT | HOG + SVM |
| :--- | :--- | :--- |
| **Accuracy** | **97.67%** | 95.33% |
| **F1 Score** | **0.9402** | 0.8692 |
| **Recall** | **91.67%** | 77.50% |
| **Precision** | 96.49% | **98.94%** |
| **AUC-ROC** | **0.9954** | 0.9881 |

The Vision Transformer significantly outperforms the baseline in **Recall**, meaning it is much better at detecting positive dementia cases.

## Usage

To run the evaluation and generate plots:

```bash
python test_models.py
```
