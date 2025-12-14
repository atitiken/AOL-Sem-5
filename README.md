# AOL-Sem-5: Alzheimer's MRI Classification

A comparison between a Vision Transformer (ViT) and a baseline HOG + SVM model for classifying MRI images (NonDemented vs. VeryMildDemented).

## Results

| Metric | ViT-tuned | HOG + SVM |Vit-vanilla
| :--- | :--- | :--- | :--- |
| **Accuracy** | **97.67%** | 95.33% | 0.8717%|
| **F1 Score** | **0.9402** | 0.8692 | 0.5600%|
| **Recall** | **91.67%** | 77.50% | 0.4083%|
| **Precision** | 96.49% | **98.94%** | 0.8909%|

The Vision Transformer significantly outperforms the baseline in **Recall**, meaning it is much better at detecting positive dementia cases.

## Usage

To run the evaluation and generate plots:

```bash
python test_models.py
```
