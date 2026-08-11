# Autonomous Driving Obstacle Classification

Binary image classification for autonomous-driving scenes, classifying images as **Obstacle** or **No Obstacle**.

This project uses imagery from the **BDD100K** dataset and focuses on developing a reliable classification pipeline through iterative dataset construction, model experimentation, error analysis, and visual interpretability.

---

## Overview

The objective of this project is to develop a binary image classifier capable of determining whether an autonomous-driving scene contains an obstacle.

A major part of the project was the development of the dataset itself. Instead of treating dataset preparation as a single preprocessing step, the dataset was developed iteratively through repeated cycles of training, inference, error analysis, and Grad-CAM visualization.

The initial dataset was used to train baseline models. Their predictions were then analyzed to identify difficult samples, failure cases, and patterns where the models were making incorrect predictions. These observations were used to manually refine the dataset before subsequent training iterations.

After approximately 20 iterations of this process, a final dataset containing **76,210 images** was established.

Three CNN architectures were then trained and evaluated on the final dataset:

- ResNet18
- MobileNetV3-Small
- EfficientNet-B0

EfficientNet-B0 was ultimately selected as the final model based on its overall balance between predictive performance and model complexity.

---

## Key Highlights

- Built a task-specific **Obstacle vs No Obstacle** dataset from BDD100K imagery.
- Iteratively refined the dataset using model predictions and error analysis.
- Used inference results to investigate false positives, false negatives, high-confidence predictions, and uncertain predictions.
- Used **Grad-CAM** to investigate which image regions influenced model predictions.
- Trained and compared three CNN architectures.
- Achieved **91.48% validation accuracy** with EfficientNet-B0.
- Final EfficientNet-B0 contains **4.34M parameters** with an approximate model size of **16.54 MB**.
- Implemented the training and inference pipeline using PyTorch.
- Used configuration-driven training experiments.
- Maintained dedicated notebooks for dataset analysis, training, model comparison, inference, and interpretability.

---

# Dataset

## Source

The project uses imagery from the **BDD100K** dataset, a large-scale autonomous-driving dataset containing diverse road scenes.

The original BDD100K dataset is **not included in this repository**.

The original dataset annotations were used as the starting point for constructing the binary classification task.

## Classes

| Class | Description |
|---|---|
| `No Obstacle` | Scene classified as not containing the target obstacle |
| `Obstacle` | Scene classified as containing the target obstacle |

---

# Dataset Development

Dataset development was one of the major components of this project.

The initial dataset contained approximately:

- **4,000 training images**
- **2,200 validation images**

Rather than directly scaling this dataset and training a final model, an iterative approach was followed.

After each training iteration, model predictions were analyzed using inference results and Grad-CAM visualizations. Difficult samples and failure cases were identified and manually reviewed. These observations were then used to refine the dataset before retraining.

The overall process was:

```text
BDD100K
   |
   v
Initial Dataset
   |
   v
Model Training
   |
   v
Inference
   |
   +------------------+
   |                  |
   v                  v
Error Analysis     Grad-CAM
   |                  |
   +--------+---------+
            |
            v
    Manual Dataset Refinement
            |
            v
        Retraining
            |
            +----> Repeat

The process was repeated for approximately 20 iterations, eventually resulting in the final dataset used for model comparison.

The analysis focused on:

False positives
False negatives
High-confidence incorrect predictions
Uncertain predictions
Difficult obstacle scenes
Difficult no-obstacle scenes

This iterative process was used to better understand both the dataset and the failure modes of the models.

Final Dataset

The final dataset contains 76,210 images.

Dataset Split
Split	Images
Training	74,028
Validation	2,182
Total	76,210
Overall Class Distribution
Class	Images
No Obstacle	38,961
Obstacle	37,249
Training Split
Class	Images
No Obstacle	37,637
Obstacle	36,391
Validation Split
Class	Images
No Obstacle	1,324
Obstacle	858
Model Experiments

Three CNN architectures were evaluated using the final dataset.

ResNet18

ResNet18 was used as a standard residual CNN baseline.

MobileNetV3-Small

MobileNetV3-Small was evaluated as a lightweight architecture with substantially fewer parameters and a smaller model footprint.

EfficientNet-B0

EfficientNet-B0 was evaluated as a compact CNN architecture designed to provide a strong balance between model capacity and computational efficiency.

Model Comparison

The final validation results were:

Model	Val Accuracy	Precision	Recall	F1	Val Loss	Best Epoch	Total Parameters	Trainable Parameters	Model Size
EfficientNet-B0	91.48%	0.930	0.930	0.930	0.386	3	4.34M	4.34M	16.54 MB
MobileNetV3-Small	91.29%	0.924	0.933	0.929	0.397	10	1.07M	1.07M	4.10 MB
ResNet18	91.20%	0.921	0.935	0.928	0.376	8	11.31M	11.31M	43.14 MB

The comparison considers both predictive performance and model complexity.

Final Model Selection
EfficientNet-B0

EfficientNet-B0 was selected as the final model based on its overall balance between predictive performance and model complexity.

It achieved the highest validation accuracy among the three models while remaining considerably smaller than ResNet18.

Metric	Result
Validation Accuracy	91.48%
Precision	0.930
Recall	0.930
F1 Score	0.930
Validation Loss	0.3859
Best Epoch	3
Parameters	4.34M
Model Size	16.54 MB

MobileNetV3-Small provides a substantially smaller model, while ResNet18 achieves slightly higher recall. However, EfficientNet-B0 provides the strongest overall performance-complexity trade-off for this project.

Error Analysis

Model evaluation was not limited to aggregate validation metrics.

Inference results were used to inspect individual predictions and investigate cases where the classifier failed.

False Positive
Ground Truth: No Obstacle
Prediction:   Obstacle
False Negative
Ground Truth: Obstacle
Prediction:   No Obstacle

Additional samples were examined based on prediction confidence, including:

High-confidence correct predictions
High-confidence incorrect predictions
Uncertain predictions
Difficult obstacle examples
Difficult no-obstacle examples

This analysis also contributed to the iterative dataset refinement process.

Grad-CAM Analysis

Grad-CAM was used to investigate which regions of an image contributed to model predictions.

The analysis was performed on different categories of samples, including:

High-confidence obstacle predictions
High-confidence no-obstacle predictions
Uncertain predictions
False positives
False negatives

The purpose was to determine whether the model was focusing on relevant visual regions and to better understand incorrect predictions.

The complete analysis is available in:

notebooks/05-gradcam-analysis.ipynb
Training

The training pipeline is implemented using PyTorch and is configuration-driven.

Training parameters are defined in:

config/config.json

The configuration supports parameters such as:

Model architecture
Number of classes
Learning rate
Batch size
Number of epochs
Image size
Weight decay
Backbone freezing strategy
Early stopping
Gradient accumulation
Experiment output paths

Training is started with:

python main.py --config="config/config.json"
Experiment Tracking

Experiments were organized to preserve information about individual training runs.

A typical experiment contains:

experiment/
├── best_model.pth
├── config.json
├── history.json
├── summary.json
└── tensorboard/

Experiment artifacts, model checkpoints, logs, and generated datasets are kept outside the source-controlled repository.

This keeps the GitHub repository focused on the source code, configuration, notebooks, and documentation.

Inference

The project includes a dedicated inference pipeline for evaluating trained models on images.

Inference results can include:

Image path
Predicted class
Prediction probability
Confidence score
Uncertainty information

The inference workflow is documented in:

notebooks/04-inference.ipynb
Notebooks

The notebooks provide the main analysis and experimental workflow used throughout the project.

Notebook	Description
01-dataset-overview.ipynb	Dataset development history, dataset statistics, class distribution, and sample visualization
02-training-models.ipynb	Model training and experiment execution
03-model-comparison.ipynb	Training curves, model metrics, and model complexity comparison
04-inference.ipynb	Inference results, confidence analysis, and prediction-level analysis
05-gradcam-analysis.ipynb	Grad-CAM visualizations and model error analysis
Repository Structure
autonomous-binary-classifier/
│
├── config/
│   ├── config.json
│   └── search_space.py
│
├── notebooks/
│   ├── 01-dataset-overview.ipynb
│   ├── 02-training-models.ipynb
│   ├── 03-model-comparison.ipynb
│   ├── 04-inference.ipynb
│   └── 05-gradcam-analysis.ipynb
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── gradcam.py
│   ├── inference.py
│   ├── model.py
│   ├── train.py
│   └── tuning.py
│
├── utils/
│   └── path_utils.py
│
├── .gitignore
├── main.py
├── requirements.txt
└── README.md

Large datasets, model checkpoints, logs, and experiment artifacts are excluded from version control.

Installation
Requirements
Python 3.11+
PyTorch
torchvision
NumPy
Pandas
OpenCV
Pillow
Matplotlib
scikit-learn
TensorBoard
tqdm

Install the required dependencies:

pip install -r requirements.txt
Usage
1. Configure the experiment

Update:

config/config.json

with the required dataset paths, model configuration, and training parameters.

2. Train a model
python main.py --config="config/config.json"
3. Perform analysis

The notebooks can then be used to inspect:

Dataset
   |
   +-- Training
   |
   +-- Model Comparison
   |
   +-- Inference
   |
   +-- Grad-CAM Analysis
Hardware

The main training experiments were performed using Kaggle GPU infrastructure with NVIDIA T4 GPUs.

The project can also be developed and executed locally using CPU-based PyTorch, depending on the workload.

Tech Stack
Programming
Python
Deep Learning
PyTorch
Torchvision
CNNs
Transfer Learning
Grad-CAM
Data Processing
NumPy
Pandas
OpenCV
Pillow
Visualization
Matplotlib
TensorBoard
Development
Jupyter Notebook
Git
GitHub
Kaggle
Project Workflow

The complete workflow can be summarized as:

BDD100K
   |
   v
Dataset Construction
   |
   v
Initial Training
   |
   v
Inference + Error Analysis
   |
   v
Grad-CAM Investigation
   |
   v
Dataset Refinement
   |
   v
Repeated Training
   |
   v
Final Dataset
   |
   v
Model Comparison
   |
   v
EfficientNet-B0 Selection
   |
   v
Final Inference
   |
   v
Interpretability Analysis
Future Improvements

Potential extensions include:

Evaluation on additional autonomous-driving datasets
Object detection for explicit obstacle localization
Segmentation-assisted obstacle analysis
Confidence calibration
Robustness evaluation under different weather and lighting conditions
Quantization and inference optimization
Real-time inference benchmarking
Deployment on edge hardware
Dataset Attribution

This project uses imagery from the BDD100K dataset for research and experimentation.

The original BDD100K dataset is not redistributed with this repository.

Author

Gracy Yadav

Machine Learning / Computer Vision