# Handwritten Greek Letter Classifier

## Authors

Leonardo Cobaleda, Jorge Ramirez, Zakhar Sennikov

## Overview

This project contains two Python scripts, `train.py` and `test.py`, that use a Convolutional Neural Network (CNN) to classify handwritten Greek letters. The system was built using PyTorch and trained on a dataset containing grayscale 100x100 images. We experimented with different activation functions, optimizers, and dropout values to determine the optimal configuration for classification accuracy.

## Packages Used

- `pandas`
- `torch`
- `matplotlib`
- `scikit-learn`

These are commonly used Python libraries for machine learning and data handling. Make sure they are installed in your Python environment.

---

## `train.py`

### Purpose

Trains a CNN model using specified configurations.

### Inputs

- `x_train_project.csv`: Input images as flattened pixel arrays (normalized internally).
- `t_train_project.csv`: Corresponding labels for each image (integer class from 0 to 9).

### Parameters (modifiable inside the script)

- `num_epochs`: Number of training epochs (default: 25).
- `activ_func`: Activation function to use (`RELU`, `ELU`, `SELU`, `Leaky_RELU`).
- `dropout`: Dropout rate for regularization (default: 0.3).
- `my_optim`: Optimizer to use (`adam`, `sgd`, `rmsprop`).
- `loss_fn`: Loss function to use (`cross_entropy`).

### Output

- `model.pth`: Saved best-performing model based on validation accuracy.
- `model_final.pth`: Final model at the end of all training epochs.
- Console logs of training/validation accuracy and saved checkpoints.
- A validation accuracy plot saved during training.

---

## `test.py`

### Purpose

Evaluates the performance of a trained CNN model using saved weights.

### Inputs

- `x_train_project.csv`: CSV file of test images (normalized within script).
- `t_train_project.csv`: CSV file of true labels for test images.

### How to Run

From the command line:

```bash
python test.py path/to/x_test.csv path/to/t_test.csv
```

### Output

- Displays overall test accuracy.
- Prints per-class accuracy (for classes 0 through 9).
- Shows evaluation time and progress bar.
- Returns a list of predicted class labels.

### Notes

- Only the best model (`model.pth`) is used during testing.
- Model architecture must match the one defined in `train.py` for weight loading to work correctly.
- The training script includes commented-out experiments with different activations and optimizers used during model comparison.

```

```
