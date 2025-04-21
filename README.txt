# Handwritten Greek Letter Classification

## Overview

This project trains and tests a neural network for classifying handwritten lowercase Greek letters using PyTorch. It includes support for both standard and "hard" test datasets. The model dynamically adjusts to either a Convolutional Neural Network (CNN) or a fallback Linear Network, depending on the input shape.

## Repository Contents

- `train.py`: Trains the model using CNN or linear layers, saves the best model.
- `test.py`: Evaluates the trained model on a standard test dataset.
- `test_hard.py`: Evaluates the trained model on a harder test dataset.

## Dependencies

Install the required packages via pip:

```bash
pip install pandas torch scikit-learn matplotlib
Input Files
Ensure the following CSV files are present in the same directory:

x_train_project.csv – Input features for training

t_train_project.csv – Target labels for training

x_test_hard.csv – Input features for hard test evaluation

t_test_hard.csv – Target labels for hard test evaluation

Running Instructions
Training
To run the training script:

bash
Copy
Edit
python train.py
You will be prompted to enter the number of epochs (default is 25). The script will:

Normalize input data

Split into training/validation sets

Train using CNN (or fallback to linear layers)

Save model.pth (best checkpoint) and model_final.pth (final model)

Warnings:
Training with fewer than 10 epochs may result in underfitting and poor performance.
Example:

kotlin
Copy
Edit
epoch 3/3 — val acc: 0.7592
epoch 25/25 — val acc: 0.8992
epoch 50/50 — val acc: 0.9025
Testing (Standard Test Set)
bash
Copy
Edit
python test.py
This evaluates the trained model (model.pth) on the training/test dataset and provides overall and per-class accuracy.

Testing (Hard Test Set)
bash
Copy
Edit
python test_hard.py
This runs evaluation on the hard dataset, using the same model architecture and weights, and prints per-class performance.

Output
Model Weights: model.pth (best accuracy), model_final.pth (last epoch)

Logs: Printed to console, includes:

Training progress bar

Loss and validation accuracy

Class-wise performance on test sets

Notes
Model defaults to CNN with input shape (N, 1, 100, 100). If this fails, it uses a linear model.

Scripts use CPU by default but support GPU (cuda) if available.

Visual progress bars are printed using Unicode characters (▁▂▃▄▅▆▇█) for fun and clarity.

Author(s)
Leonardo Cobaleda