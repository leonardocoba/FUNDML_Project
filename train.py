import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time
import sys


def train(x, y, num_epochs=50, activ_func="Leaky_RELU", dropout=0.3, my_optim="adam", loss_fn="cross_entropy"):
    print("\n" + "="*50)
    print("starting training process")
    print("="*50)
    start_time = time.time()

    # Split the dataset into training and validation sets
    print("\nsplitting train/val...")
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    print(f"✅ split done")

    # Convert numpy arrays to PyTorch tensors and reshape inputs for CNN
    x_train_tensor = torch.tensor(x_train, dtype=torch.float).reshape(-1, 1, 100, 100)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float).reshape(-1, 1, 100, 100)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Map string to actual PyTorch activation classes
    activations = {
        "SELU": nn.SELU,
        "RELU": nn.ReLU,
        "ELU": nn.ELU,
        "Leaky_RELU": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid
    }

    if activ_func not in activations:
        print(f"❌ invalid activation function: {activ_func}")
        return

    print(f"\nbuilding cnn model: Activation Func: {activ_func}")
    activation = activations[activ_func]

    # Build the CNN model with three convolutional layers and a fully connected output
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        activation(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        activation(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        activation(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128 * 12 * 12, 256),
        activation(),
        nn.Dropout(dropout),
        nn.Linear(256, 10)
    )

    # Set up data loaders
    train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=64)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function
    if loss_fn == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        print(f"❌ invalid loss function: {loss_fn}")
        return

    # Select optimizer
    if my_optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif my_optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif my_optim == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    else:
        print(f"❌ invalid optimizer: {my_optim}")
        return

    best_accuracy = 0
    val_acc_logs = []
    progress_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

    for epoch in range(num_epochs):
        print(f"\nepoch {epoch+1}/{num_epochs}")
        model.train()
        total_loss = 0
        print("   training: ", end="")
        sys.stdout.flush()

        for i, (xb, yb) in enumerate(train_loader):
            progress = int((i / len(train_loader)) * len(progress_chars))
            if i % max(1, len(train_loader) // 10) == 0:
                print(progress_chars[min(progress, 7)], end="")
                sys.stdout.flush()

            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate model performance on validation set
        model.eval()
        correct, total = 0, 0
        print("\n   validating: ", end="")
        with torch.no_grad():
            for i, (xb, yb) in enumerate(val_loader):
                progress = int((i / len(val_loader)) * len(progress_chars))
                if i % max(1, len(val_loader) // 10) == 0:
                    print(progress_chars[min(progress, 7)], end="")
                    sys.stdout.flush()

                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                preds = torch.argmax(out, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        acc = correct / total
        print(f"\nval acc: {acc:.4f}")
        val_acc_logs.append(acc)

        # Save best model
        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(model.state_dict(), "model.pth")
            print(f"✅ model saved with acc: {acc:.4f}")

    torch.save(model.state_dict(), "model_final.pth")
    print("✅ final model saved")
    print(f"best val acc: {best_accuracy:.4f}")
    return val_acc_logs


if __name__ == "__main__":
    x = pd.read_csv("x_train_project.csv").values.astype("float32") / 255.0
    t = pd.read_csv("t_train_project.csv").squeeze().values
    logs = train(x, t, num_epochs=50, activ_func="Leaky_RELU", dropout=0.3, my_optim="adam", loss_fn="cross_entropy")
    plt.plot(logs)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    # train(x, t, activ_func="RELU")
    # train(x, t, activ_func="ELU")
    # train(x, t, activ_func="SELU")