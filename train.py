import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time
import sys

def train(num_epochs=25):
    print("\n" + "="*50)
    print("starting training process")
    print("="*50)
    start_time = time.time()

    # load the data
    print("\nloading training data...")
    try:
        loading_start = time.time()
        x = pd.read_csv("x_train_project.csv").values
        y = pd.read_csv("t_train_project.csv").squeeze()
        loading_time = time.time() - loading_start
        print(f"✅ loaded data in {loading_time:.2f} sec")
        print(f"x shape: {x.shape}")
        print("label counts:\n", pd.Series(y).value_counts())
    except Exception as e:
        print(f"❌ error loading data: {e}")
        return None

    # normalize data
    print("\nnormalizing inputs...")
    try:
        x = x / 255.0
        print("✅ normalization complete")
    except Exception as e:
        print(f"❌ failed to normalize: {e}")

    # split into training and validation
    print("\nsplitting train/val...")
    try:
        split_start = time.time()
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
        split_time = time.time() - split_start
        print(f"✅ split done in {split_time:.2f} sec")
        print(f"train size: {x_train.shape[0]} | val size: {x_val.shape[0]}")
    except Exception as e:
        print(f"❌ split failed: {e}")

    # tensor conversion
    print("\nconverting to tensors...")
    tensor_start = time.time()
    try:
        print("trying cnn shape...")
        x_train_tensor = torch.tensor(x_train, dtype=torch.float).reshape(-1, 1, 100, 100)
        y_train_tensor = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.long)
        x_val_tensor = torch.tensor(x_val, dtype=torch.float).reshape(-1, 1, 100, 100)
        y_val_tensor = torch.tensor(y_val.values if hasattr(y_val, 'values') else y_val, dtype=torch.long)
        print("✅ cnn reshape success")
    except Exception as e:
        print(f"❌ cnn reshape failed: {e}")
        print("fallback to linear model")
        x_train_tensor = torch.tensor(x_train, dtype=torch.float)
        y_train_tensor = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.long)
        x_val_tensor = torch.tensor(x_val, dtype=torch.float)
        y_val_tensor = torch.tensor(y_val.values if hasattr(y_val, 'values') else y_val, dtype=torch.long)

        print("\nbuilding linear model...")
        model = nn.Sequential(
            nn.Linear(x.shape[1], 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        print("✅ linear model ready")
    else:
        print("\nbuilding cnn model...")
        model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 25 * 25, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
        print("✅ cnn model ready")

    print(f"tensor + model setup took {time.time() - tensor_start:.2f} sec")

    # data loaders
    print("\nsetting up dataloaders...")
    train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=64)
    print(f"train loader: {len(train_loader)} batches")
    print(f"val loader: {len(val_loader)} batches")

    # device config
    print("\nchecking for gpu...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"✅ using device: {device}")
    except:
        device = torch.device("cpu")
        print("❌ fallback to cpu")

    # loss + optimizer
    print("\nsetting up loss/optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("✅ loss + optimizer ready")

    # training loop
    print("\n" + "="*50)
    print(f"training for {num_epochs} epochs")
    print("="*50)

    best_accuracy = 0
    training_start = time.time()
    progress_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nepoch {epoch+1}/{num_epochs}")
        print("   training: ", end="")
        sys.stdout.flush()

        model.train()
        total_loss = 0
        batch_count = len(train_loader)

        for i, (xb, yb) in enumerate(train_loader):
            # progress bar visual (just visual indicator, chatgpt style)
            progress = int((i / batch_count) * len(progress_chars))
            if i % max(1, batch_count // 10) == 0:
                print(progress_chars[min(progress, 7)], end="")
                sys.stdout.flush()

            try:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                print(f"\n❌ error in batch {i}: {e}")
                continue

        # validation
        print("\n   validating: ", end="")
        model.eval()
        correct, total = 0, 0
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
        print(f"\n📈 results — loss: {total_loss/len(train_loader):.4f} | val acc: {acc:.4f} | time: {time.time() - epoch_start:.2f}s")

        if acc > best_accuracy:
            best_accuracy = acc
            print(f"🏆 new best acc: {acc:.4f} — saving model")
            torch.save(model.state_dict(), "model.pth")

    # save final
    try:
        torch.save(model.state_dict(), "model_final.pth")
        print("✅ final model saved.")
    except Exception as e:
        print(f"❌ error saving final model: {e}")

    print("\n" + "="*50)
    print("training done")
    print(f"⏱️ training time: {time.time() - training_start:.2f}s")
    print(f"⏱️ total time: {time.time() - start_time:.2f}s")
    print(f"🏆 best val acc: {best_accuracy:.4f}")
    print("="*50 + "\n")

    return model

if __name__ == "__main__":
    print("🚀 starting script...")
    try:
        user_input = input("enter number of epochs (default 25): ")
        num_epochs = int(user_input) if user_input.strip() else 25
        model = train(num_epochs=num_epochs)
        print("✅ script finished.")
    except Exception as e:
        print(f"❌ script failed with error: {e}")
