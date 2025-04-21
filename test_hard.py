
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import sys
import numpy as np

def test_hard():
    print("\n🧪 HARD TESTING STARTED")
    print("=" * 50)
    start_time = time.time()

    # Load hard test data
    print("\nloading hard test data...")
    try:
        x_test = pd.read_csv("x_test_hard.csv").values.astype("float32")
        t_test = pd.read_csv("t_test_hard.csv").squeeze().values
        print(f"✅ loaded {len(x_test)} samples")
    except Exception as e:
        print(f"❌ failed to load test data: {e}")
        return None

    # Normalize and reshape
    print("\npreprocessing...")
    try:
        x_test /= 255.0
        x_test_tensor = torch.tensor(x_test).reshape(-1, 1, 100, 100)
        t_test_tensor = torch.tensor(t_test, dtype=torch.long)
        print("✅ normalization complete")
    except Exception as e:
        print(f"❌ preprocessing failed: {e}")
        return None

    # CNN model architecture (same as training)
    print("\nbuilding CNN model...")
    try:
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
    except Exception as e:
        print(f"❌ failed to build model: {e}")
        return None

    # Load weights
    print("\nloading model weights...")
    try:
        model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
        model.eval()
        print("✅ model weights loaded")
    except Exception as e:
        print(f"❌ could not load weights: {e}")
        return None

    # Dataloader
    print("\ncreating test dataloader...")
    try:
        test_loader = DataLoader(TensorDataset(x_test_tensor, t_test_tensor), batch_size=64)
        print(f"✅ test loader ready: {len(test_loader)} batches")
    except Exception as e:
        print(f"❌ dataloader error: {e}")
        return None

    # Evaluation
    print("\nevaluating model...")
    correct = 0
    total = 0
    all_preds = []

    progress_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

    with torch.no_grad():
        for i, (xb, yb) in enumerate(test_loader):
            progress = int((i / len(test_loader)) * len(progress_chars))
            if i % max(1, len(test_loader) // 10) == 0:
                print(progress_chars[min(progress, 7)], end="")
                sys.stdout.flush()

            out = model(xb)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.tolist())
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    accuracy = correct / total
    print(f"\n🎯 test accuracy: {accuracy:.4f} ({correct}/{total} correct)")

    # Class-level accuracy
    try:
        print("\n📊 class-level results:")
        class_correct = [0] * 10
        class_total = [0] * 10

        with torch.no_grad():
            for xb, yb in test_loader:
                out = model(xb)
                preds = torch.argmax(out, dim=1)
                c = (preds == yb).squeeze()
                for i, label in enumerate(yb):
                    label = label.item()
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                print(f"   class {i}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")
    except Exception as e:
        print(f"❌ class-level metrics failed: {e}")

    print(f"\n✅ test done in {time.time() - start_time:.2f}s")
    print("=" * 50)
    return all_preds

if __name__ == "__main__":
    test_hard()
