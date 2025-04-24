import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import sys
import argparse

def test(x_csv, t_csv):
    print("\ntesting started")
    print("=" * 40)
    start_time = time.time()

    # Load test data
    print("loading test data...")
    try:
        x_test = pd.read_csv(x_csv).values.astype("float32") / 255.0
        t_test = pd.read_csv(t_csv).squeeze().values
        print(f"✅ test data loaded: {len(x_test)} samples")
    except Exception as e:
        print(f"❌ error loading test data: {e}")
        return None

    # Convert to tensors and reshape for CNN
    try:
        x_test_tensor = torch.tensor(x_test, dtype=torch.float).reshape(-1, 1, 100, 100)
        t_test_tensor = torch.tensor(t_test, dtype=torch.long)

        model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
        print("✅ cnn model structure loaded")
    except Exception as e:
        print(f"⚠ failed to reshape or build model: {e}")
        return None

    # Dataloader
    print("creating test dataloader...")
    test_loader = DataLoader(TensorDataset(x_test_tensor, t_test_tensor), batch_size=64)
    print(f"✅ test loader ready — {len(test_loader)} batches")

    # Load weights
    print("loading model weights...")
    try:
        model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
        model.eval()
        print("✅ weights loaded")
    except Exception as e:
        print(f"❌ couldn't load model weights: {e}")
        return None

    # Evaluation
    print("\nevaluating model...")
    print("   computing predictions: ", end="")
    correct = 0
    total = 0
    all_preds = []
    progress_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

    with torch.no_grad():
        for i, (xb, yb) in enumerate(test_loader):
            progress = int((i / len(test_loader)) * len(progress_chars))
            if i % max(1, len(test_loader) // 10) == 0:
                print(progress_chars[min(progress, 7)], end="")

            out = model(xb)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.tolist())
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    accuracy = correct / total
    print(f"\ntest accuracy: {accuracy:.4f} ({correct}/{total})")

    # Per-class accuracy
    try:
        print("\nchecking class-specific performance...")
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

        print("\nper-class accuracy:")
        for i in range(10):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                print(f"   class {i}: {acc:.4f} ({class_correct[i]}/{class_total[i]})")
    except Exception as e:
        print(f"\\couldn't compute per-class metrics: {e}")

    print(f"\n✅ total test time: {time.time() - start_time:.2f}s")
    print("=" * 40)

    return all_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CNN model on Greek handwriting data")
    parser.add_argument("x_csv", type=str, help="Path to x_test CSV")
    parser.add_argument("t_csv", type=str, help="Path to t_test CSV (labels)")
    args = parser.parse_args()
    test(args.x_csv, args.t_csv)
