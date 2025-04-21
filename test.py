import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time

def test():
    print("\n🧪 TESTING STARTED\n" + "="*40)
    start_time = time.time()

    # Load test data
    print("📂 Loading test data...")
    try:
        x_test = pd.read_csv("x_train_project.csv").values.astype("float32")
        t_test = pd.read_csv("t_train_project.csv").squeeze().values
        print(f"✅ Test data loaded: {len(x_test)} samples")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None

    # Normalize and reshape test data
    print("🔄 Preprocessing test data...")
    x_test /= 255.0
    
    try:
        # Try to reshape for CNN
        x_test_tensor = torch.tensor(x_test).reshape(-1, 1, 100, 100)
        t_test_tensor = torch.tensor(t_test, dtype=torch.long)
        
        # Create the same CNN model structure used in training
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
        print("✅ Using CNN model architecture")
    except Exception as e:
        print(f"⚠️ Could not reshape for CNN: {e}")
        print("⚠️ Falling back to linear model")
        
        # Fall back to linear model if reshaping fails
        x_test_tensor = torch.tensor(x_test)
        t_test_tensor = torch.tensor(t_test, dtype=torch.long)
        
        # Create the same linear model structure used in training
        model = nn.Sequential(
            nn.Linear(x_test.shape[1], 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        print("✅ Using linear model architecture")

    # Create DataLoader
    print("🔄 Creating test data loader...")
    test_loader = DataLoader(TensorDataset(x_test_tensor, t_test_tensor), batch_size=64)
    print(f"✅ Created test loader with {len(test_loader)} batches")

    # Load model weights
    print("📂 Loading trained model weights...")
    try:
        model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
        model.eval()
        print("✅ Model weights loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model weights: {e}")
        print("   This could happen if the model architecture doesn't match the saved weights.")
        print("   Make sure you're using the same architecture in training and testing.")
        return None

    # Evaluate
    print("\n📊 Evaluating model...")
    print("   Computing predictions: ", end="")
    correct = 0
    total = 0
    all_preds = []
    
    # Progress indicator characters
    progress_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

    with torch.no_grad():
        for i, (xb, yb) in enumerate(test_loader):
            # Update progress indicator
            progress = int((i / len(test_loader)) * 8)
            if i % max(1, len(test_loader) // 10) == 0:
                print(progress_chars[min(progress, 7)], end="")
                
            out = model(xb)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.tolist())
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    accuracy = correct / total
    print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({correct}/{total} correct)")
    
    # Display class-specific performance if possible
    try:
        classes = list(range(10))  # Assumes 10 classes (0-9)
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
        
        print("\n📊 Class-specific accuracy:")
        for i in range(10):
            if class_total[i] > 0:
                accuracy = class_correct[i] / class_total[i]
                print(f"   Class {i}: {accuracy:.4f} ({class_correct[i]}/{class_total[i]})")
    except Exception as e:
        print(f"⚠️ Could not compute class-specific performance: {e}")

    print(f"\n✅ Total testing time: {time.time() - start_time:.2f}s")
    print("="*40)

    return all_preds

if __name__ == "__main__":
    test()