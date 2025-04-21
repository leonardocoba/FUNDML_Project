import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time
import sys

def train():
    print("\n" + "="*50)
    print("TRAINING PROCESS STARTING")
    print("="*50)
    start_time = time.time()
    
    # Load data
    print("\n📂 Loading data files...")
    try:
        loading_start = time.time()
        x = pd.read_csv("x_train_project.csv").values
        y = pd.read_csv("t_train_project.csv").squeeze()
        loading_time = time.time() - loading_start
        print(f"✅ Data loaded successfully in {loading_time:.2f} seconds")
        print(f"📊 x shape: {x.shape}")
        print(f"📊 y unique: {pd.Series(y).value_counts()}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Normalize input
    print("\n🔄 Normalizing input data...")
    x = x / 255.0
    print("✅ Normalization complete")
    
    # Split train/val
    print("\n🔀 Splitting data into train/validation sets...")
    split_start = time.time()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    split_time = time.time() - split_start
    print(f"✅ Split complete in {split_time:.2f} seconds")
    print(f"📊 Training set size: {x_train.shape[0]}")
    print(f"📊 Validation set size: {x_val.shape[0]}")
    
    # Convert to PyTorch tensors
    print("\n🔄 Converting data to PyTorch tensors...")
    tensor_start = time.time()
    try:
        # First try with reshaping for CNN
        print("   Attempting to reshape data for CNN...")
        x_train_tensor = torch.tensor(x_train, dtype=torch.float).reshape(-1, 1, 100, 100)
        y_train_tensor = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.long)
        x_val_tensor = torch.tensor(x_val, dtype=torch.float).reshape(-1, 1, 100, 100)
        y_val_tensor = torch.tensor(y_val.values if hasattr(y_val, 'values') else y_val, dtype=torch.long)
        print("✅ Successfully reshaped data for CNN architecture")
    except Exception as e:
        print(f"⚠️ Warning: Could not reshape data for CNN: {e}")
        print("⚠️ Falling back to linear model with flattened data")
        # Fall back to original approach with flattened data
        x_train_tensor = torch.tensor(x_train, dtype=torch.float)
        y_train_tensor = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.long)
        x_val_tensor = torch.tensor(x_val, dtype=torch.float)
        y_val_tensor = torch.tensor(y_val.values if hasattr(y_val, 'values') else y_val, dtype=torch.long)
        
        print("\n🏗️ Building linear model architecture...")
        # Define simple model (similar to original)
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
        print("✅ Linear model created")
    else:
        print("\n🏗️ Building CNN architecture...")
        # Define CNN model if reshaping worked
        model = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Flatten and fully connected layers
            nn.Flatten(),
            nn.Linear(64 * 25 * 25, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
        print("✅ CNN model created")
    
    tensor_time = time.time() - tensor_start
    print(f"✅ Tensor conversion and model initialization complete in {tensor_time:.2f} seconds")
    
    # Wrap in DataLoader
    print("\n🔄 Creating data loaders...")
    train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val_tensor, y_val_tensor), batch_size=64)
    print(f"✅ Created train loader with {len(train_loader)} batches")
    print(f"✅ Created validation loader with {len(val_loader)} batches")
    
    # Try using GPU, fallback to CPU if not available
    print("\n🖥️ Setting up computation device...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"✅ Using device: {device}")
    except:
        device = torch.device("cpu")
        print("⚠️ Failed to use GPU, falling back to CPU")
    
    # Loss and optimizer
    print("\n🔄 Setting up loss function and optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("✅ Using CrossEntropyLoss and Adam optimizer with lr=0.001")
    
    # Training loop
    num_epochs = 50
    print("\n" + "="*50)
    print(f"🚀 STARTING TRAINING FOR {num_epochs} EPOCHS")
    print("="*50)
    
    best_accuracy = 0
    training_start = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\n📊 Epoch {epoch+1}/{num_epochs}")
        
        # Progress bar characters
        progress_chars = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        
        # Training
        print("   Training: ", end="")
        sys.stdout.flush()
        model.train()
        total_loss = 0
        batch_count = len(train_loader)
        
        for i, (xb, yb) in enumerate(train_loader):
            # Update progress bar
            progress = int((i / batch_count) * 8)
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
                print(f"\n❌ Error during training batch {i}: {e}")
                continue
        
        # Validation
        print("\n   Validating: ", end="")
        sys.stdout.flush()
        model.eval()
        correct, total = 0, 0
        try:
            with torch.no_grad():
                for i, (xb, yb) in enumerate(val_loader):
                    # Update progress bar
                    progress = int((i / len(val_loader)) * 8)
                    if i % max(1, len(val_loader) // 10) == 0:
                        print(progress_chars[min(progress, 7)], end="")
                        sys.stdout.flush()
                        
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    preds = torch.argmax(out, dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            
            accuracy = correct / total
            epoch_time = time.time() - epoch_start
            
            print(f"\n   📈 Results - Loss: {total_loss/len(train_loader):.4f} - Val Accuracy: {accuracy:.4f} - Time: {epoch_time:.2f}s")
            
            # Save if best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                print(f"   🏆 New best accuracy: {accuracy:.4f}")
                torch.save(model.state_dict(), "model.pth")
                print("   💾 Model checkpoint saved")
            
        except Exception as e:
            print(f"\n❌ Error during validation: {e}")
    
    # Save final model
    try:
        torch.save(model.state_dict(), "model_final.pth")
        print("\n💾 Final model saved to model_final.pth")
    except Exception as e:
        print(f"\n❌ Error saving final model: {e}")
    
    total_training_time = time.time() - training_start
    total_time = time.time() - start_time
    
    print("\n" + "="*50)
    print("🏁 TRAINING COMPLETE")
    print(f"⏱️ Training time: {total_training_time:.2f} seconds")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    print(f"🏆 Best validation accuracy: {best_accuracy:.4f}")
    print("="*50 + "\n")
    
    return model

if __name__ == "__main__":
    print("🚀 Script starting...")
    try:
        model = train()
        print("✅ Script completed successfully!")
    except Exception as e:
        print(f"❌ Script failed with error: {e}")