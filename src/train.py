import torch.nn as nn
import torch
from src.utils import get_dataloaders
from src.model import WasteClassifier

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print(f"Training on: {device}")

train_loader, val_loader, test_loader = get_dataloaders()

model = WasteClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100

best_val_acc = 0.0

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
      images = images.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)

      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      predicted = outputs.argmax(dim=1)
      correct += (predicted == labels).sum().item()
      total += labels.size(0)

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
      for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        val_correct += (predicted == labels).sum().item()
        val_total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_acc = val_correct / val_total

    if epoch_val_acc > best_val_acc:
    best_val_acc = epoch_val_acc
    torch.save(model.state_dict(), 'waste_classifier.pth')
    print(f"  Model saved with val acc: {best_val_acc:.4f}")
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}")

torch.save(model.state_dict(), 'waste_classifier.pth')