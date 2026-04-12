import torch
import torchvision.transforms as transforms
from PIL import Image
from src.model import WasteClassifier, CLASS_NAMES

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image_path, model, device):

  model.eval()

  image = Image.open(image_path).convert('RGB')
  image = transform(image)
  image = image.unsqueeze(0)
  image = image.to(device)
  with torch.no_grad():
    output = model(image)
    prediction = output.argmax(dim=1).item()

  return CLASS_NAMES[prediction]