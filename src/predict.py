import torch
import torchvision.transforms as transforms
from PIL import Image
from src.model import WasteClassifier, CLASS_NAMES
import argparse

device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()

    model = WasteClassifier().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    print(predict(args.image, model, device))