from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
from model import ClsHead, create_model, count_parameters
import time

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Loading model...")
    model = create_model().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    print("Total number of trainable parameters:", count_parameters(model))
    img_path = 'dataset-resized/metal/metal1.jpg'
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    output = model(image)
    output = output.logits
    _, predicted = output.max(1)
    print(f"the predicted class is: {predicted}")
    time.sleep(5)

main()
