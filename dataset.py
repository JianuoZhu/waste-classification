from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms

class WasteDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.dataframe = pd.read_csv(annotations_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path, label = self.dataframe.iloc[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 将标签转换为 Tensor
        match label:
            case "plastic": label = torch.tensor(0)
            case "paper&cardboard": label = torch.tensor(1)
            case "metal": label = torch.tensor(2)
            case "others": label = torch.tensor(3)
            case _: label = torch.tensor(3)

        return image, label

def load_dataset(annotations_file, batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = WasteDataset(annotations_file, transform=transform)
    return dataset

if __name__ == '__main__':
    loader = load_dataset('annotations.csv')
    for img, label in loader:
        print(img.shape, label.shape)
        break
