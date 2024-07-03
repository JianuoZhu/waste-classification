from transformers import ViTForImageClassification
import torch.nn as nn

class ClsHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClsHead, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, features):
        output = self.linear(features)
        return output

def create_model():
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224'
    )
    model.classifier = ClsHead(model.config.hidden_size, 4)
    return model

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

if __name__ == '__main__':
    print("Loading model...")
    model = create_model()
    model.classifier = ClsHead(model.config.hidden_size, 4)
    print(model)
    total_params = count_parameters(model)
    print(f"Total number of trainable parameters: {total_params}")
