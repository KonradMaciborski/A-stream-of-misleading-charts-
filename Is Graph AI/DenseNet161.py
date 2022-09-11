import torch.nn as nn
import torchvision


class DenseNet161(nn.Module):
    def __init__(self, output_features, num_units=512, drop=0.5,
                 num_units1=512, drop1=0.5):
        super().__init__()
        model = torchvision.models.densenet161(pretrained=True)
        n_inputs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(n_inputs, num_units),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(num_units, num_units1),
            nn.ReLU(),
            nn.Dropout(p=drop1),
            nn.Linear(num_units1, output_features))
        self.model = model

    def forward(self, x):
        return self.model(x)
