import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()

class AnimalNetwork(nn.Module):
    def __init__(self, input_channels: int, n_classes: int) -> None:
        super(AnimalNetwork, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Adjust the input channels based on the number of color channels in your animal images (3 for RGB)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)

        # Adjust the input dimension calculation based on the image size (64x64)
        out_dim = self.calc_out_dim(64, kernel_size=5, stride=1, padding=2)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=out_dim * out_dim * 256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc_output = nn.Linear(in_features=256, out_features=n_classes)

        self.to(self.device)

    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2 * padding) / stride) + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc_output(x)
        proba = F.softmax(logits, dim=1)
        return logits, proba

    def predict(self, x):
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        models_path = file_path / 'models' / model_name
        torch.save(self.state_dict(), models_path)

    def load_model(self, model_name: str):
        models_path = file_path / 'models' / model_name
        self.load_state_dict(torch.load(models_path))
