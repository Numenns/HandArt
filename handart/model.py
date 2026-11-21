from pathlib import Path

import cv2
import torch
from torch import nn
from torch.nn import functional as F


ASSETS = Path(__file__).resolve().parent.parent / "assets"
LABELS = ("2 dedos", "Indice", "Palma", "Pinza")
IMAGE_SIZE = 128


class CNN(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 30 * 30, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(LABELS))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc3(F.relu(self.fc2(x)))


def prepare_frame(rgb):
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    pixels = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0)
    return ((pixels - 0.5) / 0.5).unsqueeze(0)


class GestureModel:
    def __init__(self, path: Path = ASSETS / "modelos" / "regularizada_mejor.pt"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.labels = tuple(checkpoint["clases"])
        if self.labels != LABELS or checkpoint["img_size"] != IMAGE_SIZE:
            raise ValueError("El checkpoint no coincide con las clases o el tamaño de HandArt.")
        self.epoch = checkpoint["epoch"]
        self.network = CNN(dropout=checkpoint["dropout"])
        self.network.load_state_dict(checkpoint["state_dict"], strict=True)
        self.network.eval()

    @torch.inference_mode()
    def predict(self, rgb) -> dict[str, float]:
        logits = self.network(prepare_frame(rgb))
        scores = logits.softmax(dim=1)[0].tolist()
        return dict(zip(self.labels, scores, strict=True))
