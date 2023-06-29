
import_torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class ProductImageClassifier(torch.nn.module):
    def __init__(self):
        resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        print(dir(resnet50))
    def forward(self, X):
        return 

def train(model, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), tr=0.01)
    for epoxh in range(epochs):
        for batch in dataloader:
            features, labels = batch
            predictions = model(features)
            


