import torch.nn as nn
 
 
class PEModel(nn.Module):
    def __init__(self):
        super(PEModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(12, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 2)
        )
 
    def forward(self, inputs):
        output = self.classifier(inputs.float())
 
        return output