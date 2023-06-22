import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer


class TransformerModel(nn.Module):
    def __init__(self, input_dim=12, output_dim=2, hidden_dim=512, num_layers=8, num_heads=8):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        embedded = self.embedding(inputs.float())
        embedded = embedded.permute(1, 0, 2)  # Reshape to (seq_len, batch_size, hidden_dim)
        outputs = self.transformer(embedded, embedded)
        outputs = outputs.permute(1, 0, 2)  # Reshape back to (batch_size, seq_len, hidden_dim)
        logits = self.fc(outputs[:, -1, :])  # Take the last output token for classification
        return logits



