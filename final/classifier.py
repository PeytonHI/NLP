import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, hidden_dim, output_dim=3):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, claim_encoding, evidence_encoding):
        combined_features = torch.cat([
            claim_encoding,
            evidence_encoding,
            claim_encoding * evidence_encoding,
            torch.abs(claim_encoding - evidence_encoding)
        ], dim=-1)
        return self.fc(combined_features)
