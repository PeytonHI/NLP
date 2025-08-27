import torch
import torch.nn as nn
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(BiLSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x should be of type torch.long (integer indices)
        embedded = self.embedding(x)  # Convert indices to embeddings
        # embedded will now be of type torch.float32
        lstm_out, _ = self.lstm(embedded)
        return lstm_out
