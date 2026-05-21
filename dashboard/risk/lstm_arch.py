# risk/lstm_arch.py
# Exact copy of LSTMFeatureExtractor from 02_train_hybrid_model.py
# Must NOT be modified — architecture must match saved weights exactly

import torch.nn as nn

LSTM_HIDDEN  = 64
LSTM_LAYERS  = 2
LSTM_OUT_DIM = 8

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size,
                 hidden=LSTM_HIDDEN,
                 layers=LSTM_LAYERS,
                 out_dim=LSTM_OUT_DIM):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden,
            num_layers=layers, dropout=0.3, batch_first=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, 1),
        )
        self.compress = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, out_dim), nn.Tanh(),
        )
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x):
        out, _   = self.lstm(x)
        attn_w   = __import__('torch').softmax(self.attn(out), dim=1)
        ctx      = (attn_w * out).sum(dim=1)
        features = self.compress(ctx)
        logits   = self.head(features).squeeze(-1)
        return features, logits