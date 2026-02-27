from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)  # biLSTM

    def forward(self, lstm_out):
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # [B,T,1]
        context = torch.sum(attn_weights * lstm_out, dim=1)        # [B,2H]
        return context, attn_weights

class UltimateLSTMClassifier(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 256, num_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.attention = TemporalAttention(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout / 2),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        context, _ = self.attention(out)
        return self.head(context)

class LSTMTemporalClassifier:
    def __init__(self, weights_path: str, device: str, feat_dim: int = 8):
        self.device = device
        self.model = UltimateLSTMClassifier(feat_dim=feat_dim).to(device)
        state = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict_label(self, window_feats: np.ndarray) -> str:
        """
        window_feats: [T, 8] float32
        Returns "Fight" if class=1 else "No Fight"
        """
        xb = torch.tensor(window_feats[None, ...], dtype=torch.float32, device=self.device)
        logits = self.model(xb)
        pred = int(torch.argmax(logits, dim=1).item())
        return "Fight" if pred == 1 else "No Fight"