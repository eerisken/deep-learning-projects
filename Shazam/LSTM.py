import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioMappingNet(nn.Module):
    def __init__(self, n_mels=64, embedding_dim=128, lstm_hidden=128, lstm_layers=1):
        super(AudioMappingNet, self).__init__()
        
        # ------------------------------
        # 1️⃣ CNN feature extractor
        # ------------------------------
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=(1,1)),  # input: (B,1,n_mels,T)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),  # downsample (n_mels//2, T//2)

            nn.Conv2d(32, 64, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),  # downsample further

            nn.Conv2d(64, 128, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # ------------------------------
        # 2️⃣ BiLSTM for temporal modeling
        # ------------------------------
        # After CNN, we collapse the mel-dim, keep time as sequence
        self.lstm = nn.LSTM(
            input_size=128*(n_mels//4),  # depends on pooling
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        # ------------------------------
        # 3️⃣ Fully connected embedding
        # ------------------------------
        self.fc = nn.Linear(2*lstm_hidden, embedding_dim)  # BiLSTM -> embedding

    def forward(self, x):
        """
        x: (B, 1, n_mels, T) log-mel spectrogram
        """
        B, C, H, W = x.size()
        x = self.cnn(x)  # (B, 128, H', W')
        x = x.permute(0, 3, 1, 2)  # (B, W', C, H')
        x = x.contiguous().view(B, x.size(1), -1)  # flatten mel+channels -> seq_len x features

        lstm_out, _ = self.lstm(x)  # (B, seq_len, 2*lstm_hidden)
        # Take mean over time dimension
        embedding = lstm_out.mean(dim=1)
        embedding = self.fc(embedding)  # final embedding
        embedding = F.normalize(embedding, p=2, dim=1)  # optional: normalize

        return embedding

