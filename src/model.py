from torch import nn
import config


class ReviewAnalyzeModel(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM, padding_idx=padding_idx)
        self.lstm = nn.LSTM(config.EMBEDDING_DIM, config.HIDDEN_DIM, batch_first=True)
        self.linear = nn.Linear(config.HIDDEN_DIM, 1)

    def forward(self, x):
        # x.shape is [batch_size, seq_len]
        embedded = self.embedding(x)
        # embedded.shape is [batch_size, seq_len, embed_dim]
        output, _ = self.lstm(embedded)
        # output.shape is [batch_size, seq_len, hidden_dim]
        last_hidden = output[:, -1, :]
        # last_hidden.shape is [batch_size, hidden_dim]
        output = self.linear(last_hidden).squeeze(1)
        # output.shape is [batch_size]
        return output
