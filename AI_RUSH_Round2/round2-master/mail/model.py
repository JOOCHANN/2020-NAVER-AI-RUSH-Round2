import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.vocab_size = args.vocab_size
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.layers = args.layers
        self.dropout_rate = args.dropout

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.title_rnn = nn.GRU(self.embedding_dim, self.hidden_dim, self.layers, batch_first=True)
        self.content_rnn = nn.GRU(self.embedding_dim, self.hidden_dim, self.layers, batch_first=True)
        self.fc = nn.Linear(2 * self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, title, content):
        title = self.embedding(title)
        content = self.embedding(content)

        title = self.dropout(title)
        content = self.dropout(content)

        title, _ = self.title_rnn(title)
        content, _ = self.content_rnn(content)

        title = title[:, -1, :]
        content = content[:, -1, :]

        title = self.dropout(title)
        content = self.dropout(content)

        x = torch.cat((title, content), 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
