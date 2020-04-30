import torch


class TorchCNN1DClassification(torch.nn.Module):

    def __init__(self, num_embedding, embedding_dim, pretrained_vector=None,
                 n_filters=100, filter_sizes=[3, 4, 5], num_label=1, dropout=0.2):
        super().__init__()
        if pretrained_vector is not None:
            self.embedding = torch.nn.Embedding(num_embedding, embedding_dim).from_pretrained(pretrained_vector)
        else:
            self.embedding = torch.nn.Embedding(num_embedding, embedding_dim)
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(in_channels=embedding_dim,
                            out_channels=n_filters,
                            kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = torch.nn.Linear(len(filter_sizes) * n_filters, num_label)
        self.dropout = torch.nn.Dropout(dropout)
        self.num_label = num_label

    def forward(self, x):
        from torch.nn import functional as F
        # x =  (seq_len, bs)
        nn_out = self.embedding(x)

        # nn_out = (seq_len, bs, emb_size)
        nn_out = nn_out.permute(1, 2, 0)

        # nn_out = (bs, emb_size, seq_len)
        nn_out = [F.relu(conv(nn_out)) for conv in self.convs]

        # nn_out_n = (bs, emb_size, seq_len - filter_size[n] + 1)
        nn_out = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in nn_out]

        # nn_out_n = (bs, emb_size)
        nn_out = self.dropout(torch.cat(nn_out, dim=1))

        # nn_out = (bs, num_label)
        nn_out = self.fc(nn_out)

        return nn_out
