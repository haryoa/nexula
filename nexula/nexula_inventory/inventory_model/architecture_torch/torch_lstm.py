import torch


class TorchLSTMClassification(torch.nn.Module):
    dict_convert = {
        'mean': 1,
        'max': 1,
        'tail_bidirectional': 2,
        'tail': 1
    }

    def __init__(self, num_embedding, embedding_dim, pretrained_vector=None, lstm_unit=100, bidirectional=True,
                 num_label=1, dropout=0.2, num_lstm_layer=1, final_operation=['mean', 'max', 'tail']):
        super().__init__()
        if pretrained_vector is not None:
            self.embedding = torch.nn.Embedding(num_embedding, embedding_dim).from_pretrained(pretrained_vector)
        else:
            self.embedding = torch.nn.Embedding(num_embedding, embedding_dim)
        final_operation = [operation + '_bidirectional' if (bidirectional and operation == 'tail')
                           else operation for operation in final_operation]
        self.bidirectional = bidirectional
        self.lstm = torch.nn.LSTM(embedding_dim, lstm_unit, bidirectional=bidirectional, batch_first=False,
                                  num_layers=num_lstm_layer)
        self.leak_relu = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.final_operation = final_operation
        # calculate input unit of last layer
        last_input_unit = sum([self.dict_convert[operation] * lstm_unit for operation in final_operation])
        last_input_unit = last_input_unit * 2 if bidirectional else last_input_unit
        self.linear_out = torch.nn.Linear(last_input_unit, num_label)

    def forward(self, x):
        # x =  (seq_len, bs)
        nn_out = self.embedding(x)

        # nn_out = (seq_len, bs, emb_dim)
        nn_out, _ = self.lstm(nn_out)

        collection_lstm_op = []

        # Combine all of operations
        # TODO Refactor into new functions
        if 'mean' in self.final_operation:
            nn_out_mean = nn_out.mean(0)
            collection_lstm_op.append(nn_out_mean)
        if 'max' in self.final_operation:
            nn_out_max = nn_out.max(0)[0]
            collection_lstm_op.append(nn_out_max)
        if 'tail' in self.final_operation or 'tail_bidirectional' in self.final_operation:
            right_tail = nn_out[-1, :, :]  # (bs, dim)
            left_tail = nn_out[0, :, :]  # (bs, dim)
            tail_concate = torch.cat((right_tail, left_tail), dim=1)
            collection_lstm_op.append(tail_concate)

        nn_out = torch.cat(collection_lstm_op, dim=1)
        nn_out = self.leak_relu(nn_out)
        nn_out = self.linear_out(nn_out)
        return nn_out
