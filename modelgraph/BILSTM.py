import torch
import torch.nn.functional as F
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size, dropout=0.1):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc =nn.Linear(2*hidden_size, out_size)
        self.dropout =nn.Dropout(dropout)

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))
        emb = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True)
        emb, _ = self.bilstm(emb)
        # print("shape of x: ")
        # print(x.shape)
        emb, _ = nn.utils.rnn.pad_packed_sequence(emb, batch_first=True, padding_value=0., total_length=x.shape[1])
        scores = self.fc(emb)

        return scores

    def test(self, x, lengths, _):
        logits = self.forward(x, lengths)
        _, batch_tagids = torch.max(logits, dim=2)
        return batch_tagids

def cal_loss(logits, targets, tag2id):
    PAD = tag2id.get('<pad>')
    assert PAD is not None
    mask = (targets != PAD)
    targets = targets[mask]
    out_size = logits.size(2)
    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1, -1, out_size)
    ).contiguous().view(-1, out_size)
    assert logits.size(0) == targets.size(0)
    loss = F.cross_entropy(logits, targets)
    return loss