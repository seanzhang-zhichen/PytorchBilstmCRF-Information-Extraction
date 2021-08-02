import torch
import torch.nn as nn
from modelgraph.BILSTM import BiLSTM
from itertools import zip_longest

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        super(BiLSTM_CRF, self).__init__()
        self.bilstm = BiLSTM(vocab_size, emb_size, hidden_size, out_size)
        self.transition = nn.Parameter(torch.ones(out_size, out_size) * 1 / out_size)

    def forward(self, sents_tensor, lengths):
        emission = self.bilstm(sents_tensor, lengths)
        batch_size, max_len, out_size = emission.size()
        crf_scores = emission.unsqueeze(2).expand(-1, -1, out_size, -1) + self.transition.unsqueeze(0)
        return crf_scores

    def test(self, test_sents_tensor, lengths, tag2id):
        start_id = tag2id['<start>']
        end_id = tag2id['<end>']
        pad = tag2id['<pad>']
        tagset_size = len(tag2id)

        crf_scores =self.forward(test_sents_tensor, lengths)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        B , L , T, _ =crf_scores.size()

        viterbi = torch.zeros(B, L, T).to(device)
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)

        lengths = torch.LongTensor(lengths).to(device)

        for step in range(L):
            batch_size_t =(lengths > step).sum().item()
            if step == 0:
                viterbi[:batch_size_t, step, :] = crf_scores[: batch_size_t, step, start_id, :]
                backpointer[:batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(viterbi[:batch_size_t, step-1, :].unsqueeze(2) + crf_scores[:batch_size_t, step, :, :], dim=1)
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        backpointer = backpointer.view(B, -1)
        tagids = []
        tags_t = None
        for step in range(L-1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L-1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)
                new_in_batch = torch.LongTensor([end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat([tags_t, new_in_batch], dim=0)
                index = torch.ones(batch_size_t).long() * (step *tagset_size)
                index = index.to(device)
                index += offset.long()

            try:
                tags_t = backpointer[:batch_size_t].gather(dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())
        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()

        return tagids


def cal_lstm_crf_loss(crf_scores, targets, tag2id):
    pad_id = tag2id.get('<pad>')
    start_id = tag2id.get('<start>')
    end_id = tag2id.get('<end>')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, max_len = targets.size()
    target_size = len(tag2id)
    mask = (targets != pad_id)
    lengths = mask.sum(dim=1)
    targets = indexed(targets, target_size, start_id)
    targets = targets.masked_select(mask)
    flatten_scores = crf_scores.masked_select(
        mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)
    ).view(-1, target_size*target_size).contiguous()
    golden_scores = flatten_scores.gather(
        dim=1, index=targets.unsqueeze(1)).sum()
    scores_upto_t = torch.zeros(batch_size, target_size).to(device)
    for t in range(max_len):
        batch_size_t = (lengths > t).sum().item()
        if t == 0:
            scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t,
                                           t, start_id, :]
        else:
            scores_upto_t[:batch_size_t] = torch.logsumexp(
                crf_scores[:batch_size_t, t, :, :] +
                scores_upto_t[:batch_size_t].unsqueeze(2),
                dim=1
            )
    all_path_scores = scores_upto_t[:, end_id].sum()
    loss = (all_path_scores - golden_scores) / batch_size
    return loss

def indexed(targets, tagset_size, start_id):
    batch_size, max_len = targets.size()
    for col in range(max_len-1, 0, -1):
        targets[:, col] += (targets[:, col-1] * tagset_size)
    targets[:, 0] += (start_id * tagset_size)
    return targets