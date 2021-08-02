import time
from collections import Counter
import pickle

from operate_bilstm import BiLSTM_operator
from evaluating import Metrics
from utils import save_model


def bilstm_train_and_eval(train_data,dev_data,test_data,word2id,tag2id,crf=True,remove_0=False):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)

    bilstm_operator = BiLSTM_operator(vocab_size,out_size,crf=crf)
    model_name = "bilstm_crf" if crf else "bilstm"

    print("start to train the {} ...".format(model_name))
    bilstm_operator.train(train_word_lists,train_tag_lists,dev_word_lists,dev_tag_lists,word2id,tag2id)
    save_model(bilstm_operator, "./ckpts/" + model_name + ".pkl")

    print("训练完毕,共用时{}秒.".format(int(time.time() - start)))
    print("评估{}模型中...".format(model_name))
    pred_tag_lists, test_tag_lists = bilstm_operator.test(
        test_word_lists, test_tag_lists, word2id, tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_0=remove_0)
    dtype = 'Bi_LSTM+CRF' if crf else 'Bi_LSTM'
    metrics.report_scores(dtype=dtype)

    return pred_tag_lists