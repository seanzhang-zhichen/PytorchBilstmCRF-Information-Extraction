from data import build_corpus
from evaluate import bilstm_train_and_eval
from utils import extend_maps,prepocess_data_for_lstmcrf, save_obj, load_obj


print("读取数据中...")
train_word_lists,train_tag_lists,word2id,tag2id = build_corpus("train")
dev_word_lists,dev_tag_lists = build_corpus("dev",make_vocab=False)
test_word_lists,test_tag_lists = build_corpus("test",make_vocab=False)


print("正在训练评估Bi-LSTM+CRF模型...")
crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
save_obj(crf_word2id, 'crf_word2id')
save_obj(crf_tag2id, 'crf_tag2id')
# import os
# #保存word2id
# if os.path.exists('data/crf_word2id.pkl'):
#     crf_word2id = load_obj('crf_word2id')
# else:
#     save_obj(crf_word2id, 'crf_word2id')
#
# #保存tag2id
# if os.path.exists('data/crf_tag2id.pkl'):
#     crf_tag2id = load_obj('crf_tag2id')
# else:
#     save_obj(crf_tag2id, 'crf_tag2id')


print(' '.join([i[0] for i in crf_tag2id.items()]))

train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(
    train_word_lists, train_tag_lists
)


dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(
    dev_word_lists, dev_tag_lists
)
test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
    test_word_lists, test_tag_lists, test=True
)


lstmcrf_pred = bilstm_train_and_eval(
    (train_word_lists, train_tag_lists),
    (dev_word_lists, dev_tag_lists),
    (test_word_lists, test_tag_lists),
    crf_word2id, crf_tag2id
)