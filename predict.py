#%%
import torch
import pickle
from utils import load_obj, tensorized


def predict(model, text):
    text_list = list(text)
    text_list.append("<end>")
    text_list = [text_list]
    crf_word2id = load_obj('crf_word2id')
    crf_tag2id = load_obj('crf_tag2id')
    # vocab_size = len(crf_word2id)
    # out_size = len(crf_tag2id)
    pred_tag_lists = model.predict(text_list, crf_word2id, crf_tag2id)
    return pred_tag_lists[0]


def result_process(text_list, tag_list):
    tuple_result = zip(text_list, tag_list)
    sent_out = []
    tags_out = []
    outputs = []
    words = ""
    for s, t in tuple_result:
        if t.startswith('B-') or t == 'O':
            if len(words):
                sent_out.append(words)
                # print(sent_out)
            if t != 'O':
                tags_out.append(t.split('-')[1])
            else:
                tags_out.append(t)
            words = s
            # print(words)
        else:
            words += s
        # %%
    if len(sent_out) < len(tags_out):
        sent_out.append(words)
    outputs.append(''.join([str((s, t)) for s, t in zip(sent_out, tags_out)]))
    return outputs, [*zip(sent_out, tags_out)]



#%%
if __name__ == '__main__':

    modelpath = './ckpts/bilstm_crf.pkl'
    f = open(modelpath, 'rb')
    s = f.read()
    model = pickle.loads(s)

    text = '法外狂徒张三丰，身份证号362502190211032345'
    tag_res = predict(model, text)
    result, tuple_re = result_process(list(text), tag_res)

    print(text)
    # #%%
    #print(tuple_re)
    # print(result)
    result = []
    tag = []
    for s,t in tuple_re:
        if t !='O':
            result.append(s)
            tag.append(t)
    print([*zip(result, tag)])




