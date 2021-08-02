from codecs import open
import os


def build_corpus(split, make_vocab=True, data_dir='./data'):
    assert split.lower() in ["train","dev","test"]
    word_lists = []
    tag_lists = []
    with open(os.path.join(data_dir,split+".char"),'r',encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word,tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists,tag_lists,word2id,tag2id
    else:
        return word_lists,tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps