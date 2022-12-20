import json
import spacy
from tqdm import tqdm
import random
import argparse
import numpy as np


parser = argparse.ArgumentParser(description="")

parser.add_argument("--sent_num", type=int, default=9)
parser.add_argument('--replace_method', type=str, default="word2vec")

args = parser.parse_args()


def hard_ground(sent):
    sent = sent.lower()
    doc = nlp(sent)
    res = []
    for t in doc:
        res.append(t.lemma_)
    return res


def reconstruct_sent(e0, e1, dictionary, corpora, method_name, model):
    res = []
    if e0 in dictionary:
        for idx in dictionary[e0]:
            phrase_list = corpora[idx][:-1]
            if e1 not in phrase_list:
                pos_list = [*range(len(phrase_list))]
                pos_list.remove(phrase_list.index(e0))
                if len(pos_list) == 0:
                    continue
                if method_name == "random":
                    pos = random.choice(pos_list)
                elif method_name == "word2vec":
                    scores = [model.wv.similarity(e0, phrase_list[p]) for p in pos_list]
                    pos = pos_list[np.argmax(scores)]
                phrase_list[pos] = e1
            sent = " ".join(phrase_list)
            if sent[-1] != ".":
                sent = sent + "."
            if sent[-2:] == " .":
                sent = sent[:-2] + "."
            res.append(sent)
    return res


if __name__ == "__main__":
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])

    with open("data/external_corpora.txt", "r") as f:
        corpora = [line for line in f.readlines()]

    # seg_corpora = []
    # for sent in tqdm(corpora):
    #     seg_corpora.append(hard_ground(sent))
    # with open("old/corpora_segmented.json", "w") as f:
    #     json.dump(seg_corpora, f)
    with open("old/corpora_segmented.json", "r") as f:
        seg_corpora = json.load(f) 
    assert len(corpora) == len(seg_corpora)

    if args.replace_method == "word2vec":
        import gensim
        model = gensim.models.Word2Vec(seg_corpora, min_count = 1, vector_size = 100, window = 5, sg = 1)
    else:
        model = None

    cor_dictionary = {}
    for idx, res in enumerate(seg_corpora):
        for i in res:
            if i not in cor_dictionary:
                cor_dictionary[i] = []
            cor_dictionary[i].append(idx)


    for type in ("dev", "test", "train"):
        inputs = []

        with open("data/DimonGen" + type + ".json", "r") as f:
            inputs += [json.loads(line) for line in f.readlines()]

        input_entity_pairs = [e["inputs"] for e in inputs]

        input_dictionary = {}

        for i, inp in enumerate(input_entity_pairs):
            e0 = hard_ground(inp[0])[0]
            e1 = hard_ground(inp[1])[0]

            if e0 not in input_dictionary:
                input_dictionary[e0] = {}
            input_dictionary[e0][e1] = i

            if e1 not in input_dictionary:
                input_dictionary[e1] = {}
            input_dictionary[e1][e0] = i

        matching_sents = [[] for i in range(len(input_entity_pairs))]

        for res, sent in tqdm(zip(seg_corpora, corpora), total=len(corpora)):
            for i in range(len(res)):
                if res[i] in input_dictionary:
                    for j in range(i + 1, len(res)):
                        if res[j] in input_dictionary[res[i]]:
                            idx = input_dictionary[res[i]][res[j]]
                            sent = sent.lower().strip()
                            matching_sents[idx].append(sent)

        sent_num = args.sent_num
        for i in tqdm(range(len(matching_sents))):
            temp = set(matching_sents[i])
            matching_sents[i] = []
            for ex_i in temp:
                flag = True
                for ex_j in matching_sents[i]:
                    if ex_i[:len(ex_i) * 9 // 10] in ex_j:
                        flag = False
                        break
                if flag:
                    matching_sents[i].append(ex_i)

            if len(matching_sents[i]) < sent_num:
                e0 = hard_ground(input_entity_pairs[i][0])[0]
                e1 = hard_ground(input_entity_pairs[i][1])[0]
                pool = set()
                pool.update(reconstruct_sent(e0, e1, cor_dictionary, seg_corpora, args.replace_method, model))
                pool.update(reconstruct_sent(e1, e0, cor_dictionary, seg_corpora, args.replace_method, model))

                diff = sent_num - len(matching_sents[i])
                k = min(diff, len(pool))
                add_sents = random.sample(list(pool), k)
                matching_sents[i] += add_sents
            
        with open("retrieval/output/" + type + "_stage1.json", "w") as f:
            json.dump(matching_sents, f, indent=0)
