# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from tqdm.auto import tqdm
import scipy.sparse as sp
from math import log
import numpy as np


def ordered_word_pair(a, b):
    if a > b:
        return b, a
    else:
        return a, b


def get_adj(lda, lda_topic_num, lda_dict, tokenize_sentences, train_size, word_id_map, word_list, args):
    window_size = 20
    total_W = 0
    word_occurrence = {}
    word_pair_occurrence = {}

    node_size = train_size + lda_topic_num

    # vocab_length = len(word_list)

    def update_word_and_word_pair_occurrence(q):
        unique_q = list(set(q))
        for i in unique_q:
            try:
                word_occurrence[i] += 1
            except:
                word_occurrence[i] = 1
        for i in range(len(unique_q)):
            for j in range(i + 1, len(unique_q)):
                word1 = unique_q[i]
                word2 = unique_q[j]
                word1, word2 = ordered_word_pair(word1, word2)
                try:
                    word_pair_occurrence[(word1, word2)] += 1
                except:
                    word_pair_occurrence[(word1, word2)] = 1

    if not args.easy_copy:
        print("Calculating PMI")
    for ind in range(train_size):
        words = tokenize_sentences[ind]

        q = []
        # push the first (window_size) words into a queue
        for i in range(min(window_size, len(words))):
            q += [word_id_map[words[i]]]
        # update the total number of the sliding windows
        total_W += 1
        # update the number of sliding windows that contain each word and word pair
        update_word_and_word_pair_occurrence(q)

        now_next_word_index = window_size
        # pop the first word out and let the next word in, keep doing this until the end of the document
        while now_next_word_index < len(words):
            q.pop(0)
            q += [word_id_map[words[now_next_word_index]]]
            now_next_word_index += 1
            # update the total number of the sliding windows
            total_W += 1
            # update the number of sliding windows that contain each word and word pair
            update_word_and_word_pair_occurrence(q)

    # TODO: w2w的关系要彻底取消。
    # 但是要保留PMI信息，因为topic的edge要构建于PMI之上。

    row = []
    col = []
    weight = []
    extra_word_list = []
    # doc_topic = lda.print_topics(num_topics=lda_topic_num, num_words=10)
    for i in range(lda_topic_num):  
        for j in range(i + 1, lda_topic_num):
            topic_1_word = lda.get_topic_terms(i, topn=args.terms)
            topic_1_word = [lda_dict[idx[0]] for idx in topic_1_word]
            topic_2_word = lda.get_topic_terms(j, topn=args.terms)
            topic_2_word = [lda_dict[idx[0]] for idx in topic_2_word]
            extra_word_list += topic_2_word
            extra_word_list += topic_1_word

            PMI_score_list = []
            for word_1 in topic_1_word:
                for word_2 in topic_2_word:
                    word1, word2 = ordered_word_pair(word_id_map[word_1], word_id_map[word_2])
                    if word_1 == word_2:
                        count = word_occurrence[word_id_map[word_1]]
                    else:
                        try:
                            count = word_pair_occurrence[(word1, word2)]
                        except KeyError:
                            count = 1e-14
                            # print((word_1, word_2))
                    word_freq_i = word_occurrence[word1]
                    word_freq_j = word_occurrence[word2]
                    tmp_pmi = log((count * total_W) / (word_freq_i * word_freq_j))
                    if tmp_pmi <= 0:
                        PMI_score_list.append(0)
                        continue
                    PMI_score_list.append(tmp_pmi)
            PMI_avg = np.sum(PMI_score_list)  # mean
            row.append(train_size + i)
            col.append(train_size + j)
            weight.append(PMI_avg)
            row.append(train_size + j)
            col.append(train_size + i)
            weight.append(PMI_avg)

    # low freq word collect
    # 收集低频信息
    # word_freq_tuple = []
    # for key in word_occurrence:
    #     word_freq_tuple.append((key, word_occurrence[key]))
    # word_freq_tuple.sort(key=lambda x: x[1], reverse=True)
    # import math
    # cut_range = math.floor(len(word_freq_tuple) * 0.25)
    # for w in word_freq_tuple[:cut_range]:
    #     extra_word_list.append(w[0])

    if args.only_topic:
        extra_word_list = []

    extra_word_list = list(set(extra_word_list))  # real word
    len_extra_word = len(extra_word_list)
    node_size += len(extra_word_list)
    print(f'the len of extra word: {len(extra_word_list)}')
    from itertools import combinations
    for comb in list(combinations(extra_word_list, 2)):
        word1 = word_id_map[comb[0]]
        word2 = word_id_map[comb[1]]
        word1, word2 = ordered_word_pair(word1, word2)
        try:
            count = word_pair_occurrence[(word1, word2)]
        except KeyError:
            count = 1e-14
        word_freq_i = word_occurrence[word1]
        word_freq_j = word_occurrence[word2]
        tmp_pmi = log((count * total_W) / (word_freq_i * word_freq_j))
        if tmp_pmi <= 0:
            continue
        row.append(train_size + lda_topic_num + int(extra_word_list.index(comb[0])))
        col.append(train_size + lda_topic_num + int(extra_word_list.index(comb[1])))
        weight.append(tmp_pmi)
        row.append(train_size + lda_topic_num + int(extra_word_list.index(comb[1])))
        col.append(train_size + lda_topic_num + int(extra_word_list.index(comb[0])))
        weight.append(tmp_pmi)

    print("PMI finished.")

    doc_emb = np.zeros((train_size, lda_topic_num + len_extra_word))

    # TODO
    # 在这里，doc_emb要彻底取消。
    # 变成topic_size feature

    # get each word appears in which document
    word_doc_list = {}  # real word
    for word in extra_word_list:
        word_doc_list[word] = []  # real word

    for i in range(train_size):
        doc_words = tokenize_sentences[i]
        unique_words = set(doc_words)
        for word in unique_words:
            if word not in extra_word_list:
                continue
            exist_list = word_doc_list[word]
            exist_list.append(i)
            word_doc_list[word] = exist_list

    # document frequency
    word_doc_freq = {}  # real word
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)  # word_doc_freq is real word, not int

    # term frequency
    doc_word_freq = {}
    for doc_id in range(train_size):
        words = tokenize_sentences[doc_id]
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    # important
    # final doc node and edge
    for i in range(train_size):
        words = tokenize_sentences[i]
        bow_format_text = lda_dict.doc2bow(words)
        vector = lda.get_document_topics(bow_format_text, minimum_probability=0.001)
        # print(vector)
        for prob in vector:
            j = prob[0]
            row.append(i)
            col.append(train_size + j)
            weight.append(prob[1])
            doc_emb[i][j] = prob[1]

        doc_word_set = set()
        for word in words:  # real word
            if word in doc_word_set or word not in extra_word_list:
                continue
            word_id = word_id_map[word]
            key = str(i) + ',' + str(word_id)
            freq = doc_word_freq[key]
            j = extra_word_list.index(word)
            row.append(i)
            col.append(train_size + lda_topic_num + j)
            idf = log(1.0 * train_size / word_doc_freq[word])
            w = freq * idf
            weight.append(w)
            doc_word_set.add(word)
            doc_emb[i][j + lda_topic_num] = w/len(words)

    adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj, doc_emb, word_doc_freq, extra_word_list
