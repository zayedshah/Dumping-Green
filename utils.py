import re
import copy
import pickle
import numpy as np
from math import exp
from collections import OrderedDict
import torch
from torch.autograd import Variable
import os

SILENT = '0 <SILENT>'
UNK = '<UNK>'


def save_pickle(d, path):
    print('save pickle to', path)
    with open(path, mode='wb') as f:
        pickle.dump(d, f)


def load_pickle(path):
    # print('load', path)
    with open(path, mode='rb') as f:
        return pickle.load(f)

def remove_punct(word):
    return re.sub(r'[^a-zA-Z0-9]', '', word)


def get_entities_dict(path):
    #get a dictionary of entities as keys and empty lists as values
    counts = dict()
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            key = line.split()[0]
            if key not in counts:
                counts[key] = []
            else:
                continue
    new_list = []
    for key, value in counts.items():
        new_list.append((key, value))
    new_list.sort()
    new_dict = dict()
    for item in new_list:
        new_dict[item[0]] = []
    return new_dict


def get_entities(fpath):
    # outputs list of entities for each type (dictionary key)

    entities = OrderedDict(get_entities_dict(fpath))
    with open(fpath, 'r') as file:
        # e.g. conditions<\t>heart problems
        lines = file.readlines()
        for l in lines:
            l = re.sub(r'\n', '', l)
            wds = l.split('\t')
            slot_type = wds[0] # ex) R_price
            slot_val = wds[1] # ex) cheap
            if slot_type in entities:
                if slot_val not in entities[slot_type]:
                    entities[slot_type].append(slot_val)
    return entities


def load_embd_weights(word2vec, vocab_size, embd_size, w2i):
    embedding_matrix = np.zeros((vocab_size, embd_size))
    print('embed_matrix.shape', embedding_matrix.shape)
    found_ct = 0
    for word, idx in w2i.items():
        # words not found in embedding index will be all-zeros.
        if word in word2vec.wv:
            embedding_matrix[idx] = word2vec.wv[word]
            found_ct += 1
    print(found_ct, 'words are found in word2vec. vocab_size is', vocab_size)
    return torch.from_numpy(embedding_matrix).type(torch.FloatTensor)


def preload(fpath, vocab):#, system_acts):
    # goes through dialog and builds vocab from user utterances and also system actions
    with open(fpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # e.g. do you have something else<\t>sure let me find an other option for you
        for idx, l in enumerate(lines):
            l = l.rstrip()
            if l != '':
                ls = l.split("\t")
                uttr = ls[0].split(' ')
                if len(ls) == 2: # includes user and system utterance
                    for w in uttr:
                        w = remove_punct(w)
                        if w not in vocab:
                            vocab.append(w)
    vocab = sorted(vocab)
    return vocab


def load_data(fpath, entities, w2i, action_dict):
    # inputs from get_entities and preload
    data = []
    with open(fpath, 'r', encoding='utf-8') as f:
        # e.g. do you have something else<\t>sure let me find an other option for you
        lines = f.readlines()
        # x: user uttr, y: sys act, c: context, b: BoW, p: previous sys act, f: action filter
        x, y, c, b, p, f = [], [], [], [], [], []
        context = [0] * len(entities.keys())
        for idx, l in enumerate(lines):
            l = l.rstrip()
            if l == '':
                data.append((x, y, c, b, p, f))
                # reset
                x, y, c, b, p, f = [], [], [], [], [], []
                context = [0] * len(entities.keys())
            else:
                ls = l.split("\t")
                uttr = ls[0].split(' ')
                update_context(context, uttr, entities)
                act_filter = generate_act_filter(action_dict, context)
                bow = get_bow(uttr, w2i)
                sys_act = SILENT
                if len(ls) == 2: # includes user and system utterance
                    sys_act = ls[1]
                else:
                    continue

                x.append(uttr)
                if len(y) == 0:
                    p.append(SILENT)
                else:
                    p.append(y[-1])
                y.append(sys_act)
                c.append(copy.deepcopy(context))
                b.append(bow)
                f.append(act_filter)
    return data


def update_context(context, sentence, entities, context_settings={}):
    '''

    :param context: list of 0s and 1s to indicate which entities appear in the utterance
    :param sentence: list of words appear in the utterance
    :param entities:
    :param context_settings:
    :return:
    '''
    # indicator vector for all entities found in sentence
    for idx, (ent_key, ent_vals) in enumerate(sorted(entities.items())):
        for w in sentence:
            if any([e in w for e in ent_vals]):
                context[idx] = 1
                w = remove_punct(w)
                context_settings[ent_key] = w
    # TODO: make this better using regular expressions or levenshtein distance


def update_context_fuzzy(context, sentence, entities, context_settings={}, threshold=0.9):
    '''
    Use Levenstein distance to allow fuzzy match.
    :param context:
    :param sentence:
    :param entities:
    :param context_settings:
    :param threshold: adjust this to control the fuzz. threshold = 1 means exact match, 0 means always match.
    :return:
    '''
    for idx, (ent_key, ent_vals) in enumerate(sorted(entities.items())):
        r, w = get_candidate(sentence, ent_vals, get_ratio)
        if r > threshold:
            context[idx] = 1
            w = remove_punct(w)
            context_settings[ent_key] = w


def get_candidate(input_text, ent_vals, score_function):
    r = -1
    candidate = ""
    for entity in ent_vals:
        ratio = score_function(entity, input_text)
        if ratio < r:
            continue
        if ratio > r:
            r = ratio
            candidate = entity
    return r, candidate


def get_ratio(needle, haystack):
    d = fuzzy_substring_distance(needle, haystack)
    m = len(needle) - d
    return exp(-d / 5) * (m / len(needle))


def fuzzy_substring_distance(needle, haystack):
    """Calculates the fuzzy match of needle in haystack,
    using a modified version of the Levenshtein distance
    algorithm.
    The function is modified from the Levenshtein function
    in the bktree module by Adam Hupp
    :type needle: string
    :type haystack: string"""
    m, n = len(needle), len(haystack)

    # base cases
    if m == 1:
        return needle not in haystack
    if not n:
        return m

    row1 = [0] * (n + 1)
    for j in range(0, n + 1):
        if j == 0 or not haystack[j - 1].isalnum():
            row1[j] = 0
        else:
            row1[j] = row1[j - 1] + 1

    for i in range(0, m):
        row2 = [i + 1]
        for j in range(0, n):
            cost = (needle[i] != haystack[j])
            row2.append(min(row1[j + 1] + 1, row2[j] + 1, row1[j] + cost))
        row1 = row2

    d = n + m
    for j in range(0, n + 1):
        if j == 0 or j == n or not haystack[j].isalnum():
            d = min(d, row1[j])
    return d


def generate_act_filter(action_dict, context):
    mask = [0] * len(action_dict)
    ''' context: {'0age_groups': [], '1conditions': [], '2delivery_methods': [], '3drugs': [],
                  '4periods': [], '5severities': [], '6strengths': [], '7symptoms': [], '8units': []}
    '''
    for action_num, action_data in action_dict.items():

        # if no conditions, for standard small talk
        if not action_data[2] and not action_data[3]:
            mask[action_num] = 1

        # to mask if certain context is already available
        if not action_data[2] and action_data[3]:
            if context[action_data[3][0]] == 0:
                mask[action_num] = 1

        # to mask unless context is avialable
        if action_data[2] and not action_data[3]:
            if all([context[a] == 1 for a in action_data[2]]):
                mask[action_num] = 1
    return mask


def get_bow(sentence, w2i):
    # generate bag of words vector for a sentence
    bow = [0] * len(w2i)
    for word in sentence:
        word = remove_punct(word)
        if word in w2i:
            bow[w2i[word]] += 1
    return bow


def check_for_unk(uttr, w2i, entities):
    # check sentence for words not in vocabulary
    unknown = []
    for word in uttr:
        word = remove_punct(word)
        if word not in w2i and word not in entities:
            unknown.append(word)
    return unknown


def add_padding(data, seq_len):
    pad_len = max(0, seq_len - len(data))
    data += [0] * pad_len
    data = data[:seq_len]
    return data


def make_word_vector(uttrs_list, w2i, dialog_maxlen, uttr_maxlen):
    # returns batch of lists of word indices (as defined in w2i)
    dialog_list = []
    for uttrs in uttrs_list:
        dialog = []
        for sentence in uttrs:
            sent_vec = [w2i[remove_punct(w)] if remove_punct(w) in w2i else w2i[UNK] for w in sentence]
            sent_vec = add_padding(sent_vec, uttr_maxlen)
            dialog.append(sent_vec)
        for _ in range(dialog_maxlen - len(dialog)):
            dialog.append([0] * uttr_maxlen)
        dialog = torch.LongTensor(dialog[:dialog_maxlen])
        dialog_list.append(dialog)
    return to_var(torch.stack(dialog_list, 0))


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def sub_reqs(cur, string, req_sub_dict, context_details):
    # searches through entity placeholders in text and substitutes relevant context to the conversation
    for requirement in req_sub_dict.keys():
        if requirement in string:
            exec(req_sub_dict[requirement][0], locals())
            exec(req_sub_dict[requirement][1], locals())
            string = re.sub(requirement, locals()['sub_data'], string)
    return string



def save_embeds(KeyedVectors, word2vec_path, w2i, args, embedding_file):
    # save embedding matrix from word2vec with words in w2i
    # print('loading a word2vec binary...')
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    print('done')
    pre_embd_w = load_embd_weights(word2vec, len(w2i), args.embd_size, w2i)
    save_pickle(pre_embd_w, embedding_file)
    return pre_embd_w






