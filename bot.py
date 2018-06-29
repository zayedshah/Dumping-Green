import random
import copy
import argparse
import sqlite3
import pickle, re
import os


from gensim.models.keyedvectors import KeyedVectors
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils import save_pickle, load_pickle, preload, load_embd_weights, load_data, to_var, update_context, save_embeds
from utils import get_entities, make_word_vector, generate_act_filter, get_bow, check_for_unk, sub_reqs
from models import HybridCodeNetwork


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='each dialog formed one minibatch')
# The initial --embd_size was 300 if using /GoogleNews-vectors-negative300.bin
parser.add_argument('--embd_size', type=int, default=300, help='word embedding size')
parser.add_argument('--hidden_size', type=int, default=128, help='hidden size for LSTM')
parser.add_argument('--test', type=int, default=0, help='1 for test, or for training')
parser.add_argument('--task', type=int, default=5, help='5 for Task 5 and 6 for Task 6')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--vocab_full', type=int, default=0, help='option to use full embedding matrix')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# entities = get_entities('/Users/graeme/dialog-bAbI-tasks/dialog-babi-kb-all.txt')
entities = get_entities('entities.txt')

# for idx, (ent_name, ent_vals) in enumerate(entities.items()):
#     print('entities', idx, ent_name)

# assert args.task == 5 or args.task == 6, 'task must be 5 or 6'
# if args.task == 5:
fpath_train = 'dialogues_train.txt'
fpath_test = 'dialogues_test.txt'
save_path = 'new_trained_model.pt'
# word2vec_path = 'faqmodel.bin'
word2vec_path = '/Users/mac/Downloads/GoogleNews-vectors-negative300.bin'
# word2vec_path = 'C:/Zayed/FellowshipAI/Dialog/GoogleNews-vectors-negative300.bin'
database = 'NovartisQA'
# embedding_file = 'faqmodel.pkl'
embedding_file = 'pre_embd_w_full.pickle'

SILENT = '0 <SILENT>'
UNK = '<UNK>'
system_acts = [SILENT]

if args.vocab_full:
    #TODO make this work, I think matrix is too big to pickle, maybe adapt model to use word2vec object directly
    # print('loading a word2vec binary...')
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    print('done')

    w2i = dict((w, i) for i, w in enumerate(word2vec.index2word))
    i2w = dict((i, w) for i, w in enumerate(word2vec.index2word))
    w2i[UNK] = 0
    i2w[0] = UNK

    try:
        pre_embd_w = load_pickle(embedding_file)
    except:
        # embedding_matrix = word2vec.wv
        # pre_embd_w_full = torch.from_numpy(embedding_matrix).type(torch.FloatTensor)
        # save_pickle(pre_embd_w_full, 'pre_embd_w_full.pickle')
        print('saving full embedding matrix...')
        pre_embd_w = load_embd_weights(word2vec, len(w2i), args.embd_size, w2i)
        save_pickle(pre_embd_w, embedding_file)
else:
    vocab = []
    vocab = preload(fpath_train, vocab) # only read training for vocab because OOV vocabrary should not know.
    # print(vocab)
    training_words = len(vocab)

    master_unk_words = []
    try:
        with open('unknown_words.txt', 'r') as f:
            for line in f:
                line = re.sub('\n','',line)
                master_unk_words.append(line)
    except:
        pass

    for word in master_unk_words:
        if word not in vocab:
            vocab += [word]

    w2i = dict((w, i) for i, w in enumerate(vocab, 1))
    i2w = dict((i, w) for i, w in enumerate(vocab, 1))
    w2i[UNK] = 0
    i2w[0] = UNK

    # get and save embeddings for words in vocab
    try:
        pre_embd_w = load_pickle(embedding_file)
        if pre_embd_w.size(0) < len(w2i):
            pre_embd_w = save_embeds(KeyedVectors, word2vec_path, w2i, args, embedding_file)
    except:
        pre_embd_w = save_embeds(KeyedVectors, word2vec_path, w2i, args,embedding_file)

action_dict = pickle.load(open('actions.pkl', mode='rb'))
req_sub_dict = pickle.load(open('req_sub_dict.pkl', mode='rb'))


# print('vocab size:', len(vocab))
# print('action size:', len(action_dict))
# generate model
# model = HybridCodeNetwork(len(vocab), len(w2i), args.embd_size, args.hidden_size, len(action_dict), len(entities.keys()), pre_embd_w)
# model = HybridCodeNetwork(training_words, len(w2i), args.embd_size, args.hidden_size, len(action_dict), len(entities.keys()), pre_embd_w)
model = HybridCodeNetwork(-1, len(w2i), args.embd_size, args.hidden_size, len(action_dict), len(entities.keys()), pre_embd_w)

if torch.cuda.is_available():
    model.cuda()
optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()))


def padding(data, default_val, maxlen):
    for i, d in enumerate(data):
        pad_len = maxlen - len(d)
        for _ in range(pad_len):
            data[i].append([default_val] * len(entities.keys()))
    return to_var(torch.FloatTensor(data))


def get_data_from_batch(batch, w2i, action_dict):
    # vectorises input data
    uttrs_list = [d[0] for d in batch]
    dialog_maxlen = max([len(uttrs) for uttrs in uttrs_list])
    uttr_maxlen = max([len(u) for uttrs in uttrs_list for u in uttrs])
    uttr_var = make_word_vector(uttrs_list, w2i, dialog_maxlen, uttr_maxlen)

    batch_labels = [d[1] for d in batch]
    labels_var = []
    for labels in batch_labels:
        vec_labels = [int(l.split()[0]) for l in labels]
        pad_len = dialog_maxlen - len(labels)
        for _ in range(pad_len):
            vec_labels.append(0)
        labels_var.append(torch.LongTensor(vec_labels))
    labels_var = to_var(torch.stack(labels_var, 0))

    batch_prev_acts = [d[4] for d in batch]
    prev_var = []
    for prev_acts in batch_prev_acts:
        vec_prev_acts = []
        for act in prev_acts:
            tmp = [0] * len(action_dict.keys())
            tmp[int(act.split()[0])] = 1
            vec_prev_acts.append(tmp)
        pad_len = dialog_maxlen - len(prev_acts)
        for _ in range(pad_len):
            vec_prev_acts.append([0] * len(action_dict.keys()))
        prev_var.append(torch.FloatTensor(vec_prev_acts))
    prev_var = to_var(torch.stack(prev_var, 0))

    context = copy.deepcopy([d[2] for d in batch])
    context = padding(context, 1, dialog_maxlen)

    bow = copy.deepcopy([[b[:training_words+1] for d in batch for b in d[3]]])
    # bow = copy.deepcopy([d[3] for d in batch])
    bow = padding(bow, 0, dialog_maxlen)

    act_filter = copy.deepcopy([d[5] for d in batch])
    act_filter = padding(act_filter, 0, dialog_maxlen)

    return uttr_var, labels_var, context, bow, prev_var, act_filter


def categorical_cross_entropy(preds, labels):
    loss = Variable(torch.zeros(1))
    for p, label in zip(preds, labels):
        loss -= torch.log(p[label] + 1.e-7).cpu()
    loss /= preds.size(0)
    return loss


# TODO make work with batch size > 1
def train(model, data, optimizer, w2i, action_dict, n_epochs=2, batch_size=1):
    print('----Train---')
    data = copy.copy(data)
    for epoch in range(n_epochs):
        print('Epoch', epoch)
        random.shuffle(data)
        acc, total = 0, 0
        for batch_idx in range(0, len(data)-batch_size, batch_size):
            batch = data[batch_idx:batch_idx+batch_size]
            uttrs, labels, contexts, bows, prevs, act_fils = get_data_from_batch(batch, w2i, action_dict)
            # preds = model(uttrs, contexts, bows, prevs, act_fils)
            preds = model(uttrs, contexts, to_var(torch.FloatTensor([])), prevs, act_fils)
            action_size = preds.size(-1)
            preds = preds.view(-1, action_size)
            labels = labels.view(-1)
            loss = categorical_cross_entropy(preds, labels)
            acc += torch.sum(labels == torch.max(preds, 1)[1]).data[0]
            total += labels.size(0)
            if batch_idx % (100 * batch_size) == 0:
                print('Acc: {:.3f}% ({}/{})'.format(100 * acc/total, acc, total))
                print('loss', loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test(model, data, w2i, action_dict, batch_size=1, feedback=False):
    print('----Test---')
    model.eval()
    acc, total = 0, 0
    failed_dialogs = 0
    test_set_size = len(data)
    for batch_idx in range(0, test_set_size-batch_size, batch_size):
        batch = data[batch_idx:batch_idx+batch_size]
        uttrs, labels, contexts, bows, prevs, act_fils = get_data_from_batch(batch, w2i, action_dict)

        # preds = model(uttrs, contexts, bows, prevs, act_fils)
        preds = model(uttrs, contexts, to_var(torch.FloatTensor([])), prevs, act_fils)
        action_size = preds.size(-1)
        preds = preds.view(-1, action_size)
        labels = labels.view(-1)
        # loss = F.nll_loss(preds, labels)
        if torch.sum(labels == torch.max(preds, 1)[1]).data[0] < preds.size(0):
            failed_dialogs += 1
            if feedback:
                print('mistake made, dialog:')
                for user, act, label in zip(batch[0][0], torch.max(preds, 1)[1], labels.data):
                    print(' '.join(user))
                    print('correct action: %s %s' % ('action' + str(label), action_dict[label][0]))
                    print('action taken: %s %s' % ('action' + str(act.data[0]), action_dict[act.data[0]][0]))
                    print('')
        acc += torch.sum(labels == torch.max(preds, 1)[1]).data[0]
        total += labels.size(0)
    print('Test Acc (action): {:.3f}% ({}/{})'.format(100 * acc/total, acc, total))
    completed_dialogs = test_set_size - failed_dialogs
    print('Test Acc (dialogue): {:.3f}% ({}/{})'.format(100 * completed_dialogs/test_set_size, completed_dialogs,
                                                        test_set_size))


context1 = [0] * len(entities.keys())
context_settings1 = {e: [] for e in entities.keys()}
uttr_list1, context_list1, bow_list1, prev_list1, act_fil_list1 = [[], [], [], [], []]

def interactive(uttr,model, w2i, act2i):
    #connect to database
    conn = sqlite3.connect(database + '.sqlite')
    cur = conn.cursor()


    #define global variables to store updated information
    global context1, context_settings1
    global uttr_list1, context_list1, bow_list1, prev_list1, act_fil_list1

    #parse the global variables into the local variables
    context = context1
    context_settings = context_settings1
    uttr_list = uttr_list1
    context_list = context_list1
    bow_list = bow_list1
    prev_list = prev_list1
    act_fil_list = act_fil_list1

    prev = SILENT

    # uttr = input('Enter query (enter * to escape):')
    # dialog_maxlen = 1
    # uttr_maxlen = len(uttr)
    # uttr_var = make_word_vector(uttr, w2i, dialog_maxlen, uttr_maxlen)
    # for dial_loop in range(30):
    uttr = uttr.split()
    update_context(context, uttr, entities, context_settings)
    act_filter = generate_act_filter(action_dict, context)
    bow = get_bow(uttr, w2i)
    # prevs = to_var(torch.FloatTensor(act2i[prev]), 0)
    uttr_list.append(uttr)
    context_list.append(list(context))
    bow_list.append(list(bow))
    prev_list.append(prev)
    act_fil_list.append(list(act_filter))

    dialog_data = [uttr_list, [], context_list, bow_list, prev_list, act_fil_list]
    uttrs, _, contexts, bows, prevs, act_fils = get_data_from_batch([dialog_data], w2i, act2i)

    unknown_words = check_for_unk(uttr, w2i, [e for type in entities.values() for e in type])
    if unknown_words:
        for word in unknown_words:
            if word not in master_unk_words:
                print('I\'m sorry, I don\'t understand: %s' % (word))
                with open('unknown_words.txt', 'a') as file:
                    file.write(word)
                    file.write("\n")
                master_unk_words.append(word)

    # pred_out = model(uttrs, contexts, bows, prevs, act_fils)
    pred_out = model(uttrs, contexts, to_var(torch.FloatTensor([])), prevs, act_fils)
    action_size = pred_out.size(-1)
    pred_dist = pred_out.view(-1, action_size)
    pred = torch.max(pred_dist, 1)[1].data[-1]
    globals()['cur'] = locals()['cur']

    # print(action_dict[pred][0])
    # print(req_sub_dict)
    # print(context_settings)

    query = sub_reqs(cur, action_dict[pred][0], req_sub_dict, context_settings)
    exec(query, globals())
    try: exec(action_dict[pred][1].strip(), globals())
    except: globals()['answer_text'] = 'sorry, there is no information for that'
    test = globals()
    test2 = locals()
    answer_text = sub_reqs(cur, globals()['answer_text'], req_sub_dict, context_settings)

    # uttr = input(answer_text)
    prev = str(pred) + ' ' + answer_text
    conn.close()

    # parse the local variables into the global variables after the process is finished
    context1 = context
    uttr_list1 = uttr_list
    context_list1 = context_list
    bow_list1 = bow_list
    prev_list1 = prev_list
    act_fil_list1 = act_fil_list
    context_settings1 = context_settings

    # print(answer_text)
    # if uttr == '*':
    #     print('Dialogue ended')
    #     break

    return str(answer_text)
    # return context1

from flask import Flask, render_template, request
app = Flask(__name__)

train_data = load_data(fpath_train, entities, w2i, action_dict)
train(model, train_data, optimizer, w2i, action_dict)
torch.save(model.state_dict(), save_path)
#
# model = HybridCodeNetwork(len(vocab), args.embd_size, args.hidden_size, len(action_dict), len(entities.keys()), pre_embd_w)
model = HybridCodeNetwork(-1, len(w2i), args.embd_size, args.hidden_size, len(action_dict), len(entities.keys()), pre_embd_w)
model.load_state_dict(torch.load(save_path))
# interactive(model, w2i, action_dict)
# test_data = load_data(fpath_test, entities, w2i, action_dict)
# test(model, test_data, w2i, action_dict, feedback=True)
@app.route('/')
def index():
    global context1,context_settings1, uttr_list1, context_list1, bow_list1, prev_list1, act_fil_list1
    context1 = [0] * len(entities.keys())
    context_settings1 = {e: [] for e in entities.keys()}
    uttr_list1, context_list1, bow_list1, prev_list1, act_fil_list1 = [[], [], [], [], []]
    return render_template('index.html')

@app.route('/get')
def get_bot_response():
    uttr = request.args.get('msg')
    output = interactive(uttr, model, w2i, action_dict)
    # output = interactive1(uttr)
    return str(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    app.config['DEBUG'] = True
    app.debug = True
    app.config.update(
        DEBUG=True,
        SECRET_KEY='...'
    )








