import os

from build_dataset import get_dataset
from preprocess import encode_labels, preprocess_data
from build_graph import get_adj
from train import train_model
from utils import *
from model import GCN
from utils import write_logs
import time
from evaluate import get_weights_hidden, get_test_emb, test_model
import argparse
import torch
import torch.optim as optim
import scipy.sparse as sp
import torch.nn as nn
from sklearn.metrics import classification_report
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from tqdm import tqdm
from gensim.models import CoherenceModel

# FAST and Adaptive Topic Graph Convolution

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='R8', help='Dataset string: R8, R52, OH, 20NGnew, MR')
parser.add_argument('--train_size', type=float, default=1,
                    help='If it is larger than 1, it means the number of training samples. If it is from 0 to 1, it means the proportion of the original training set.')
parser.add_argument('--test_size', type=float, default=1,
                    help='If it is larger than 1, it means the number of training samples. If it is from 0 to 1, it means the proportion of the original training set.')
parser.add_argument('--remove_limit', type=int, default=2, help='Remove the words showing fewer than 2 times')
parser.add_argument('--use_gpu', type=int, default=1,
                    help='Whether to use GPU, 1 means True and 0 means False. If True and no GPU available, will use CPU instead.')
parser.add_argument('--shuffle_seed', type=int, default=None,
                    help="If not specified, train/val is shuffled differently in each experiment")
parser.add_argument('--hidden_dim', type=int, default=300, help="The hidden dimension of GCN model")
parser.add_argument('--dropout', type=float, default=0.5, help="The dropout rate of GCN model")
parser.add_argument('--learning_rate', type=float, default=0.02, help="Learning rate")
parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay, normally it is 0")
parser.add_argument('--early_stopping', type=int, default=100, help="Number of epochs of early stopping.")
parser.add_argument('--epochs', type=int, default=200, help="Number of maximum epochs")
parser.add_argument('--multiple_times', type=int, default=10,
                    help="Running multiple experiments, each time the train/val split is different")
parser.add_argument('--easy_copy', type=int, default=0,
                    help="For easy copy of the experiment results. 1 means True and 0 means False.")
parser.add_argument('--no_below', type=int, default=10,
                    help="")
parser.add_argument('--no_above', type=int, default=0.7,
                    help="")
parser.add_argument('--topic_range_limit', type=int, default=21,
                    help="")
parser.add_argument('--only_topic', type=int, default=0,
                    help="only topic? 1/0, default 0 , 0 is not only topic, 1 is only topic")
parser.add_argument('--output_other_format', type=int, default=0,  
                    help="build other format dataset, default 0")
parser.add_argument('--sp_topic', type=int, default=0)
parser.add_argument('--terms', type=int, default=65)
parser.add_argument('--zh', type=int, default=0)
  
args = parser.parse_args()

device = decide_device(args)

now_time = time.strftime('%Y-%m-%d-%H:%M:%S')
logs_name = f'./log/{now_time}_{args.dataset}_{args.train_size}_{args.test_size}_{"L" if args.sp_topic == 0 else args.sp_topic}.txt'
write_logs(logs_name, str(__file__))
for v in vars(args):
    write_logs(logs_name, f'{v} : {getattr(args, v)}')

# Get dataset
sentences, labels, train_size, test_size = get_dataset(args)
train_sentences = sentences[:train_size]
test_sentences = sentences[train_size:]
train_labels = labels[:train_size]
test_labels = labels[train_size:]

# Preprocess text and labels
labels, num_class = encode_labels(train_labels, test_labels, args)
labels = torch.LongTensor(labels).to(device)
tokenize_sentences, word_list = preprocess_data(train_sentences, test_sentences, args)
vocab_length = len(word_list)
word_id_map = {}
for i in range(vocab_length):
    word_id_map[word_list[i]] = i
if not args.easy_copy:
    print("There are", vocab_length, "unique words in total.")

write_logs(logs_name, f"There are {vocab_length} unique words in total.")

# TODO: generate LDA model
train_dictionary = Dictionary(tokenize_sentences[:train_size])
train_dictionary.filter_extremes(no_below=args.no_below, no_above=args.no_above)
train_corpus = [train_dictionary.doc2bow(text) for text in tokenize_sentences[:train_size]]

coherence_score = []
# perplexity = []
model_list = []
topic_num_list = []

if args.sp_topic == 0:
    start_topic_num = 9
    end_topic_num = args.topic_range_limit
    print('开始 寻找最佳主题num')
else:
    start_topic_num = args.sp_topic
    end_topic_num = start_topic_num + 1
    print('指定主题实验')

for num_topic in tqdm(range(start_topic_num, end_topic_num, 1)):
    lda_model = LdaModel(corpus=train_corpus,
                         id2word=train_dictionary,
                         random_state=137,
                         num_topics=num_topic)
    model_list.append(lda_model)
    coherence_model = CoherenceModel(model=lda_model,
                                     texts=tokenize_sentences[:train_size],
                                     dictionary=train_dictionary,
                                     coherence='c_v')
    coherence_score.append(round(coherence_model.get_coherence(), 10))
    print(f'主题 {num_topic}, 分数为：{coherence_model.get_coherence()}')
    write_logs(logs_name, f'主题 {num_topic}, 分数为：{coherence_model.get_coherence()}')
    topic_num_list.append(num_topic)

max_score, max_score_index = max(coherence_score), coherence_score.index(max(coherence_score))
lda = model_list[max_score_index]
topic_num = topic_num_list[max_score_index]
del model_list

# Generate Graph
adj, doc_emb, word_doc_freq, extra_word_list = get_adj(lda, topic_num, train_dictionary,
                                                       tokenize_sentences, train_size, word_id_map, word_list, args)
adj, norm_item = normalize_adj(adj + sp.eye(adj.shape[0]))
adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
features = torch.FloatTensor(doc_emb).to(device)

criterion = nn.CrossEntropyLoss()

# Generate Test input
test_emb, tokenized_test_edge = get_test_emb(tokenize_sentences[train_size:], topic_num, lda, train_dictionary,
                                             extra_word_list, word_id_map, train_size, word_doc_freq)

if not args.easy_copy:
    # Generate train/val dataset
    idx_train, idx_val = generate_train_val(args, train_size)

    # Genrate Model
    model = GCN(nfeat=topic_num + len(extra_word_list), nhid=args.hidden_dim, nclass=num_class, dropout=args.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Train the model
    train_model(args, model, optimizer, criterion, features, adj, labels, idx_train, idx_val)

    # Test
    if not args.easy_copy:
        print("Predicting on test set.")
    model_weights_list = get_weights_hidden(model, features, adj, train_size)
    test_result = test_model(model, test_emb, tokenized_test_edge, model_weights_list, device)
    print(classification_report(labels[train_size:].cpu(), test_result, digits=4))
    write_logs(logs_name, classification_report(labels[train_size:].cpu(), test_result, digits=4))

if args.multiple_times:
    test_acc_list = []
    for t in range(args.multiple_times):
        if not args.easy_copy:
            print("Round", t + 1)
        try:
            model = GCN(nfeat=topic_num + len(extra_word_list), nhid=args.hidden_dim, nclass=num_class, dropout=args.dropout).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            idx_train, idx_val = generate_train_val(args, train_size)
            train_model(args, model, optimizer, criterion, features, adj, labels, idx_train, idx_val, show_result=False)
            model_weights_list = get_weights_hidden(model, features, adj, train_size)
            test_result = test_model(model, test_emb, tokenized_test_edge, model_weights_list, device)
            test_acc_list.append(accuracy_score(labels[train_size:].cpu(), test_result))
        except Exception as e:
            write_logs(logs_name, e)
    if args.easy_copy:

        print("%.4f" % np.mean(test_acc_list), end=' ± ')
        print("%.4f" % np.std(test_acc_list))
        write_logs(logs_name, np.mean(test_acc_list))
        write_logs(logs_name, np.std(test_acc_list))

    else:
        for t in test_acc_list:
            print("%.4f" % t)
            write_logs(logs_name, t)
        print("Test Accuracy:", np.round(test_acc_list, 4).tolist())
        write_logs(logs_name, "Test Accuracy:" + str(np.round(test_acc_list, 4).tolist()))
        print("Mean:%.4f" % np.mean(test_acc_list))
        write_logs(logs_name, "MEAN:" + str(np.mean(test_acc_list)))
        # mean_result = str(round(np.mean(test_acc_list), 5))
        print("Std:%.4f" % np.std(test_acc_list))
        write_logs(logs_name, "STD:" + str(np.std(test_acc_list)))


    mean_result = str(round(np.mean(test_acc_list), 4))
    write_logs(logs_name, f'mean:{mean_result}, score:{round(max_score, 4)}, term:{len(extra_word_list)}, vocab:{vocab_length}, topic:{args.sp_topic}')
    os.rename(logs_name, logs_name.replace('.txt', '') + f'_{mean_result}_{round(max_score, 4)}.txt')
