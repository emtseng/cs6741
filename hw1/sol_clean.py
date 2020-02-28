# Test accs
# Baseline simple vec
# 0.82
# python sol_clean.py --model NB
# 0.793
# python sol_clean.py --model LR --gpu 0 --epochs 200 --lr 20 --bsz 512 --optim sgd
# 0.80
# python sol_clean.py --model CBoW --gpu 0 --epochs 10 --lr 0.001 --bsz 512 --optim adam
# 0.802
# python sol_clean.py --model CBoW --gpu 0 --epochs 10 --lr 0.001 --bsz 512 --optim adam --big-vec
# 0.810
# python sol_clean.py --model CNN --gpu 0 --epochs 10 --lr 0.001 --bsz 512 --optim adam
# 0.83
# python sol_clean.py --model CNN --gpu 0 --epochs 10 --lr 0.001 --bsz 512 --optim adam --big-vec


import collections
import argparse
import math
import random

import torch
import torch.nn as nn
# Text text processing library and methods for pretrained word embeddings
import torchtext
from torchtext.vocab import Vectors, GloVe

# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
from namedtensor.text import NamedField

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=['NB', "LR", "CBoW", "CNN"])
    parser.add_argument("--gpu", type=int, default=-1, help='gpu id. -1 uses cpu.')
    parser.add_argument("--lr", type=float, default=0.01, help='learning rate.')
    parser.add_argument("--epochs", type=int, default=10, help='number of epochs to train.')
    parser.add_argument("--dropout", type=float, default=0.1, help='dropout probability')
    parser.add_argument("--test-code", action="store_true", default=False,
                                            help='write output to predictions.txt')
    parser.add_argument("--optim", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--bsz", type=int, default=10, help='batch size')
    parser.add_argument("--train-subtrees", action="store_true")
    parser.add_argument("--big-vec", action="store_true")
    args = parser.parse_args()
    return args

args = parse_args()


seed = 1111
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


class NB(nn.Module):
    def __init__(self, vocab, num_classes, padding_idx, alpha=1):
        super(NB, self).__init__()
        vocab_size = len(vocab)
        self.num_classes = num_classes
        self.padding_idx = padding_idx
        self.token_class_counts = collections.defaultdict(lambda:alpha)
        self.class_counts = collections.defaultdict(int)
        self.class_total_counts = collections.defaultdict(lambda:alpha*vocab_size)
        self.pxy = nn.Parameter(torch.ones(vocab_size, num_classes))
        self.py = nn.Parameter(torch.zeros(num_classes))

    def update_counts(self, text, label):
        for sent, c in zip(text.unbind('batch'), label.unbind('batch')):
            c = c.item()
            self.class_counts[c] += 1
            for token in sent.tolist():
                if token == self.padding_idx:
                    continue
                self.class_total_counts[c] += 1
                self.token_class_counts[(token, c)] += 1

    def forward(self, text):
        scores = []
        def _logscore(tokens, c):
             class_count = self.class_counts[c]
             if class_count > 0:
                 logscore = 0
                 for token in tokens:
                     if token == self.padding_idx:
                         continue
                     # log P(token | class)
                     logscore += math.log(self.token_class_counts[(token, c)]) \
                                 - math.log(self.class_total_counts[c])
                 # log P(class)
                 logscore += math.log(class_count) 
             else:
                 logscore = -float('inf')
             return logscore
        logscores = []
        for sent in text.unbind('batch'):
            logscores.append([_logscore(sent.tolist(), c) \
                              for c in range(self.num_classes)])
        return NamedTensor(torch.Tensor(logscores), ('batch', 'classes'))


class LR(nn.Module):
    def __init__(self, vocab, num_classes, padding_idx):
        super(LR, self).__init__()
        vocab_size = len(vocab.itos)
        self.lut = ntorch.nn.Embedding(
            vocab_size,
            num_classes,
            padding_idx=padding_idx,
        ).augment('classes')
        #self.bias = ntorch.zeros(dict(classes=num_classes))
        self.bias = ntorch.zeros(num_classes, names=("classes",))
        self.bias_param = nn.Parameter(self.bias._tensor)
        self.bias._tensor = self.bias_param

        self.loss_fn = ntorch.nn.NLLLoss(reduction='sum') \
                                        .reduce(('batch', 'classes'))

    def forward(self, text):
        # here we use log_softmax which over-parameterizes sigmoid since there
        # are only two classes in this specific problem.
        return (self.lut(text).sum('seqlen') + self.bias).log_softmax('classes')

    def cost(self, text, label):
        preds = self.forward(text)
        return self.loss_fn(preds, label)


class CBoW(nn.Module):
    def __init__(self, vocab,  num_classes, padding_idx, dropout=0.1,
                 hidden_size=600):
        super(CBoW, self).__init__()
        vocab_size = len(vocab.itos)
        word_vectors = vocab.vectors
        self.static_lut = ntorch.nn.Embedding(
            vocab_size,
            word_vectors.size(1),
            padding_idx = padding_idx
        ).augment('embedding')
        self.static_lut.weight.data.copy_(word_vectors)
        self.static_lut.requires_grad = False
        self.lut = ntorch.nn.Embedding(
            vocab_size,
            word_vectors.size(1),
            padding_idx = padding_idx
        ).augment('embedding')
        self.lut.weight.data.copy_(word_vectors)
        noise = self.lut.weight.data.new(self.lut.weight.data.size()) \
                                    .normal_(0, 0.01)
        self.lut.weight.data += noise
        self.dropout = ntorch.nn.Dropout(dropout)
        self.proj1 = ntorch.nn.Linear(2*word_vectors.size(1), hidden_size) \
                                     .spec('embedding', "hidden")
        self.proj2 = ntorch.nn.Linear(hidden_size, num_classes) \
                                     .spec('hidden', "classes")

        self.loss_fn = ntorch.nn.NLLLoss(reduction='sum') \
                                        .reduce(('batch', 'classes'))

    def forward(self, text):
        embeddings = ntorch.cat([self.lut(text).sum('seqlen'), \
                             self.static_lut(text).sum('seqlen')], 'embedding')
        embeddings = self.dropout(embeddings)
        hidden = self.proj1(embeddings).relu()
        preds = self.proj2(hidden).log_softmax('classes')
        return preds

    def cost(self, text, label):
        preds = self.forward(text)
        return self.loss_fn(preds, label)

class CNN(nn.Module):
    def __init__(self, vocab, num_classes, padding_idx, dropout=0.5,
                 num_filters=100):
        super(CNN, self).__init__()
        vocab_size = len(vocab.itos)
        word_vectors = vocab.vectors
        self.static_lut = ntorch.nn.Embedding(
            vocab_size,
            word_vectors.size(1),
            padding_idx = padding_idx
        ).augment('embedding')
        self.static_lut.weight.data.copy_(word_vectors)
        self.static_lut.requires_grad = False
        self.lut = ntorch.nn.Embedding(
            vocab_size,
            word_vectors.size(1),
            padding_idx = padding_idx
        ).augment('embedding')
        self.lut.weight.data.copy_(word_vectors)
        kernel_sizes = [3, 4, 5]
        conv_blocks = []
        for kernel_size in kernel_sizes:
            conv1d = ntorch.nn.Conv1d(
                in_channels=word_vectors.size(1)*2,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=1,
                padding=1
            )

            conv_blocks.append(conv1d)
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dropout = ntorch.nn.Dropout(0.5)
        self.proj = ntorch.nn.Linear(
            num_filters * len(kernel_sizes), num_classes
        ).spec("embedding", "classes")

        self.loss_fn = ntorch.nn.NLLLoss(reduction='sum') \
                                        .reduce(('batch', 'classes'))

    def forward(self, text):
        embeddings = ntorch.cat([self.lut(text), self.static_lut(text)],
                                'embedding').transpose('embedding', 'seqlen')
        feature_list = [
            conv_block(embeddings).relu().max("seqlen")[0]
            for conv_block in self.conv_blocks
        ]
        hidden = ntorch.cat(feature_list, "embedding")
        preds = self.proj(self.dropout(hidden)).log_softmax("classes")
        return preds

    def cost(self, text, label):
        preds = self.forward(text)
        return self.loss_fn(preds, label)


def train_NB(model, train_iter):
    for batch in train_iter:
        model.update_counts(batch.text, batch.label)


def train_model(model, train_iter, val_iter, optimizer, epochs):
    model.train()
    best_val = 0
    state = None
    for epoch in range(epochs):
        for batch in train_iter:
            optimizer.zero_grad()
            cost = model.cost(batch.text, batch.label)
            cost = cost.div(batch.text.size('batch'))
            cost.backward()
            optimizer.step()
        correct, total, accuracy = validate(model, val_iter)
        if accuracy > best_val:
            best_val = accuracy
            state = model.state_dict()
        print ('Epoch %d: Validation Accuracy: %f'%(epoch, accuracy))
    return state

# Update: for kaggle the bucket iterator needs to have batch_size 10
def test_code(model, test):
    "All models should be able to be run with following command."
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10)
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = model(batch.text)
        # here we assume that the name for dimension classes is `classes`
        _, argmax = probs.max('classes')
        upload += argmax.tolist()

    with open("predictions.txt", "w") as f:
        f.write("Id,Category\n")
        for i, u in enumerate(upload):
            f.write(str(i) + "," + str(u) + "\n")

def validate(model, it):
    model.eval()
    correct = 0.
    total = 0.
    sentences = []
    with torch.no_grad():
        for batch in it:
            x = batch.text
            y = batch.label
            preds = model(x)
            _, argmax = preds.max('classes')
            results = argmax.to(y._tensor) == y
            correct += results.float().sum('batch').item()
            total += results.size('batch')
    model.train()
    return correct, total, correct / total
    
def main(args):
    # Our input $x$
    TEXT = NamedField(names=('seqlen',))
    
    # Our labels $y$
    LABEL = NamedField(sequential=False, names=(), unk_token=None)
    
    train, val, test = torchtext.datasets.SST.splits(
        TEXT, LABEL,
        filter_pred=lambda ex: ex.label != 'neutral',
        train_subtrees = args.train_subtrees)
       
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)
    vocab_size = len(TEXT.vocab.itos)
    num_classes = len(LABEL.vocab.itos)
    padding_idx = TEXT.vocab.stoi['<pad>']
   
    device = torch.device('cuda:%d'%args.gpu) if args.gpu>-1 else torch.device('cpu')
    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
        (train, val, test), batch_size=10, device=device,
        repeat=False)
    train_iter = torchtext.data.BucketIterator(
        train, batch_size=args.bsz, device=device, repeat=False, train=True)
  
    if args.model != 'NB': 
        # Build the vocabulary with word embeddings
        url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
        print ('loading word vecors from %s'%url)
        if not args.big_vec:
            TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))
        else:
            TEXT.vocab.load_vectors(vectors=GloVe(name="840B"))

    # Build model
    print ('Building model %s'%args.model)
    models = [NB, LR, CBoW, CNN] 
    Model = list(filter(lambda x: x.__name__ == args.model, models))[0]
    model = Model(TEXT.vocab, num_classes, padding_idx)
    if args.gpu > -1:
        model.cuda(args.gpu)

    if args.model == 'NB':
        print ('Counting frequencies')
        train_NB(model, train_iter)
        print ('Validating')
        correct, total, accuracy = validate(model, val_iter) 
        print ('Validation Accuracy: %f'%(accuracy))
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = (
            torch.optim.SGD(params, lr=args.lr)
            if args.optim == "sgd"
            else torch.optim.Adam(params, lr = args.lr)
        )
        state = train_model(model, train_iter, val_iter, optimizer, args.epochs)
        # Load best params based on val acc
        model.load_state_dict(state)

  
    print ('Testing')
    correct, total, accuracy = validate(model, test_iter) 
    print ('Test Accuracy: %f'%(accuracy))

    if args.test_code:
        print ('Writing predictions to predictions.txt')
        test_code(model, test)

if __name__ == '__main__':
    main(args)
