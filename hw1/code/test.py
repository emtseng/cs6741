import torchtext
import argparse
from datetime import datetime

from naivebayes import NaiveBayes

# Build the vocabulary with word embeddings
# url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec'
# TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

def test_code(model):
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

def test_code_NB(model, test_iter):
    "All models should be able to be run with following command."
    upload = []
    # Update: for kaggle the bucket iterator needs to have batch_size 10
    # test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10)
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = model.test(batch.text)
        # here we assume that the name for dimension classes is `classes`
        _, argmax = probs.max('classes')
        upload += argmax.tolist()

    with open("predictions_NB_{}.txt".format(datetime.now()), "w") as f:
        f.write("Id,Category\n")
        for i, u in enumerate(upload):
            f.write(str(i) + "," + str(u) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, help="which model to run",
                        required=True, choices=["NaiveBayes"])

    args = parser.parse_args()

    print("Running test for model: {}".format(args.m))

    # Our input $x$
    TEXT = torchtext.data.Field()

    # Our labels $y$
    LABEL = torchtext.data.Field(sequential=False, unk_token=None)

    # Generate train/test splits from the SST dataset, filter out neutral examples
    train, val, test = torchtext.datasets.SST.splits(
        TEXT, LABEL,
        filter_pred=lambda ex: ex.label != 'neutral')

    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits((train, val, test), batch_size=10, device=-1, repeat = False)

    if args.m == "NaiveBayes":
        epochs = 1
        alpha = 1
        model = NaiveBayes(alpha, TEXT, LABEL)
        model.train(train_iter, val_iter)
        # Evaluate on training set
        train_acc = model.evaluate(self, train_iter)
        print('train_acc: ', train_acc)
        # Evaluate on testing set
        # test_code_NB(model, test_iter)
