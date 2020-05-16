""" This script loads a base classifier and then runs PREDICT on many examples from a dataset.
"""
import argparse
# import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
from architectures import get_architecture
import datetime
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Predict on many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str,
                    help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1,
                    help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1,
                    help="stop after this many examples")
parser.add_argument(
    "--split", choices=["train", "test"], default="train", help="train or test set")
parser.add_argument("--N", type=int, default=100000,
                    help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001,
                    help="failure probability")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smoothed classifier g
    smoothed_classifier = Smooth(
        base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    X_te, y_te = get_dataset("mnist_texture", args.split)
    count = 0
    for i in range(X_te.shape[0]):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break
        # show the image
        # plt.imshow(X_te[i].reshape(16, 16), cmap='gray')
        # plt.show()
        x = torch.from_numpy(X_te[i].reshape(1, 28, 28))

        label = y_te[i]
        before_time = time()

        # make the prediction
        prediction = smoothed_classifier.predict(
            x, args.N, args.alpha, args.batch)
        # prediction = base_classifier(x.repeat((args.batch, 1, 1, 1))).argmax(1)

        after_time = time()
        correct = int(prediction == label)
        if correct == 1:
            count = count + 1

        time_elapsed = str(datetime.timedelta(
            seconds=(after_time - before_time)))

        # log the prediction and whether it was correct
        print("{}\t{}\t{}\t{}\t{}".format(i, label, prediction,
                                          correct, time_elapsed), file=f, flush=True)
    acc = count/599
    print("The accuracy is:", acc)
    f.close()
