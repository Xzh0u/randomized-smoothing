__author__ = 'Haohan Wang'

# texture images with radio and random kernel

# low frequency images

import numpy as np
import torch

from datasets import get_dataset


def fft(img):
    return np.fft.fft2(img)


def fftshift(img):
    return np.fft.fftshift(fft(img))


def ifft(img):
    return np.fft.ifft2(img)


def ifftshift(img):
    return ifft(np.fft.ifftshift(img))


def distance(i, j, r):
    dis = np.sqrt((i-14)**2+(j-14)**2)
    if dis < r:
        return 0.0
    else:
        return 1.0


def addingPattern(r, mask):
    fftshift_img = fftshift(r)

    fftshift_result = fftshift_img * mask
    result = ifftshift(fftshift_result)
    return np.real(result)


def mask_radial_MM(r=3.5):
    mask = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            mask[i, j] = distance(i, j, r=r)
    return mask


def mask_random_MM(p=0.5):
    return np.random.binomial(1, 1-p, (28, 28))


def generateData(X):
    radioMask = mask_radial_MM(r=3.5)
    highFreqData = np.zeros(X.shape, dtype=np.float32)
    for i in range(X.shape[0]):
        highFreqData[i] = addingPattern(X[i].numpy().astype(
            np.float32), 1-np.real(radioMask))

    lowFreqData = np.zeros(X.shape, dtype=np.float32)
    for i in range(X.shape[0]):
        lowFreqData[i] = addingPattern(X[i].numpy().astype(
            np.float32), np.real(radioMask))

    randomMask = mask_random_MM(p=0.5)
    randomData = np.zeros(X.shape, dtype=np.float32)
    for i in range(X.shape[0]):
        randomData[i] = addingPattern(X[i].numpy().astype(
            np.float32), np.real(randomMask))

    return randomData, lowFreqData, highFreqData


def generateTextureData():
    dataset = get_dataset("mnist", "train")

    Xtrain_random, Xtrain_lowFreq, Xtrain_highFreq = generateData(
        dataset.train_data)

    # Xval_random, Xval_lowFreq, Xval_highFreq = generateData(Xval)

    # Xtest_random, Xtest_lowFreq, Xtest_highFreq = generateData(Xtest)

    np.save('dataset_cache/MNIST_/Xtrain_random', Xtrain_random)
    np.save('dataset_cache/MNIST_/Xtrain_lowFreq', Xtrain_lowFreq)
    np.save('dataset_cache/MNIST_/Xtrain_highFreq', Xtrain_highFreq)

    # np.save('../data/MNIST/Xval_random', Xval_random)
    # np.save('../data/MNIST/Xval_lowFreq', Xval_lowFreq)
    # np.save('../data/MNIST/Xval_highFreq', Xval_highFreq)

    # np.save('../data/MNIST/Xtest_random', Xtest_random)
    # np.save('../data/MNIST/Xtest_lowFreq', Xtest_lowFreq)
    # np.save('../data/MNIST/Xtest_highFreq', Xtest_highFreq)


if __name__ == '__main__':
    np.random.seed(1)
    generateTextureData()
