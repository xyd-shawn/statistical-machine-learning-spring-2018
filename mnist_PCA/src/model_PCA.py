# -*- coding: utf-8 -*-

import sys

import numpy as np
import matplotlib.pyplot as plt

from utils import *


class PCA(object):
    def __init__(self, use_centering=True):
        self.use_centering = use_centering

    def fit(self, X):
        if self.use_centering:
            self.meanX = X.mean(axis=0)
            X = X - self.meanX
        S = X.T.dot(X / np.float(X.shape[0]))
        eigval, self.eigvec = np.linalg.eig(S)
        self.cumsum_eig = np.cumsum(eigval)

    def transform(self, X, pc):
        if self.use_centering:
            X = X - self.meanX
        d = np.where(self.cumsum_eig / self.cumsum_eig[-1] >= pc)[0][0]
        U = self.eigvec[:, :d]
        return U, X.dot(U)

    def inverse_transform(self, Z, U):
        res = Z.dot(U.T)
        if self.use_centering:
            res = res + self.meanX
        return res


if __name__ == '__main__':
    train_data = loadImageSet('../data/train-images-idx3-ubyte')
    numImg, width, height = train_data.shape
    X = np.reshape(train_data, (numImg, width * height))
    clf1 = PCA(use_centering=True)
    clf1.fit(X)
    print('finish clf1')
    clf2 = PCA(use_centering=False)
    clf2.fit(X)
    print('finish clf2')
    while True:
        ss = input('>>> ')
        if ss == 'exit':
            sys.exit()
        ind = int(ss)
        plt.subplot(3, 3, 1)
        plt.imshow(train_data[ind], cmap='gray')
        plt.subplot(3, 3, 2)
        U, Z = clf1.transform(X[ind], 0.3)
        ZZ = clf1.inverse_transform(Z, U).reshape(width, height)
        plt.imshow(ZZ.astype('int'), cmap='gray')
        plt.subplot(3, 3, 3)
        U, Z = clf2.transform(X[ind], 0.3)
        ZZ = clf2.inverse_transform(Z, U).reshape(width, height)
        plt.imshow(ZZ.astype('int'), cmap='gray')
        plt.subplot(3, 3, 4)
        plt.imshow(train_data[ind], cmap='gray')
        plt.subplot(3, 3, 5)
        U, Z = clf1.transform(X[ind], 0.6)
        ZZ = clf1.inverse_transform(Z, U).reshape(width, height)
        plt.imshow(ZZ.astype('int'), cmap='gray')
        plt.subplot(3, 3, 6)
        U, Z = clf2.transform(X[ind], 0.6)
        ZZ = clf2.inverse_transform(Z, U).reshape(width, height)
        plt.imshow(ZZ.astype('int'), cmap='gray')
        plt.subplot(3, 3, 7)
        plt.imshow(train_data[ind], cmap='gray')
        plt.subplot(3, 3, 8)
        U, Z = clf1.transform(X[ind], 0.9)
        ZZ = clf1.inverse_transform(Z, U).reshape(width, height)
        plt.imshow(ZZ.astype('int'), cmap='gray')
        plt.subplot(3, 3, 9)
        U, Z = clf2.transform(X[ind], 0.9)
        ZZ = clf2.inverse_transform(Z, U).reshape(width, height)
        plt.imshow(ZZ.astype('int'), cmap='gray')
        # plt.savefig('../results/mnist_1.png')
        plt.show()


