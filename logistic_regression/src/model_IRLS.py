# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from sklearn.model_selection import train_test_split



def data_process(filename):    # data process
    with open(filename) as f:
        texts = f.readlines()
    X = np.zeros((len(texts), 124))
    y = np.ones(len(texts))
    X[:, 0] = 1    # intercept
    for i, text in tqdm(enumerate(texts)):
        data_list = text.strip().split()
        if i % 1000 == 0:
            print(data_list)
        y[i] = (int(data_list[0]) + 1.) / 2.
        data_list = data_list[1:]
        for data in data_list:
            features = data.split(':')
            X[i, int(features[0])] = 1
    return X, y


def sigmoid(x):
    # x should be numpy.ndarray
    x[x >= 20] = 20    # prevent overflow problems
    x[x <= -20] = -20
    expx = np.exp(-x)
    return 1. / (1. + expx)


def log_likelihood(tt, y):
    # tt = X * w
    ttt = y * tt - np.log(1 + np.exp(tt))
    return ttt.sum()


def predict(X, w):
    res = sigmoid(X.dot(w))
    res[res <= 0.5] = 0
    res[res > 0.5] = 1
    return res


def modelIRLS(X_train, y_train, max_iters, lam):
    iters = 1
    sz = len(X_train)
    n_features = X_train.shape[1]
    w0 = np.zeros(n_features)
    accs, norm_w, log_lk, obj_v = [], [], [], []
    for i in range(max_iters):
        print('iteration ', iters)
        tt = X_train.dot(w0)
        mu = sigmoid(tt)
        res = deepcopy(mu)
        res[res <= 0.5] = 0
        res[res > 0.5] = 1
        accs.append((res == y_train).sum() / sz)
        norm_w.append(np.linalg.norm(w0))
        log_likelihood_value = log_likelihood(tt, y_train)
        log_lk.append(log_likelihood_value)
        obj_v.append(-(lam / 2) * (np.linalg.norm(w0) ** 2) + log_likelihood_value)
        R = mu * (1 - mu)
        R[R < 1e-8] = 1e-8
        RI = 1. / R
        z = tt - RI * (mu - y_train)
        XTR = np.zeros((n_features, sz))
        for j in range(sz):
            XTR[:, j] = R[j] * X_train.T[:, j]
        w1 = np.linalg.inv(lam * np.eye(n_features) + XTR.dot(X_train)).dot(XTR).dot(z)
        if np.linalg.norm(w1 - w0) < 1e-4:
            break
        w0 = w1
        iters += 1
    return w0, iters, accs, norm_w, log_lk, obj_v


def plot_training_process(iters, accs, norm_w, log_lk, obj_v, filename=None, fig_show=False):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(range(iters), accs)
    plt.xlabel('iterations')
    plt.ylabel('train accuracy')
    plt.subplot(2, 2, 2)
    plt.plot(range(iters), obj_v)
    plt.xlabel('iterations')
    plt.ylabel('objective value')
    plt.subplot(2, 2, 3)
    plt.plot(range(iters), norm_w)
    plt.xlabel('iterations')
    plt.ylabel('L2-norm')
    plt.subplot(2, 2, 4)
    plt.plot(range(iters), log_lk)
    plt.xlabel('iterations')
    plt.ylabel('log-likelihood')
    if filename:
        plt.savefig(filename)
    if fig_show:
        plt.show()


if __name__ == '__main__':
    train_x, train_y = data_process('../a9a/a9a')
    test_x, test_y = data_process('../a9a/a9a.t')
    with_reg = True
    max_iters = 50
    if with_reg:
        test_size = 0.3
        shuffle_ind = np.arange(len(train_x))
        np.random.shuffle(shuffle_ind)
        train_x = train_x[shuffle_ind, :]
        train_y = train_y[shuffle_ind]
        X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=test_size)
        lams = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
        train_iters = [-1] * len(lams)
        norm_final_w = [0] * len(lams)
        train_accs = [0] * len(lams)
        val_accs = [0] * len(lams)
        test_accs = [0] * len(lams)
        for i, lam in enumerate(lams):
            print('lambda = ', lam)
            w, iters, accs, norm_w, log_lk, obj_v = modelIRLS(X_train, y_train, max_iters, lam)
            train_iters[i] = iters
            norm_final_w[i] = norm_w[-1]
            train_accs[i] = accs[-1]
            res_val = predict(X_val, w)
            val_accs[i] = (res_val == y_val).sum() / len(y_val)
            res_test = predict(test_x, w)
            test_accs[i] = (res_test == test_y).sum() / len(test_y)
        for i in range(len(lams)):
            print(lams[i], '\t', train_iters[i], '\t', norm_final_w[i], '\t', train_accs[i], '\t', val_accs[i], '\t', test_accs[i])
    else:
        w, iters, accs, norm_w, log_lk, obj_v = modelIRLS(train_x, train_y, max_iters, 0)
        print(norm_w[-1])
