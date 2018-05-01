# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict



def word_bags(data, vocab_size):
    ndocs = data[:, 0].max()
    res = np.zeros((ndocs, vocab_size))
    for i in range(len(data)):
        if data[i, 1] <= vocab_size:
            res[data[i, 0] - 1, data[i, 1] - 1] += data[i, 2]
    return res

def initialize(data, ntopics):
    labels = np.random.randint(ntopics, size=(len(data)))
    lpi = np.zeros(ntopics)
    mu = np.zeros((data.shape[1], ntopics))
    for i in range(ntopics):
        lpi[i] = (labels == i).sum() / len(data)
        temp = data[labels == i, :].sum(axis=0)
        mu[:, i] = temp / temp.sum()
    return lpi, mu

def E_step(data, lpi, mu):
    gamma = np.zeros((len(data), len(lpi)))
    for i in range(len(data)):
        for j in range(len(lpi)):
            gamma[i, j] = np.log(lpi[j] + 1e-100)
            gamma[i, j] += np.dot(data[i, :], np.log(mu[:, j] + 1e-100))
        temp_mean = np.mean(gamma[i, :])
        gamma[i, :] = gamma[i, :] - temp_mean    # 处理数据溢出问题
        gamma[i, gamma[i, :] > 30] = 30
        gamma[i, gamma[i, :] < -30] = -30
        gamma[i, :] = np.exp(gamma[i, :])
        temp_sum = gamma[i, :].sum()
        gamma[i, :] = gamma[i, :] / temp_sum
    return gamma

def M_step(data, gamma):
    ntopics = gamma.shape[1]
    lpi = gamma.sum(axis=0) / len(data)
    mu = np.zeros((data.shape[1], ntopics))
    nd = data.sum(axis=1)
    for j in range(ntopics):
        temp = np.dot(gamma[:, j], nd)
        for s in range(data.shape[1]):
            mu[s, j] = np.dot(gamma[:, j], data[:, s]) / temp
    return lpi, mu


if __name__ == '__main__':
    texts = np.loadtxt('../data/train.data', dtype=int)
    print('input shape', texts.shape)
    train_vec = word_bags(texts, 53975)
    print('(ndocs, vocab_size)', train_vec.shape)

    with open('../data/vocabulary.txt') as f:
        temp = f.readlines()
    vocab = [s.strip() for s in temp]
    print('number of words', len(vocab))
    print(vocab[:10])
    with open('../data/stopwords.txt') as f:    # 停用词词典
        temp = f.readlines()
    stopwords = [s.strip() for s in temp]
    print('number of stop words', len(stopwords))
    print(stopwords[:10])
    stopwords_id = [i for i in range(len(vocab)) if vocab[i] in stopwords]
    print('number of stop words need to be removed', len(stopwords_id))

    used_words_id = []
    low_freq = 10
    words_count = train_vec.sum(axis=0)
    for i in range(train_vec.shape[1]):
        if (words_count[i] >= low_freq) and (i not in stopwords_id):    # 过滤词频小的和停用词
            used_words_id.append(i)
    print('number of used words', len(used_words_id))
    train_vec = train_vec[:, used_words_id]
    print(train_vec.shape)

    # 初始化
    ntopics = 5
    lpi, mu = initialize(train_vec, ntopics)
    print('shape of pi', lpi.shape)
    print('shape of mu', mu.shape)
    print('should be 1', lpi.sum())
    print(lpi)
    print('should all be 1', mu.sum(axis=0))

    # EM
    nepochs = 30
    for i in range(nepochs):
        print('iteration: ', i + 1)
        gamma = E_step(train_vec, lpi, mu)
        print('should all be 1', gamma.sum(axis=1))
        lpi1, mu1 = M_step(train_vec, gamma)
        print('shape of lpi1', lpi1.shape)
        print('shape of mu1', mu1.shape)
        print('distance between lpi1 and lpi', np.abs(lpi1 - lpi).sum())
        print('distance between mu1 and mu', np.abs(mu1 - mu).sum().sum())
        if np.abs(lpi1 - lpi).sum() < 1e-3 and np.abs(mu1 - mu).sum().sum() < 1e-3:
            break
        else:
            lpi, mu = lpi1, mu1

    # 寻找各类高频词
    thresh = {}
    for i in range(ntopics):
        ccc = np.percentile(mu[:, i], 99.5)
        thresh[i] =  [vocab[used_words_id[j]] for j in np.where(mu[:, i] >= ccc)[0]]
    words = defaultdict(int)
    for i in range(ntopics):
        for ww in thresh[i]:
            words[ww] += 1    # 统计各类重复出现的词
    for i in range(ntopics):
        print('topic ', i + 1, ':')
        www = [ww for ww in thresh[i] if words[ww] <= 2]
        for i in range(0, len(www), 5):
            temp = www[i:(i+5)]
            print(*temp, sep='    ')
        print()
