# -*- coding: utf-8 -*-

import struct

import numpy as np


def loadImageSet(filename):
    # load MNIST data
    binfile = open(filename, 'rb')
    buffers = binfile.read()
    head = struct.unpack_from('>IIII', buffers, 0)

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'
    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width, height])
    print('finish loading data from ' + filename)
    return imgs


def loadLabelSet(filename):
    # load MNIST label
    binfile = open(filename, 'rb')
    buffers = binfile.read()
    head = struct.unpack_from('>II', buffers, 0)

    imgNum = head[1]
    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + 'B'
    labels = struct.unpack_from(numString, buffers, offset)

    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])
    print('finish loading label from ' + filename)
    return labels
