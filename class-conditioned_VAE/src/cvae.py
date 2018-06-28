#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time
import struct

import tensorflow as tf
from six.moves import range
import numpy as np
import zhusuan as zs
from keras.utils import to_categorical
from scipy.misc import imsave


def load_image_set(filename):
    # load MNIST data
    with open(filename, 'rb') as binfile:
        buffers = binfile.read()
        head = struct.unpack_from('>IIII', buffers, 0)
        offset = struct.calcsize('>IIII')
        imgNum, width, height = head[1], head[2], head[3]
        bits = imgNum * width * height
        bitsString = '>' + str(bits) + 'B'
        imgs = struct.unpack_from(bitsString, buffers, offset)
    imgs = np.reshape(imgs, [imgNum, width, height])
    print('finish loadign data from ' + filename)
    return imgs


def load_label_set(filename):
    # load MNIST label
    with open(filename, 'rb') as binfile:
        buffers = binfile.read()
        head = struct.unpack_from('>II', buffers, 0)
        imgNum = head[1]
        offset = struct.calcsize('>II')
        numString = '>' + str(imgNum) + 'B'
        labels = struct.unpack_from(numString, buffers, offset)
    labels = np.reshape(labels, [imgNum, 1])
    print('finish loading label from ' + filename)
    return labels


@zs.reuse('model')
def cvae(observed, x_dim, y_dim, z_dim, n, n_particles=1):
    with zs.BayesianNet(observed=observed) as model:
        y = zs.Empirical('y', tf.int32, (n, y_dim))
        y = tf.to_float(y)
        z_mean = tf.zeros([n, z_dim])
        z = zs.Normal('z', z_mean, std=1., group_ndims=1, n_samples=n_particles)
        z = tf.to_float(z[0])
        yz = tf.concat([y, z], axis=1)
        lx_yz = tf.layers.dense(tf.to_float(yz), 500, activation=tf.nn.relu)
        lx_yz = tf.layers.dense(lx_yz, 500, activation=tf.nn.relu)
        x_logits = tf.layers.dense(lx_yz, x_dim)
        x_mean = zs.Implicit('x_mean', tf.sigmoid(x_logits), group_ndims=1)
        x = zs.Bernoulli('x', logits=x_logits, group_ndims=1)
    return model


def q_net(observed, x_dim, y_dim, z_dim, n_z_per_xy):
    with zs.BayesianNet(observed=observed) as variational:
        x = zs.Empirical('x', tf.int32, (None, x_dim))
        y = zs.Empirical('y', tf.int32, (None, y_dim))
        xy = tf.concat([x, y], axis=1)
        lz_xy = tf.layers.dense(tf.to_float(xy), 500, activation=tf.nn.relu)
        lz_xy = tf.layers.dense(lz_xy, 500, activation=tf.nn.relu)
        z_mean = tf.layers.dense(lz_xy, z_dim)
        z_logstd = tf.layers.dense(lz_xy, z_dim)
        z = zs.Normal('z', mean=z_mean, logstd=z_logstd, group_ndims=1, n_samples=n_z_per_xy)
    return variational


if __name__ == '__main__':
    train_data_path = '../data/train-images-idx3-ubyte'
    train_label_path = '../data/train-labels-idx1-ubyte'
    test_data_path = '../data/t10k-images-idx3-ubyte'
    test_label_path = '../data/t10k-labels-idx1-ubyte'
    train_x = load_image_set(train_data_path)
    train_y = load_label_set(train_label_path)
    test_x = load_image_set(test_data_path)
    test_y = load_label_set(test_label_path)

    imgNum, width, height = train_x.shape
    x_dim = width * height
    y_dim, z_dim = 10, 40
    x_train = train_x.reshape([imgNum, x_dim]) / 255.
    x_test = test_x.reshape([test_x.shape[0], x_dim]) / 255.
    y_train = to_categorical(train_y, num_classes=y_dim)
    y_test = to_categorical(test_y, num_classes=y_dim)

    n_particles = tf.placeholder(tf.int32, shape=[], name='n_particles')
    x_input = tf.placeholder(tf.float32, shape=[None, x_dim], name='x')
    x = tf.to_int32(tf.less(tf.random_uniform(tf.shape(x_input)), x_input))
    batch_size = 128
    y_input = tf.placeholder(tf.int32, shape=[None, y_dim], name='y')

    def log_joint(observed):
        model = cvae(observed, x_dim, y_dim, z_dim, batch_size, n_particles)
        log_pz, log_px_yz = model.local_log_prob(['z', 'x'])
        return log_pz + log_px_yz

    variational = q_net({'x': x, 'y': y_input}, x_dim, y_dim, z_dim, n_particles)
    qz_samples, log_qz = variational.query('z', outputs=True, local_log_prob=True)
    lower_bound = zs.variational.elbo(log_joint,
                                      observed={'x': x, 'y': y_input},
                                      latent={'z': [qz_samples, log_qz]},
                                      axis=0)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    infer_op = optimizer.minimize(cost)

    n_gen = 100
    x_mean = cvae({'y': y_input}, x_dim, y_dim, z_dim, n_gen, n_particles).outputs('x_mean')
    x_gen = tf.reshape(x_mean, [-1, 28, 28, 1])

    epochs = 100
    # batch_size = 128
    iters = x_train.shape[0] // batch_size
    save_freq, test_freq = 20, 10
    test_batch_size = 128
    test_iters = x_test.shape[0] // test_batch_size

    result_path = '../results/'
    y_vis = np.eye(10)
    y_vis = np.repeat(y_vis, 10, axis=0)
    print(y_vis)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            time_epoch = -time.time()
            shuffle_ind = np.arange(x_train.shape[0])
            np.random.shuffle(shuffle_ind)
            x_train, y_train = x_train[shuffle_ind], y_train[shuffle_ind]
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run([infer_op, lower_bound],
                                 feed_dict={x_input: x_batch,
                                            y_input: y_batch,
                                            n_particles: 1})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(epoch, time_epoch, np.mean(lbs)))

            if epoch % test_freq == 0:
                time_test = -time.time()
                test_lbs = []
                test_lls = []
                for t in range(test_iters):
                    test_x_batch = x_test[t * test_batch_size:(t + 1) * test_batch_size]
                    test_y_batch = y_test[t * test_batch_size:(t + 1) * test_batch_size]
                    test_lb = sess.run(lower_bound,
                                       feed_dict={x_input: test_x_batch,
                                                  y_input: test_y_batch,
                                                  n_particles: 1})
                    test_lbs.append(test_lb)
                time_test += time.time()
                print('>>> TEST ({:.1f}s)'.format(time_test))
                print('>> Test lower bound = {}'.format(np.mean(test_lbs)))

            if epoch % save_freq == 0:
                images = sess.run(x_gen, 
                                  feed_dict={y_input: y_vis, 
                                             n_particles: 1})
                cur_file = os.path.join(result_path, 'epoch_{}.jpg'.format(epoch // save_freq))
                res = np.ones((300, 300))
                for i in range(10):
                    for j in range(10):
                        res[(30 * i + 1):(30 * i + 29), (30 * j + 1):(30 * j + 29)] = np.array(images[i * 10 + j]).reshape(28, 28)
                imsave(cur_file, res) 
