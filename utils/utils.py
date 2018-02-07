# coding=utf-8
import os

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
from tensorflow.python.platform import gfile


def load_data_fashion_mnist(data_dir, one_hot=False, num_classes=10):
    train_image_file = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    train_labels_file = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    test_image_file = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_file = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
    with gfile.Open(train_image_file, 'rb') as f:
        train_images = extract_images(f)

    with gfile.Open(train_labels_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=True, num_classes=10)

    with gfile.Open(test_image_file, 'rb') as f:
        test_images = extract_images(f)

    with gfile.Open(test_labels_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=True, num_classes=10)

    return train_images, train_labels, test_images, test_labels


def accuracy(output, label):
    # output, label都是batch_size*num_class的向量
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(label, 1)), tf.float32))
