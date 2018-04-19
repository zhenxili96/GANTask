#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:14:10 2018

@author: zhlixc
"""

# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys
import random

MODEL_DIR = './imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

# Call this function with list of images. Each of elements should be a 
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  inps = []
  for img in images:
    img = img.astype(np.float32)
    inps.append(np.expand_dims(img, 0))
  bs = 100
  with tf.Session() as sess:
    preds = []
    n_batches = int(math.ceil(float(len(inps)) / float(bs)))
    for i in range(n_batches):
        sys.stdout.write(".")
        sys.stdout.flush()
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        pred = sess.run(softmax, {'ExpandDims:0': inp})
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def get_mode_score(images, all_real_images, splits=10):
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  assert(type(all_real_images) == list)
  assert(type(all_real_images[0]) == np.ndarray)
  assert(len(all_real_images[0].shape) == 3)
  assert(np.max(all_real_images[0]) > 10)
  assert(np.min(all_real_images[0]) >= 0.0)
  
  ms_gen = []
  ms_real = []
  for img in images:
    img = img.astype(np.float32)
    ms_gen.append(np.expand_dims(img, 0))
  for img in all_real_images:
    img = img.astype(np.float32)
    ms_real.append(np.expand_dims(img, 0))
    
  bs = 100
  with tf.Session() as sess:
    preds_gen = []
    preds_real = []
    n_batches_gen = int(math.ceil(float(len(ms_gen)) / float(bs)))
    n_batches_real = int(math.ceil(float(len(ms_real)) / float(bs)))
    for i in range(n_batches_gen):
        sys.stdout.write(".")
        sys.stdout.flush()
        inp = ms_gen[(i * bs):min((i + 1) * bs, len(ms_gen))]
        inp = np.concatenate(inp, 0)
        pred = sess.run(softmax, {'ExpandDims:0': inp})
        preds_gen.append(pred)
    for i in range(n_batches_real):
        sys.stdout.write("/")
        sys.stdout.flush()
        inp = ms_real[(i * bs):min((i + 1) * bs, len(ms_real))]
        inp = np.concatenate(inp, 0)
        pred = sess.run(softmax, {'ExpandDims:0': inp})
        preds_real.append(pred)
    preds_gen = np.concatenate(preds_gen, 0)
    preds_real = np.concatenate(preds_real, 0)
    scores = []
    for i in range(splits):
      ## is part
      part = preds_gen[(i * preds_gen.shape[0] // splits):((i + 1) * preds_gen.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      ## extra part
      part_real = preds_real[(i * preds_real.shape[0] // splits):((i + 1) * preds_real.shape[0] // splits), :]
      p_M_y = np.expand_dims(np.mean(part, 0), 0)
      p_M_ys = np.expand_dims(np.mean(part_real, 0), 0)
      kl_extra = p_M_y * (np.log(p_M_y) - np.log(p_M_ys))
      
      scores.append(np.exp(kl - kl_extra))
    return np.mean(scores), np.std(scores)

# This function is called automatically.
def _init_inception():
  global softmax
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
  # Works with an arbitrary minibatch size.
  with tf.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o._shape = tf.TensorShape(new_shape)
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3), w)
    softmax = tf.nn.softmax(logits)

if __name__=='__main__':
    if softmax is None:
      _init_inception()
      
    def get_images(filename):
        return scipy.misc.imread(filename)
    
    fake_imgs = glob.glob(os.path.join('./fake_1000', '*.*'))
    real_imgs = glob.glob(os.path.join('./real_1000', '*.*'))
    
    print ("ratio,mean,std")
    F = open("inception.csv", "w")
    F.write("ratio,mean,std\n")
    for i in range(101):
        fake_num = i*10
        real_num = (100-i)*10
        idx_fake = random.sample(xrange(1000), fake_num)
        idx_real = random.sample(xrange(1000), real_num)
        fake_list = [fake_imgs[j] for j in idx_fake]
        real_list = [real_imgs[j] for j in idx_real]
        imgs_list = np.concatenate((fake_list, real_list)).tolist()   
        images = [get_images(filename) for filename in imgs_list]
        all_real_images = [get_images(filename) for filename in real_imgs]
        print (np.array(all_real_images).shape)
        ### Inception score
        #mean, std = get_inception_score(images)
        ### Mode Score
        mean, std = get_mode_score(images, all_real_images)
        print("{0},{1},{2}".format(i*1.0/100, mean, std))
        F.write("{0},{1},{2}\n".format(i*1.0/100, mean, std))
        F.flush()
    F.close()