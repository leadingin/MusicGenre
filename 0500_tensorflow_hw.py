# -*- coding:utf-8 -*-

import tensorflow as tf
from __future__ import print_function

a = tf.constant(2)
b = tf.constant(3)

with tf.Session() as sess:
    print (sess.run(a+b))
