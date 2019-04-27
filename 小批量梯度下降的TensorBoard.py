#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:22:04 2019

@author: happy
"""
import tensorflow as tf
import numpy as np 
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from datetime import datetime

#定义TensorBoard读取的日志文件
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

#取数据
housing = fetch_california_housing()
m, n = housing.data.shape

#数据标准化,否则训练梯度的时候会很慢
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
learning_rate = 0.01  #学习率

#定义占位符节点,就是给小批量梯度下降用的,方便后面对这两个占位符节点赋值.
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")  #对theta进行初始化
y_pred = tf.matmul(X, theta, name="predictions")

#可以定义一个命名作用域,以免图变得杂乱庞大
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)  #梯度下降优化器
training_op = optimizer.minimize(mse)  ##使用梯度下降优化器进行求梯度下降

#初始化变量
init = tf.global_variables_initializer()


n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch


mse_summary = tf.summary.scalar('MSE', mse)  #定义一个节点,也是一个对象
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())  #定义一个文件的对象.
with tf.Session() as sess:                                                        # not shown in the book
    sess.run(init)                                                                # not shown

    for epoch in range(n_epochs):                                                 # not shown
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})  #将mse的值写入到summary_str的字符串中
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)  #将summary写入文件中
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()   

file_writer.close() 

best_theta        
                                         # not shown