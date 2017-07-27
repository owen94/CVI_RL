import numpy as np
import tensorflow as tf
import pandas as pd
import random
import gym
import matplotlib.pyplot as plt
from Replay_buffer import Replay_buffer

np.random.seed(1)
tf.set_random_seed(1)


class Actor(object):
    def __init__(self, sess, s_dim, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replace_target_steps = t_replace_iter
        self.learn_steps = 0

        self.s_dim = s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')

        self.actor_nets = self.build_network(self.S,scope='actor',n_neurons=30)
        self.target_actor_nets = self.build_network(self.S_,scope='target',n_neurons=30,trainable=False)


    def build_network(self, state, scope, n_neurons, trainable = True, w_initializer = None, b_initializer = None):
        if w_initializer is None:
            w_initializer = tf.random_normal_initializer(0., 0.3)
        if b_initializer is None:
            b_initializer = tf.constant_initializer(0.)

        with tf.variable_scope(scope):
            net = tf.layers.dense(inputs = state, units=n_neurons, activation=tf.nn.relu,
                                  kernel_initializer=w_initializer,bias_initializer=b_initializer, name='l1')
            actions = tf.layers.dense(inputs=net,units=self.action_dim,activation=tf.nn.tanh,kernel_initializer=
                                      w_initializer, bias_initializer=b_initializer, name='a')
            scaled_action = tf.multiply(actions, self.action_bound, name='scaled_a')

        return scaled_action

    def update_actor(self, s):




    def get_action(self, s):










class Critic(object):


