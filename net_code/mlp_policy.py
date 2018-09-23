import tensorflow as tf
from net_code.distributions import *
import net_code.tf_utils as U
import numpy as np


class MlpPolicy(object):
    def __init__(self, config, name="mlp"):
        self.config = config
        self.scope = name
        with tf.variable_scope(self.scope):
            self._init()

    def _init(self):
        self.pdtype = CategoricalPdType(5) # FIXED

        self.ob = tf.placeholder(tf.float32, [None, self.config["dim_obs"]])
        hid_layers_sizes = self.config["hid_layers_sizes"]

        with tf.variable_scope('vf'):
            last_out = self.ob
            for idx, i in enumerate(hid_layers_sizes):
                last_out = tf.contrib.layers.fully_connected(last_out, i, activation_fn=tf.nn.relu, scope='vf_%i' % (idx))
            self.vpred = tf.contrib.layers.fully_connected(last_out, 1, activation_fn=None, scope="vf_final")

        with tf.variable_scope("pol"):
            last_out = self.ob
            for idx, i in enumerate(hid_layers_sizes):
                last_out = tf.contrib.layers.fully_connected(last_out, i, activation_fn=tf.nn.relu, scope='pol_%i' % (idx))
            pdparam = tf.contrib.layers.fully_connected(last_out, 5, activation_fn=tf.nn.relu, scope='pol_final')
        self.pd = self.pdtype.pdfromflat(pdparam)
        # select or sample actions
        self.stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(self.stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([self.stochastic, self.ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred = self._act(stochastic, ob[None])
        return ac1[0], vpred[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
