# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class StateEmbed(object):
    def __init__(self, state_dim, embed_dim, hiddens):
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.hiddens = hiddens

    def forward(self, x):
        with tf.variable_scope('state_embed', reuse=tf.AUTO_REUSE):
            for h in self.hiddens:
                x = tf.layers.dense(x, h, activation=None)
                x = tf.nn.leaky_relu(x, alpha=0.05)
            x = tf.layers.dense(x, self.embed_dim)
            # x = tf.contrib.layers.layer_norm(x)
        return x

    def __call__(self, x):
        return self.forward(x)