import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


EPS = 1e-6


def mlp(x, hiddens, activation=None, reguliazer=None):
    for h in hiddens:
        x = tf.layers.dense(x, h, activation=activation, kernel_regularizer=reguliazer)
    return x


def mlp1(x, hiddens, activation=None, output_activation=None, reguliazer=None):
    for h in hiddens[:-1]:
        x = tf.layers.dense(x, h, activation=activation, kernel_regularizer=reguliazer)
    x = tf.layers.dense(x, hiddens[-1], activation=output_activation, kernel_regularizer=reguliazer)
    return x

def mlp_leaky(x, hiddens, reguliazer=None, alpha=0.05):
    for h in hiddens[:-1]:
        x = tf.layers.dense(x, h, activation=None, kernel_regularizer=reguliazer)
        x = tf.nn.leaky_relu(x, alpha=alpha)
    return x

def mlp1_leaky(x, hiddens, output_activation=None, reguliazer=None, alpha=0.05):
    for h in hiddens[:-1]:
        x = tf.layers.dense(x, h, activation=None, kernel_regularizer=reguliazer)
        x = tf.nn.leaky_relu(x, alpha=alpha)
    x = tf.layers.dense(x, hiddens[-1], activation=output_activation, kernel_regularizer=reguliazer)
    return x

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def save_hyperparams(Config, path):
    save_params = {}
    save_params['env_name'] = Config.env_name
    save_params['actor_lr'] = Config.actor_lr
    save_params['critic_lr'] = Config.critic_lr
    save_params['action_embed_lr'] = Config.action_embed_lr
    if Config.sac_ver == 'v1':
        save_params['alpha'] = Config.alpha
    elif Config.sac_ver == 'v2':
        save_params['alpha_lr'] = Config.alpha_lr

    save_params['state_embed_dim'] = Config.state_embed_dim
    save_params['action_embed_dim'] = Config.action_embed_dim
    save_params['state_dims'] = Config.state_dims
    save_params['action_dims'] = Config.action_dims
    save_params['cell_num'] = Config.cell_num
    save_params['seq_len'] = Config.seq_len
    save_params['max_step'] = Config.max_step
    save_params['global_max_step'] = Config.global_max_step
    save_params['episodes'] = Config.episodes

    save_params['state_embed_hiddens'] = Config.state_embed_hiddens
    save_params['action_embed_hiddens'] = Config.action_embed_hiddens
    save_params['ac_hiddens'] = Config.ac_hiddens
    np.save(path + "/save_hyperparams.npy", save_params)

def load_hyperparams(Config, path):
    save_params = np.load(path)
    if save_params['env_name'] != Config.env_name:
        raise Exception("Environment version does not match ! Please check env setting")
    Config.actor_lr = save_params['actor_lr']
    Config.critic_lr = save_params['critic_lr']
    Config.action_embed_lr = save_params['action_embed_lr']
    if Config.sac_ver == 'v2':
        Config.alpha_lr = save_params['alpha_lr']

    Config.state_embed_dim = save_params['state_embed_dim']
    Config.action_embed_dim = save_params['action_embed_dim']
    Config.state_embed_hiddens = save_params['state_embed_hiddens']
    Config.action_embed_hiddens = save_params['action_embed_hiddens']
    Config.ac_hiddens = save_params['ac_hiddens']

    return Config