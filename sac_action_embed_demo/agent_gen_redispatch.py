# 不采用AE，SAC直接输出连续变量
# 控制发电机输出（同一子站的发电机输出应同增同减）
# 发电机某一时刻sum(redispatch)应为0

import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import time
tf.disable_v2_behavior()
import os
import argparse
from sklearn.metrics.pairwise import pairwise_distances
from action_embed import ActionEmbed
from config import Config
from experience4GenRedispatch import Expericence
from action_space_gen import gen_actions_generator
from env_info import Env_Info
from utils import *

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

from grid2op import make
from grid2op.Reward import L2RPNReward
from grid2op.Reward import *
from custom_reward.TestReward import TestReward
from custom_reward.L2RPNReward_custom import L2RPNReward_custom
from custom_reward.CumRelevantReward import CumRelevantReward
# from custom_reward.RatioReward import RatioReward
# from custom_reward.BlackoutReward import BlackoutReward
from lightsim2grid.LightSimBackend import LightSimBackend
from grid2op.Parameters import Parameters
try:
    from grid2op.Chronics import MultifolderWithCache
except:
    from grid2op.Chronics import MultiFolder
    MultifolderWithCache = MultiFolder


def convert_obs(observation): # 此处为根据特征对应的范数大小规整，后续可思考是否可改进
    li_vect=  []
    saved_attr_list = ['p_or', 'q_or', 'p_ex', 'q_ex', 'rho', 'prod_p', 'prod_q', 'load_p', 'load_q']
    for el in saved_attr_list:  # observation.attr_list_vect[saved_attr_list_index]:
        v = observation._get_array_from_attr_name(el).astype(np.float32)
        v_fix = np.nan_to_num(v)
        v_norm = np.linalg.norm(v_fix)
        if v_norm > 1e6:
            v_res = (v_fix / v_norm) * 10.0
        else:
            v_res = v_fix
        li_vect.append(v_res)
    return np.concatenate(li_vect)

class Agent(object):
    def __init__(self, env, load_path=None):
        self.env = env

        self.policy_net = SoftAC(Config.state_dims, Config.state_embed_dim, Config.action_dims,
                                     Config.state_embed_hiddens,
                                     Config.ac_hiddens, Config.gamma, Config.actor_lr, Config.critic_lr, Config.tau,
                                     Config.alpha)

        self.experiences = Expericence(Config.state_dims, Config.action_dims, Config.memory_size)
        # initialize the embedding
        self.writer = None
        self.rew_record = list()
        self.step_record = list()
        self.abnormal_date = {}
        self.abnormal_date['step'] = list()
        self.abnormal_date['year'] = list()
        self.abnormal_date['month'] = list()
        self.abnormal_date['day'] = list()
        self.load_path = load_path

        if args.summary:
            self.writer = tf.summary.FileWriter(Config.summary_folder)

        if self.load_path is not None:
            self.policy_net.restore(self.load_path)

    def do_nothing(self):
        a_hat = np.zeros((1, Config.action_dims))
        action = self.env.action_space({})
        return action, a_hat

    def choose_action(self, state, or_state, random=False):
        if random:
            a_hat = 2 * np.random.random((1, Config.action_dims)) - 1
        else:
            a_hat = self.policy_net.act(state.reshape(1, -1).astype(np.float32))
        action, overflow = GenActionSpace.convert_act(a_hat, self.env.action_space, or_state.prod_p)

        action_is_legal = self.action_class_helper.legal_action(action, self.env)
        if not action_is_legal:
            action, a_hat = self.do_nothing()
        else:
            obs_simulate, reward_simulate, done_simulate, info_simulate = or_state.simulate(action)
            if obs_simulate is None:
                action, a_hat = self.do_nothing()
            has_overflow = any(obs_simulate.rho > 1.0)
            if has_overflow:
                action, a_hat = self.do_nothing()
            if info_simulate['is_dispatching_illegal']:  # 目前仅根据当前动作常遇到的问题进行非正常仿真动作的跳过
                action, a_hat = self.do_nothing()
            if done_simulate:
                action, a_hat = self.do_nothing()
        return action, a_hat

    def train_policy(self, epoch):
        s, a_hat, n_s, r, d = self.experiences.sample(Config.batch_size)
        loss_act, loss_crt, summary = self.policy_net.train(epoch, s, a_hat, n_s, r, d)  # self.action_embeddings[task_id][a_i]
        if self.writer:
            self.writer.add_summary(summary, global_step=epoch)

        # print('trainning policy. epoch {}: actor loss: {}, critic loss: {}'.format(epoch, loss_act, loss_crt))
        return loss_act, loss_crt

    def train(self):
        self.global_step = 0
        rewards = []

        if not os.path.exists('data/'):
            os.makedirs('data/')

        plt.figure(figsize=(8, 6))

        for i in range(Config.episodes):
            # todo: may need to modify
            if i > 1000:
                if np.array(self.step_record[-5:] > (Config.max_step - 1) * np.ones(5)).all():
                    break
            or_obs = env.reset()
            # # Random fast forward somewhere in the day
            # ff_rand = np.random.randint(0, 12 * 24 * 20)
            # env.fast_forward_chronics(ff_rand)
            # # Reset internal state
            # or_obs = env.current_obs
            obs = convert_obs(or_obs)
            # print("Reset Observation_Prod_P : ",or_obs.prod_p)
            print("场景信息：", or_obs.year, "年", or_obs.month, "月", or_obs.day, "日", or_obs.hour_of_day, "时",
                  or_obs.minute_of_hour, "分")
            # self.reset(new_obs)
            total_r, done, step = 0, False, 0
            origin_pflow = []
            actual_pflow = []

            while not done and step < Config.max_step:
                self.action_class_helper = env.helper_action_env
                # at first do nothing
                if self.global_step < Config.warm_up and self.load_path is None:
                    action, a_hat = self.do_nothing()
                else:
                    action, a_hat = self.choose_action(obs, or_obs)
                    print(a_hat)

                or_n_obs, r, done, _ = env.step(action)
                n_obs = convert_obs(or_n_obs)
                total_r += r
                step += 1

                self.experiences.store(obs, a_hat, n_obs, r, done)

                or_obs = or_n_obs
                obs = n_obs

                actual_pflow.append(or_obs.prod_p)
                origin_pflow.append((np.array(or_obs.prod_p) - np.array(or_obs.actual_dispatch)).tolist())

                # train policy
                if self.experiences.get_size() > Config.batch_size * 2:
                    self.train_policy(self.global_step)

                self.global_step += 1
                if self.global_step % 2000 == 0:
                    self.policy_net.save(self.global_step, path=Config.model_save_folder + "/" + Config.REW_ver + "/sac/")

            print('Episode: {}, Steps: {} Total reward: {}'.format(i, step, total_r))

            # Plot Generators' Redispatch Graph
            if i % 20 == 0:
                print('Plot Redispatch Graph!')
                actual_pflow = np.array(actual_pflow)
                origin_pflow = np.array(origin_pflow)
                X = range(len(actual_pflow))
                label = []
                for idx in GenActionSpace.genid:
                    plt.plot(X, actual_pflow.T[idx], lw=2)
                    plt.plot(X, origin_pflow.T[idx], '--', lw=2)
                    label.append("ActualPF_Gen_"+str(idx))
                    label.append("OriginPF_Gen_"+str(idx))
                plt.legend(label)
                plt.show()
                plt.clf()


            # Record the date corresponding to the abnormal step
            if i > 3000:
                if step in range(100):
                    cur_obs = env.current_obs
                    self.abnormal_date['step'].append(step)
                    self.abnormal_date['year'].append(cur_obs.year)
                    self.abnormal_date['month'].append(cur_obs.month)
                    self.abnormal_date['day'].append(cur_obs.day)

            rewards.append(total_r)
            self.rew_record.append(total_r)
            self.step_record.append(step)
            if len(rewards) > 100:
                rewards.pop(0)
            avg_reward = np.mean(rewards)

            summary = tf.Summary(value=[tf.Summary.Value(tag="reward1", simple_value=avg_reward)])

            if self.writer:
                self.writer.add_summary(summary, global_step=self.global_step)
            if i >1000 and i % 100 == 0:
                np.save("saved_data/" + Config.env_name + "/" + Config.REW_ver + "/rew_record.npy", np.array(self.rew_record))
                np.save("saved_data/" + Config.env_name + "/" + Config.REW_ver + "/step_record.npy", np.array(self.step_record))
                action_embed_mat = self.action_embed.get_embedding()
                np.save("saved_data/" + Config.env_name + "/" + Config.REW_ver + "/action_embed_mat.npy", action_embed_mat)
                np.save("saved_data/" + Config.env_name + "/" + Config.REW_ver + "/abnormal_date.npy", self.abnormal_date)
            if self.global_step >= Config.global_max_step:
                break

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', default=0, type=int)
    parser.add_argument('-seed', default=0, type=int)
    parser.add_argument('-summary', default=True, type=bool)
    parser.add_argument('-ckpt_path', type=str, required=False)
    parser.add_argument('-ckpt_step', type=str, required=False)
    parser.add_argument('-source_t', type=int, required=False)
    return parser.parse_args()

def print_dur_time(t):
    t = int(t)
    h = t // 3600
    m = (t - 3600*h) // 60
    s = t - 3600*h - 60*m
    print("Total training time : {}h-{}m-{}s".format(h, m, s))

if __name__ == '__main__':

    # 训练前进行设定
    Config.env_name = "l2rpn_case14_sandbox"
    # Config.env_name = "l2rpn_neurips_2020_track2_x1"
    reward_class = L2RPNReward
    Config.sac_ver = "gen"
    load_Config = False
    load_Config_path = None

    if reward_class == L2RPNReward:
        Config.REW_ver = "Test_L2RPNReward"         # use L2RPN reward
    elif reward_class == CombinedScaledReward:
        Config.REW_ver = "Test_custom_reward_xpd"   # use custom reward by XPD
    elif reward_class == TestReward:
        Config.REW_ver = "Test_custom_reward_csy"   # use custom reward by CSY
    elif reward_class == L2RPNReward_custom:
        Config.REW_ver = "Test_L2RPNReward_custom"  # use accumulative reward, restrict one step reward to [0, 1]
    elif reward_class == CumRelevantReward:
        Config.REW_ver = "Test_CumRelevantReward"   # use accumulative reward by xpd
    else:
        raise Exception("Invalid Reward Class !")

    Config.REW_ver += "__GenRedispatch__"

    if Config.sac_ver == "v1":
        from sac import SoftAC
    elif Config.sac_ver == "v2":
        from sac_v2 import SoftAC
        Config.REW_ver = Config.REW_ver + "_sac_v2"
    elif Config.sac_ver == "gen":
        from sac4GenRedispatch import SoftAC
        Config.REW_ver = Config.REW_ver + "_gen_"
    else:
        raise Exception("Invalid SAC Version !")

    args = parse()
    print(args)
    # np.random.seed(args.seed)
    # tf.random.set_random_seed(args.seed)

    # prepare for the env
    backend = LightSimBackend()
    # game_param = Parameters()
    # game_param.NB_TIMESTEP_COOLDOWN_SUB = 2
    # game_param.NB_TIMESTEP_COOLDOWN_LINE = 2
    # chronics = MultifolderWithCache
    env = make(Config.env_name,
               # param=game_param,
               reward_class=reward_class,
               backend=backend,
               # chronics_class=chronics
               )
    env.chronics_handler.set_chunk_size(128)

    # Register custom reward for training
    # cr = env.reward_helper.template_reward
    # cr.addReward("Ratio", RatioReward(), 1.0)
    # cr.addReward("Blackout", BlackoutReward(), 50.0)
    # Initialize custom rewards
    # cr.initialize(env)
    # Set reward range to something managable
    # cr.set_range(-10.0, 10.0)

    grid_info = Env_Info(env)
    GenActionSpace = gen_actions_generator(grid_info)

    # GridWorld Settings
    Config.state_dims = np.size(convert_obs(env.reset()))
    print("Observation size : ", Config.state_dims)
    Config.action_dims = GenActionSpace.adjust_num
    print("Action space dim : ", Config.action_dims)
    if load_Config:
        Config = load_hyperparams(Config, load_Config_path)

    if not os.path.exists(Config.model_save_folder + "/" + Config.REW_ver + "/"):
        os.makedirs(Config.model_save_folder + "/" + Config.REW_ver + "/")
    # else:
    #     raise Exception("File has already exist ! Please check the version of reward !")
    if not os.path.exists(Config.summary_folder + "/" + Config.REW_ver + "/"):
        os.makedirs(Config.summary_folder + "/" + Config.REW_ver + "/")
    if not os.path.exists("saved_data/"+Config.env_name + "/" + Config.REW_ver + "/"):
        os.makedirs("saved_data/"+Config.env_name + "/" + Config.REW_ver + "/")

    # TODO : load model
    # load_path = "saved_models/0/embedding-428000"
    load_path = None

    agent = Agent(env, load_path)

    strf_style = "%Y-%m-%d %H:%M:%S"
    time_start = time.time()
    print("Env : ", Config.env_name)
    print("Start Training !")
    print("Time : ", time.strftime(strf_style,time.localtime(time_start)))

    agent.train()

    print("Complete ! Total Steps : ", agent.global_step)
    time_end = time.time()
    print("Time : ", time.strftime(strf_style,time.localtime(time_end)))
    dur_time = time_end - time_start
    print_dur_time(dur_time)
    np.save("saved_data/" + Config.env_name + "/" + Config.REW_ver + "/rew_record.npy", np.array(agent.rew_record))
    np.save("saved_data/" + Config.env_name + "/" + Config.REW_ver + "/step_record.npy", np.array(agent.step_record))
    action_embed_mat = agent.action_embed.get_embedding()
    np.save("saved_data/" + Config.env_name + "/" + Config.REW_ver + "/action_embed_mat.npy", action_embed_mat)
    path = "saved_data/" + Config.env_name + "/" + Config.REW_ver
    save_hyperparams(Config, path)