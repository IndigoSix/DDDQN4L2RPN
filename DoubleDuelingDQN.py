# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import json
import math
import numpy as np
import tensorflow as tf
from itertools import combinations
import operator
from functools import reduce
from grid2op.Agent import AgentWithConverter
from grid2op.Agent import DoNothingAgent
from grid2op.Converter import IdToAct

from .DoubleDuelingDQN_NN import DoubleDuelingDQN_NN
from .prioritized_replay_buffer import PrioritizedReplayBuffer

LR_DECAY_STEPS = 1024*32
LR_DECAY_RATE = 0.95
INITIAL_EPSILON = 0.99
FINAL_EPSILON = 0.001
DECAY_EPSILON = 1024*32
DISCOUNT_FACTOR = 0.99
PER_CAPACITY = 1024*64
PER_ALPHA = 0.7
PER_BETA = 0.5
UPDATE_FREQ = 64
UPDATE_TARGET_HARD_FREQ = 16
UPDATE_TARGET_SOFT_TAU = -1


class DoubleDuelingDQN(AgentWithConverter):#(AgentWithConverter):
    def __init__(self,
                 observation_space,
                 action_space,
                 name=__name__,
                 num_frames=4,
                 is_training=False,
                 batch_size=32,
                 lr=1e-5):
        # Call parent constructor
        #AgentWithConverter.__init__(self, action_space,
                                    #action_space_converter=IdToAct)
        AgentWithConverter.__init__(self, action_space)
        self.agent_action_space = action_space#完整的动作空间
        #self.do_nothing = self.action_space()
        self.obs_space = observation_space
        
        # Store constructor params
        self.name = name
        self.num_frames = num_frames
        self.is_training = is_training
        self.batch_size = batch_size
        self.lr = lr
        
        # Declare required vars
        self.Qmain = None
        self.obs = None
        self.state = []
        self.frames = []

        # Declare training vars
        self.per_buffer = None
        self.done = False
        self.frames2 = None
        self.epoch_rewards = None
        self.epoch_alive = None
        self.Qtarget = None
        self.epsilon = 0.0

        # Compute dimensions from intial spaces
        self.observation_size = 752#直接根据目前期望保留的观测值特征赋值建立DNN，后续需要调整程序更为自动化#self.obs_space.size_obs()
        self.data_dir = os.path.abspath(".")
        action_dir = os.path.join(self.data_dir, 'saved_actions.npy') 
        print(action_dir)
        #self.action_space = np.load(action_dir, allow_pickle=True)
        self.action_space = self.saved_actions_generator()
        #self.action_space = np.load("./saved_actions.npy", allow_pickle=True)#读取自定义动作空间作为智能体的动作空间
        self.action_size = self.action_space.shape[0]
        print("神经网络动作种类",self.action_size)

        # Load network graph
        self.Qmain = DoubleDuelingDQN_NN(self.action_size,
                                         self.observation_size,
                                         num_frames=self.num_frames,
                                         learning_rate=self.lr,
                                         learning_rate_decay_steps=LR_DECAY_STEPS,
                                         learning_rate_decay_rate=LR_DECAY_RATE)
        # Setup training vars if needed
        if self.is_training:
            self._init_training()

    def _init_training(self):
        self.epsilon = INITIAL_EPSILON
        self.frames2 = []
        self.epoch_rewards = []
        self.epoch_alive = []
        self.per_buffer = PrioritizedReplayBuffer(PER_CAPACITY, PER_ALPHA)
        self.Qtarget = DoubleDuelingDQN_NN(self.action_size,
                                           self.observation_size,
                                           num_frames = self.num_frames)

    def _reset_state(self, current_obs):
        # Initial state
        self.obs = current_obs
        self.state = self.convert_obs(self.obs)
        self.done = False

    def _reset_frame_buffer(self):
        # Reset frame buffers
        self.frames = []
        if self.is_training:
            self.frames2 = []

    def _save_current_frame(self, state):
        self.frames.append(state.copy())
        if len(self.frames) > self.num_frames:
            self.frames.pop(0)

    def _save_next_frame(self, next_state):
        self.frames2.append(next_state.copy())
        if len(self.frames2) > self.num_frames:
            self.frames2.pop(0)

    def _adaptive_epsilon_decay(self, step):
        ada_div = DECAY_EPSILON / 10.0
        step_off = step + ada_div
        ada_eps = INITIAL_EPSILON * -math.log10((step_off + 1) / (DECAY_EPSILON + ada_div))
        ada_eps_up_clip = min(INITIAL_EPSILON, ada_eps)
        ada_eps_low_clip = max(FINAL_EPSILON, ada_eps_up_clip)
        return ada_eps_low_clip
            
    def _save_hyperparameters(self, logpath, env, steps):
        r_instance = env.reward_helper.template_reward
        hp = {
            "lr": self.lr,
            "lr_decay_steps": LR_DECAY_STEPS,
            "lr_decay_rate": LR_DECAY_RATE,
            "batch_size": self.batch_size,
            "stack_frames": self.num_frames,
            "iter": steps,
            "e_start": INITIAL_EPSILON,
            "e_end": FINAL_EPSILON,
            "e_decay": DECAY_EPSILON,
            "discount": DISCOUNT_FACTOR,
            "per_alpha": PER_ALPHA,
            "per_beta": PER_BETA,
            "per_capacity": PER_CAPACITY,
            "update_freq": UPDATE_FREQ,
            "update_hard": UPDATE_TARGET_HARD_FREQ,
            "update_soft": UPDATE_TARGET_SOFT_TAU,
            "reward": dict(r_instance)
        }
        hp_filename = "{}-hypers.json".format(self.name)
        hp_path = os.path.join(logpath, hp_filename)
        with open(hp_path, 'w') as fp:
            json.dump(hp, fp=fp, indent=2)

    ## Agent Interface
    def convert_obs(self, observation):#此处为根据特征对应的范数大小规整，后续可思考是否可改进
        li_vect=  []
        saved_attr_list = ['prod_p', 'load_p', 'rho', 'p_or', 'p_ex', 'line_status', 'timestep_overflow', 'topo_vect', 'time_before_cooldown_line',  'time_next_maintenance', 'duration_next_maintenance', 'target_dispatch', 'actual_dispatch']
        #得到期望保留观测值熟悉对应在整个观测值特征中的指标
        for el in saved_attr_list:#observation.attr_list_vect[saved_attr_list_index]:
            v = observation._get_array_from_attr_name(el).astype(np.float32)
            v_fix = np.nan_to_num(v)
            v_norm = np.linalg.norm(v_fix)
            if v_norm > 1e6:
                v_res = (v_fix / v_norm) * 10.0
            else:
                v_res = v_fix
            li_vect.append(v_res)
        return np.concatenate(li_vect)

    def convert_act(self, action):#当前只能支持对1个动作的转换
        action_array = self.action_space[action]
        # Transfer the l2rpn action array to a dict of grid2op
        action_dict = {}
        obj_id_dict = {}
        real_action = self.agent_action_space(action_dict)#首先初始时默认不动作


        #line status set
        offset = 0
        set_lines_status_array = action_array[:59]
        set_lines_id_list = list(np.where(set_lines_status_array != 0)[0])
        if len(set_lines_id_list) != 0:
            for i in range(len(set_lines_id_list)):
                if set_lines_status_array[set_lines_id_list[i]] == -1:
                    real_action = self.agent_action_space.disconnect_powerline(line_id = set_lines_id_list[i])
                else:
                    real_action = self.agent_action_space.reconnect_powerline(line_id=set_lines_id_list[i], bus_ex=int(set_lines_status_array[set_lines_id_list[i]]//10), bus_or=int(set_lines_status_array[set_lines_id_list[i]]%10))
        
        # generator redipatch
        offset += 472
        generator_redipatch_array = action_array[offset:]
        generator_redipatch_id_list = list(np.where(generator_redipatch_array != 0)[0])
        if len(generator_redipatch_id_list) != 0:
            for i in range(len(generator_redipatch_id_list)):
                real_action = self.agent_action_space({"redispatch": [(generator_redipatch_id_list[i], generator_redipatch_array[generator_redipatch_id_list[i]])]})
        

        # load reconfigration
        offset += -177
        change_load_array = action_array[offset:offset+37]
        change_load_id_list = list(np.where(change_load_array == 1)[0])
        if len(change_load_id_list) != 0:
            obj_id_dict["loads_id"] = [i for i in change_load_id_list]

        # generator reconfiguration
        offset += 37
        change_gen_array = action_array[offset:offset+22]
        change_gen_id_list = list(np.where(change_gen_array == 1)[0])
        if len(change_load_id_list) != 0:
            obj_id_dict["generators_id"] = [i for i in change_gen_id_list]
        
        # line ox reconfiguration
        offset += 22
        change_lines_or_array = action_array[offset:offset+59]
        change_or_id_list = list(np.where(change_lines_or_array == 1)[0])
        if len(change_or_id_list) != 0:
            obj_id_dict["lines_or_id"] = [i for i in change_or_id_list]

          #print(obj_id_dict["lines_or_id"])
        # line ex
        offset += 59
        change_lines_ex_array = action_array[offset:offset+59]
        change_ex_id_list = list(np.where(change_lines_ex_array == 1)[0])
        if len(change_ex_id_list) != 0:
            obj_id_dict["lines_ex_id"] = [i for i in change_ex_id_list]
        
        if  len(obj_id_dict) != 0:
            action_dict["change_bus"] = obj_id_dict
        
        real_action = self.agent_action_space(action_dict)
            
        return real_action

    ## Baseline Interface
    def reset(self, observation):
        self._reset_state(observation)
        self._reset_frame_buffer()

    def my_act(self, state, reward, done=False):
        # Register current state to stacking buffer
        self._save_current_frame(state)
        # We need at least num frames to predict
        if len(self.frames) < self.num_frames:
            return 802, 0 # Do nothing and a flag(0) for go ahead without simulation
        # Infer with the last num_frames states
        a, Q_scores = self.Qmain.predict_move(np.array(self.frames))
        return a, Q_scores
    
    def load(self, path):
        self.Qmain.load_network(path)
        if self.is_training:
            self.Qmain.update_target_hard(self.Qtarget.model)

    def save(self, path):
        self.Qmain.save_network(path)

    ## Training Procedure
    def train(self, env,
              iterations,
              save_path,
              num_pre_training_steps=0,
              logdir = "logs-train"):
        # Make sure we can fill the experience buffer
        if num_pre_training_steps < self.batch_size * self.num_frames:
            num_pre_training_steps = self.batch_size * self.num_frames

        # Loop vars
        num_training_steps = iterations
        num_steps = num_pre_training_steps + num_training_steps
        step = 0
        self.epsilon = INITIAL_EPSILON
        alive_steps = 0
        total_reward = 0
        self.done = True

        # Create file system related vars
        logpath = os.path.join(logdir, self.name)
        os.makedirs(save_path, exist_ok=True)
        modelpath = os.path.join(save_path, self.name + ".h5")
        self.tf_writer = tf.summary.create_file_writer(logpath, name=self.name)
        self._save_hyperparameters(save_path, env, num_steps)
        
        # Training loop
        while step < num_steps:
            # Init first time or new episode
            if self.done:
                new_obs = env.reset() # This shouldn't raise
                # Random fast forward somewhere in the day
                ff_rand = np.random.randint(0, 12*24*25) 
                env.fast_forward_chronics(ff_rand)
                # Reset internal state
                new_obs = env.current_obs
                print("场景信息：",new_obs.year, "年",new_obs.month, "月",new_obs.day, "日",new_obs.hour_of_day, "时",new_obs.minute_of_hour,"分")
                self.reset(new_obs)
            if step % 1000 == 0:
                print("Step [{}] -- Random [{}]".format(step, self.epsilon))

            # Save current observation to stacking buffer
            self._save_current_frame(self.state)

            # Choose an action
            sim_flag = 0#define the simulation flag for DNN predictions
            if step <= num_pre_training_steps:
                a = self.Qmain.random_move()
            elif np.random.rand(1) < self.epsilon:
                a = self.Qmain.random_move()
            elif len(self.frames) < self.num_frames:
                a = 802 # Do nothing
            else:
                a, q_socres = self.Qmain.predict_move(np.array(self.frames))
                sim_flag = 1

            # Convert it to a valid action
            action_class_helper = env.helper_action_env
            if not sim_flag:
                act = self.convert_act(a)
                action_is_legal = action_class_helper.legal_action(act, env)
                if not action_is_legal:
                    a = 802#若神经网络选择动作不合理，则默认动作采取不动作
                    act = self.agent_action_space({})
            else:
                top_actnums = -20#若为模型输出结果，初始挑选20-1个Q值最大动作，逐步收紧
                print("step:",step)
                if step % 1000 == 0:
                    top_actnums = min(-5, top_actnums+1)
                top_actions = np.argsort(q_socres)[-1: top_actnums: -1].tolist()
                #print("action guided by simulation",top_actions)
                max_score = float('-inf')
                a = 802
                chosen_action = self.agent_action_space({})#预先定义选取动作为无动作，以避免解集均失效
                top_actions.insert(0,802)#将不动作添加入最优解集首位，以其为基准规整数值，避免当前redispreward容易溢出的情况下电网无意义调整出力
                for pre_action in tuple(top_actions):
                    pre_act = self.convert_act(pre_action)
                    #print("当前检测动作编号",pre_action)
                    action_is_legal = action_class_helper.legal_action(pre_act, env)
                    #print("当前检测动作基本合理性",action_is_legal)
                    if not action_is_legal:
                        continue
                    else:
                        obs_simulate, reward_simulate, done_simulate, info_simulate= new_obs.simulate(pre_act)
                        calculated_socre = info_simulate["rewards"]["gameplay"]*100 + info_simulate["rewards"]["Overflow"]*50 + info_simulate["rewards"]["Redisp"]*0.01#由于当前仿真功能无法正常输出具体值，利用权重得到计算预测奖励值
                        #calculated_socre_trans = np.interp(calculated_socre, [-100.1, 155.05], [-10, 10])
                        if obs_simulate is None:
                            continue
                        has_overflow = any(obs_simulate.rho>1.0)
                        if has_overflow:
                            calculated_socre = calculated_socre - 10#若仿真后场景中存在过载的线路，则应予以进一步的惩罚————等同于多一条线路重在
                        reward_simulate = calculated_socre#利用计算得到的奖励值代替不正常输出的奖励值
                        #print("预测其他奖励",info_simulate["rewards"]["Redisp"])
                        #print("当前检测动作预测奖励值",reward_simulate)
                        if  info_simulate['is_dispatching_illegal']:#目前仅根据当前动作常遇到的问题进行非正常仿真动作的跳过
                            continue
                        else:
                            if not done_simulate and reward_simulate > max_score:
                                max_score = reward_simulate
                                chosen_action = pre_act
                                a = pre_action
                                print('current best action: {}, score: {:.4f}'.format(chosen_action, reward_simulate))
                                #print("预计奖励：",reward_simulate)
                act = chosen_action
            #print(act)
            # Execute action
            new_obs, reward, self.done, info = env.step(act)
            #print("真实出力：",new_obs.prod_p)
            #print("真实奖励值：",reward)
            new_state = self.convert_obs(new_obs)
            if info["is_illegal"] or info["is_ambiguous"] or \
               info["is_dispatching_illegal"] or info["is_illegal_reco"]:
                print (a, info)

            # Save new observation to stacking buffer
            self._save_next_frame(new_state)

            # Save to experience buffer
            if len(self.frames2) == self.num_frames:
                self.per_buffer.add(np.array(self.frames),
                                    a, reward,
                                    np.array(self.frames2),
                                    self.done)

            # Perform training when we have enough experience in buffer
            if step >= num_pre_training_steps:
                training_step = step - num_pre_training_steps
                # Decay chance of random action
                self.epsilon = self._adaptive_epsilon_decay(training_step)

                # Perform training at given frequency
                if step % UPDATE_FREQ == 0 and len(self.per_buffer) >= self.batch_size:
                    # Perform training
                    self._batch_train(training_step, step)

                    if UPDATE_TARGET_SOFT_TAU > 0.0:
                        # Update target network towards primary network
                        self.Qmain.update_target_soft(self.Qtarget.model, tau=UPDATE_TARGET_SOFT_TAU)

                # Every UPDATE_TARGET_HARD_FREQ trainings, update target completely
                if UPDATE_TARGET_HARD_FREQ > 0 and step % (UPDATE_FREQ * UPDATE_TARGET_HARD_FREQ) == 0:
                    self.Qmain.update_target_hard(self.Qtarget.model)

            total_reward += reward
            if self.done:
                self.epoch_rewards.append(total_reward)
                self.epoch_alive.append(alive_steps)
                print("Survived [{}] steps".format(alive_steps))
                print("Total reward [{}]".format(total_reward))
                alive_steps = 0
                total_reward = 0
            else:
                alive_steps += 1
            
            # Save the network every 1000 iterations
            if step > 0 and step % 1000 == 0:
                self.save(modelpath)

            # Iterate to next loop
            step += 1
            # Make new obs the current obs
            self.obs = new_obs
            self.state = new_state

        # Save model after all steps
        self.save(modelpath)

    def _batch_train(self, training_step, step):
        """Trains network to fit given parameters"""

        # Sample from experience buffer
        sample_batch = self.per_buffer.sample(self.batch_size, PER_BETA)
        s_batch = sample_batch[0]
        a_batch = sample_batch[1]
        r_batch = sample_batch[2]
        s2_batch = sample_batch[3]
        d_batch = sample_batch[4]
        w_batch = sample_batch[5]
        idx_batch = sample_batch[6]

        Q = np.zeros((self.batch_size, self.action_size))

        # Reshape frames to 1D
        input_size = self.observation_size * self.num_frames
        input_t = np.reshape(s_batch, (self.batch_size, input_size))
        input_t_1 = np.reshape(s2_batch, (self.batch_size, input_size))

        # Save the graph just the first time
        if training_step == 0:
            tf.summary.trace_on()

        # T Batch predict
        Q = self.Qmain.model.predict(input_t, batch_size = self.batch_size)

        ## Log graph once and disable graph logging
        if training_step == 0:
            with self.tf_writer.as_default():
                tf.summary.trace_export(self.name + "-graph", step)

        # T+1 batch predict
        Q1 = self.Qmain.model.predict(input_t_1, batch_size=self.batch_size)
        Q2 = self.Qtarget.model.predict(input_t_1, batch_size=self.batch_size)

        # Compute batch Qtarget using Double DQN
        for i in range(self.batch_size):
            doubleQ = Q2[i, np.argmax(Q1[i])]
            Q[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                Q[i, a_batch[i]] += DISCOUNT_FACTOR * doubleQ

        # Batch train
        loss = self.Qmain.train_on_batch(input_t, Q, w_batch)

        # Update PER buffer
        priorities = self.Qmain.batch_sq_error
        # Can't be zero, no upper limit
        priorities = np.clip(priorities, a_min=1e-8, a_max=None)
        self.per_buffer.update_priorities(idx_batch, priorities)

        # Log some useful metrics every even updates
        if step % (UPDATE_FREQ * 2) == 0:
            with self.tf_writer.as_default():
                mean_reward = np.mean(self.epoch_rewards)
                mean_alive = np.mean(self.epoch_alive)
                if len(self.epoch_rewards) >= 100:
                    mean_reward_100 = np.mean(self.epoch_rewards[-100:])
                    mean_alive_100 = np.mean(self.epoch_alive[-100:])
                else:
                    mean_reward_100 = mean_reward
                    mean_alive_100 = mean_alive
                tf.summary.scalar("mean_reward", mean_reward, step)
                tf.summary.scalar("mean_alive", mean_alive, step)
                tf.summary.scalar("mean_reward_100", mean_reward_100, step)
                tf.summary.scalar("mean_alive_100", mean_alive_100, step)
                tf.summary.scalar("loss", loss, step)
                tf.summary.scalar("lr", self.Qmain.train_lr, step)

            print("loss =", loss)

    def saved_actions_generator(self):
        saved_actions = np.zeros((803,494))#row refs to num_action, column refers to size_action_space
        LineNum = 59
        StationNum = 36
        GenNum = 22
        LoadNum = 37
        ElementNum = 177

        #first build line set action vector
        LineV = np.array([-1,11,12,21,22],dtype=float)
        #LineV = LineV[:,np.newaxis]
        for i in range(LineNum):
            saved_actions[i*5:(i+1)*5, i] = LineV

        #then build generators redispatch action vector
        gen_max_ramp_up = np.array([ 1.4,  0. ,  1.4, 10.4,  1.4,  0. ,  0. ,  0. ,  0. ,  0. ,  2.8, 0. ,  0. ,  2.8,  0. ,  0. ,  4.3,  0. ,  0. ,  2.8,  8.5,  9.9], dtype=float)
        #num2discrete = 5
        discretefactor = np.array([-0.95, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 0.95], dtype =float)#the action_num of each gen is 2*(num2discrete)
        gen_redisID = np.where(gen_max_ramp_up>0)[0]
        DispGenNum = len(gen_redisID)
        for i in range(len(gen_redisID)):
            saved_actions[(LineNum *5+ i*8):(LineNum *5+ (i+1)*8), 2*LineNum+2*ElementNum+gen_redisID[i]] = gen_max_ramp_up[gen_redisID[i]]*discretefactor#NB only when ramp_up = ramo_down this equation can be applied

        #build elements reconfiguration action vector
        Reconfig_action = np.array([1,2],dtype=float)
        action_bus_num = 9
        #action_bus = [4,7,9,21,23,26,28,29,33]
        # gen_location_num = [0,1,2,1,3,1,1,2,1]
        # load_location_num = [0,1,1,1,1,1,0,1,1]
        # line_location_num = [6,4,4,6,6,7,4,4,5]
        gen_action_id = [[],[1],[2,3],[9],[11,12,13],[14],[16],[17,18],[20]]
        load_action_id = [[],[8],[10],[22],[24],[27],[],[29],[31]]
        line_action_id = [[201,202,204,105,106,155],[206,207,108,109],[209,210,118,119],[226,227,228,129,130,136],[230,131,132,134,137,138],[236,237,238,239,140,141,156],[241,242,144,157],[243,244,150,151],[248,249,250,252,158]] 
        element_id = [[]] * action_bus_num
        bus_actionIndex = []
        for i in range(action_bus_num):
            element_id[i] = gen_action_id[i] + load_action_id[i] + line_action_id[i]
            combinsII = [c for c in combinations(element_id[i],2)]
            combinsIII = [d for d in combinations(element_id[i],3)]
            BusActCanlist = combinsII +  combinsIII
            for j in range(len(BusActCanlist)):
                lineChecklist = list(set(line_action_id[i]).intersection(set(list(BusActCanlist[j]))))
                if len(lineChecklist) >= 2:
                    bus_actionIndex.append(list(BusActCanlist[j]))
        for i in range(len(bus_actionIndex)):
            gen_reconfig_id = list(set(bus_actionIndex[i]).intersection(set(list(reduce(operator.add, gen_action_id)))))
            load_reconfig_id = list(set(bus_actionIndex[i]).intersection(set(list(reduce(operator.add, load_action_id)))))
            line_reconfig_id = list(set(bus_actionIndex[i]).intersection(set(list(reduce(operator.add, line_action_id)))))
            # append gen action
            if len(gen_reconfig_id) > 0:
                saved_actions[(LineNum*5 + DispGenNum*8 + i),2*LineNum + ElementNum + LoadNum + gen_reconfig_id[0]] = 1
            # append load action
            if len(load_reconfig_id) > 0:
                saved_actions[(LineNum*5 + DispGenNum*8 + i),2*LineNum + ElementNum + load_reconfig_id[0]] = 1
            # append line action
            for j in range(len(line_reconfig_id)):
                if line_reconfig_id[j]//100 == 1:
                    saved_actions[(LineNum*5 + DispGenNum*8 + i), 2*LineNum + ElementNum + LoadNum + GenNum + int(line_reconfig_id[j]%100)] = 1
                else:
                    saved_actions[(LineNum*5 + DispGenNum*8 + i), 2*LineNum + ElementNum + LoadNum + GenNum + LineNum + int(line_reconfig_id[j]%100)] = 1

        # #generators
        # for i in range(len(gen_action_id)):
            # saved_actions[(LineNum*5 + GenNum*8 + i*2):(LineNum*5 + GenNum*8 + (i+1)*2), 2*LineNum + 4*LineNum + 2*LoadNum + gen_action_id[i]] = Reconfig_action
        # #loads
        # for i in range(len(load_action_id)):
            # saved_actions[(LineNum*5 + GenNum*8 + i*2):(LineNum*5 + GenNum*8 + (i+1)*2), 2*LineNum + 4*LineNum + load_action_id[i]] = Reconfig_action
        # #lines
        # for j in range(len(bus_line_id)):
        #     element_actions[(GenNum*2 + LoadNum*2 + j*2):(GenNum*2 + LoadNum*2 + (j+1)*2), int(bus_line_id[j]%100)+int(bus_line_id[j]//100)-1] = Reconfig_action
        
        return saved_actions

    def act(self, observation, reward, done=False):
        transformed_observation = self.convert_obs(observation)
        encoded_act, Q_info = self.my_act(transformed_observation, reward, done)
        if not isinstance(Q_info,int):#若返回的为Q值
            encoded_act = 802
            chosen_action = self.agent_action_space({})
            currnt_ratio = observation.rho#获取当前线路负载率
            if any(currnt_ratio > 0.9):#若当前线路潮流负载率过高，则择优选择智能体仿真动作
                top_actions = np.argsort(Q_info)[-1: -4: -1].tolist()
                #top_actions.insert(0,802)#将不动作添加入最优解集首位，以其为基准规整数值
                max_score = float('-inf')
                for pre_action in tuple(top_actions):
                    pre_act = self.convert_act(pre_action)
                    print("当前检测动作编号",pre_action)
                    obs_simulate, reward_simulate, done_simulate, info_simulate= observation.simulate(pre_act)
                    overflow_penalty = 0.0
                    #calculated_socre = info_simulate["rewards"]["gameplay"]*100 + info_simulate["rewards"]["Overflow"]*50 + info_simulate["rewards"]["Redisp"]*0.01#由于当前仿真功能无法正常输出具体值，利用权重得到计算预测奖励值
                    if obs_simulate is None or info_simulate["is_illegal"] or info_simulate["is_ambiguous"]:#利用仿真信息构建gamepalyreward
                        gameplay_simu = -1.0
                        close2overflow_simu = 0.0
                    else:
                        gameplay_simu = 1.0
                        thermal_limits = np.array([ 43.3, 205.2, 341.2, 204. , 601.4, 347.1, 319.6, 301.4, 330.3,
                                                    274.1, 307.4, 172.3, 354.3, 127.9, 174.9, 152.6,  81.8, 204.3,
                                                    561.5, 561.5,  98.7, 179.8, 193.4, 239.9, 164.8, 100.4, 125.7,
                                                    278.2, 274. ,  89.9, 352.1, 157.1, 124.4, 154.6,  86.1, 106.7,
                                                    148.5, 129.6, 136.1,  86. , 313.2, 198.5, 599.1, 206.8, 233.7,
                                                    395.8, 516.7, 656.4, 583. , 583. , 263.1, 222.6, 322.8, 340.6,
                                                    305.2, 360.1, 395.8, 274.2, 605.5], dtype=float)
                        lineflow_ratio = obs_simulate.rho
                        close_to_overflow = 0.0
                        for ratio, limit in zip(lineflow_ratio, thermal_limits):
                            # Seperate big line and small line
                            if (limit < 400.00 and ratio > 0.90) or ratio >= 0.95:
                                close_to_overflow += 1.0
                        close_to_overflow = np.clip(close_to_overflow,
                                                    0.0, 5.0)
                        close2overflow_simu = np.interp(close_to_overflow,
                                                        [0.0, 5.0],
                                                        [1.0, 0.0])
                        has_overflow = any(obs_simulate.rho>1.0)
                        if has_overflow:
                            overflow_penalty = -10.0
                    calculated_socre = gameplay_simu*100 + close2overflow_simu*50 + reward_simulate*0.01 + overflow_penalty#由于当前仿真功能无法正常输出具体值，利用权重得到计算预测奖励值
                    #calculated_socre_trans = np.interp(calculated_socre, [-100.1, 155.05], [-10, 10])
                    reward_simulate = calculated_socre#利用计算得到的奖励值代替不正常输出的奖励值
                    print("当前检测动作预测奖励值",reward_simulate)
                    if  info_simulate['is_dispatching_illegal']:#目前仅根据当前动作常遇到的问题进行仿真动作的跳过
                        continue
                    else:
                        if not done_simulate and reward_simulate > max_score:
                            max_score = reward_simulate
                            chosen_action = pre_act
                            encoded_act = pre_action
            elif any(currnt_ratio > 0.8):#若当前线路潮流负载率较高，则从不动作和智能体最佳动作择优选择
                top_actions = np.argsort(Q_info)[-1: -2: -1].tolist()
                top_actions.insert(0,802)#将不动作添加入最优解集首位，以其为基准规整数值
                max_score = float('-inf')
                for pre_action in tuple(top_actions):
                    pre_act = self.convert_act(pre_action)
                    print("当前检测动作编号",pre_action)
                    obs_simulate, reward_simulate, done_simulate, info_simulate= observation.simulate(pre_act)
                    overflow_penalty = 0.0
                    #calculated_socre = info_simulate["rewards"]["gameplay"]*100 + info_simulate["rewards"]["Overflow"]*50 + info_simulate["rewards"]["Redisp"]*0.01#由于当前仿真功能无法正常输出具体值，利用权重得到计算预测奖励值
                    if obs_simulate is None or info_simulate["is_illegal"] or info_simulate["is_ambiguous"]:#利用仿真信息构建gamepalyreward
                        gameplay_simu = -1.0
                        close2overflow_simu = 0.0
                    else:
                        gameplay_simu = 1.0
                        thermal_limits = np.array([ 43.3, 205.2, 341.2, 204. , 601.4, 347.1, 319.6, 301.4, 330.3,
                                                    274.1, 307.4, 172.3, 354.3, 127.9, 174.9, 152.6,  81.8, 204.3,
                                                    561.5, 561.5,  98.7, 179.8, 193.4, 239.9, 164.8, 100.4, 125.7,
                                                    278.2, 274. ,  89.9, 352.1, 157.1, 124.4, 154.6,  86.1, 106.7,
                                                    148.5, 129.6, 136.1,  86. , 313.2, 198.5, 599.1, 206.8, 233.7,
                                                    395.8, 516.7, 656.4, 583. , 583. , 263.1, 222.6, 322.8, 340.6,
                                                    305.2, 360.1, 395.8, 274.2, 605.5], dtype=float)
                        lineflow_ratio = obs_simulate.rho
                        close_to_overflow = 0.0
                        for ratio, limit in zip(lineflow_ratio, thermal_limits):
                            # Seperate big line and small line
                            if (limit < 400.00 and ratio > 0.90) or ratio >= 0.95:
                                close_to_overflow += 1.0
                        close_to_overflow = np.clip(close_to_overflow,
                                                    0.0, 5.0)
                        close2overflow_simu = np.interp(close_to_overflow,
                                                        [0.0, 5.0],
                                                        [1.0, 0.0])
                        has_overflow = any(obs_simulate.rho>1.0)
                        if has_overflow:
                            overflow_penalty = -10.0
                    calculated_socre = gameplay_simu*100 + close2overflow_simu*50 + reward_simulate*0.01 + overflow_penalty#由于当前仿真功能无法正常输出具体值，利用权重得到计算预测奖励值
                    #calculated_socre_trans = np.interp(calculated_socre, [-100.1, 155.05], [-10, 10])
                    reward_simulate = calculated_socre#利用计算得到的奖励值代替不正常输出的奖励值
                    print("当前检测动作预测奖励值",reward_simulate)
                    if  info_simulate['is_dispatching_illegal']:#目前仅根据当前动作常遇到的问题进行仿真动作的跳过
                        continue
                    else:
                        if not done_simulate and reward_simulate > max_score:
                            max_score = reward_simulate
                            chosen_action = pre_act
                            encoded_act = pre_action
        return self.convert_act(encoded_act)