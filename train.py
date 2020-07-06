#!/usr/bin/env python3

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import argparse
import tensorflow as tf

from l2rpn_baselines.DoubleDuelingDQN.DoubleDuelingDQN import DoubleDuelingDQN as DDDQNAgent
#from grid2op.Chronics.MultiFolder import Multifolder

DEFAULT_NAME = "DoubleDuelingDQN"
DEFAULT_SAVE_DIR = "./models"
DEFAULT_LOG_DIR = "./logs-train"
DEFAULT_PRE_STEPS = 256
DEFAULT_TRAIN_STEPS = 10000#1024
DEFAULT_N_FRAMES = 2
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-5


def cli():
    parser = argparse.ArgumentParser(description="Train baseline DDQN")
    # Paths
    parser.add_argument("--name", default=DEFAULT_NAME,
                        help="The name of the model")
    parser.add_argument("--data_dir", default="rte_case14_realistic",
                        help="Path to the dataset root directory")
    parser.add_argument("--save_dir", required=False,
                        default=DEFAULT_SAVE_DIR, type=str,
                        help="Directory where to save the model")
    parser.add_argument("--load_file", required=False,
                        default= None,#"DoubleDuelingDQN.h5",#
                        help="Path to model.h5 to resume training with")
    parser.add_argument("--logs_dir", required=False,
                        default=DEFAULT_LOG_DIR, type=str,
                        help="Directory to save the logs")
    # Params
    parser.add_argument("--num_pre_steps", required=False,
                        default=DEFAULT_PRE_STEPS, type=int,
                        help="Number of random steps before training")
    parser.add_argument("--num_train_steps", required=False,
                        default=DEFAULT_TRAIN_STEPS, type=int,
                        help="Number of training iterations")
    parser.add_argument("--num_frames", required=False,
                        default=DEFAULT_N_FRAMES, type=int,
                        help="Number of stacked states to use during training")
    parser.add_argument("--batch_size", required=False,
                        default=DEFAULT_BATCH_SIZE, type=int,
                        help="Mini batch size (defaults to 1)")
    parser.add_argument("--learning_rate", required=False,
                        default=DEFAULT_LR, type=float,
                        help="Learning rate for the Adam optimizer")
    return parser.parse_args()


def train(env,
          name = DEFAULT_NAME,
          iterations = DEFAULT_TRAIN_STEPS,
          save_path = DEFAULT_SAVE_DIR,
          load_path = None,
          logs_path = DEFAULT_LOG_DIR,
          num_pre_training_steps = DEFAULT_PRE_STEPS,
          num_frames = DEFAULT_N_FRAMES,
          batch_size= DEFAULT_BATCH_SIZE,
          learning_rate= DEFAULT_LR):

    # Limit gpu usage
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    agent = DDDQNAgent(env.observation_space,
                       env.action_space,
                       name=name,
                       is_training=True,
                       batch_size=batch_size,
                       num_frames=num_frames,
                       lr=learning_rate)

    if load_path is not None:
        agent.load(load_path)
        print("load success!")

    agent.train(env,
                iterations,
                save_path,
                num_pre_training_steps,
                logs_path)


if __name__ == "__main__":
    from grid2op.MakeEnv import make
    from grid2op.Reward import *
    from grid2op.Action import *
    from grid2op.Parameters import Parameters
    #from lightsim2grid import LightSimBackend
    import sys

    args = cli()
    # Use custom params
    #params = Parameters()
    #params.MAX_SUB_CHANGED = 2

    # Create grid2op game environement
    env = make("D:/NewTasks/L2RPN/code/l2rpn_wcci_2020",#args.data_dir,
               difficulty="competition",#param=params,
               #backend=LightSimBackend(),
               #action_class=PowerlineSetAndDispatchAction,#TopologyChangeAndDispatchAction,
               reward_class=CombinedScaledReward,
               other_rewards={"gameplay": GameplayReward, "Overflow":CloseToOverflowReward, "Redisp": RedispReward})

    # Only load 128 steps in ram
    env.chronics_handler.set_chunk_size(128)

    # Register custom reward for training
    cr = env.reward_helper.template_reward
    cr.addReward("Overflow",CloseToOverflowReward(), 50.0)
    cr.addReward("gameplay", GameplayReward(), 100.0)
    cr.addReward("Redisp", RedispReward(), 1e-2)
    # Initialize custom rewards
    cr.initialize(env)
    # Set reward range to something managable
    cr.set_range(-10.0, 10.0)

    train(env,
          name = args.name,
          iterations = args.num_train_steps,
          num_pre_training_steps = args.num_pre_steps,
          save_path = args.save_dir,
          load_path = args.load_file,
          logs_path = args.logs_dir,
          num_frames = args.num_frames,
          batch_size = args.batch_size,
          learning_rate = args.learning_rate)
