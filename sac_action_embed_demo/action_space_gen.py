import numpy as np
from config import Config

def gen_topo_actions(SubNum):
    path = "actions/"+Config.env_name+"/Sub"
    topo_actions = np.load(path + "0.npy")
    for i in range(1, SubNum):
        temp = np.load(path + str(i) + ".npy")
        topo_actions = np.vstack([topo_actions, temp])
    return topo_actions

def saved_actions_generator(grid_info):

    LineNum = grid_info.LineNum
    GenNum = grid_info.GenNum
    LoadNum = grid_info.LoadNum
    SubNum = grid_info.SubNum
    ElementNum = grid_info.ElementNum
    gen_max_ramp_up = np.array(grid_info.GenRampUp)
    gen_redisID = np.where(gen_max_ramp_up > 0)[0]
    valid_gen_num = len(gen_redisID)
    total_col = LineNum * 2 + ElementNum * 2 + GenNum
    # saved_actions = np.zeros((LineNum*5+valid_gen_num*8+1,total_col))#row refs to num_action, column refers to size_action_space

    lines_actions = np.zeros((LineNum*5, total_col))
    topo_actions = gen_topo_actions(SubNum)
    gen_actions = np.zeros((valid_gen_num*8, total_col))
    do_nothing_action = np.zeros((1, total_col))

    #first build line set action vector
    LineV = np.array([-1, 11, 12, 21, 22],dtype=float)
    #LineV = LineV[:,np.newaxis]
    for i in range(LineNum):
        lines_actions[i*5:(i+1)*5, i] = LineV

    #num2discrete = 5
    discretefactor = np.array([-0.95, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 0.95], dtype =float)#the action_num of each gen is 2*(num2discrete)

    for i in range(len(gen_redisID)):
        gen_actions[(i*8) : ((i+1)*8), 2*LineNum+2*ElementNum+gen_redisID[i]] = \
            gen_max_ramp_up[gen_redisID[i]]*discretefactor#NB only when ramp_up = ramo_down this equation can be applied

    # action_mat = np.vstack([lines_actions, topo_actions, gen_actions, do_nothing_action])
    action_mat = np.vstack([topo_actions, do_nothing_action])

    return action_mat