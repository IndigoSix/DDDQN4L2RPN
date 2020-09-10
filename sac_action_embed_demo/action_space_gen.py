import numpy as np
from config import Config
from copy import deepcopy

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

class gen_actions_generator():
    # 仅生成发电机set动作，连接同一子站的多台发电机采取统一动作

    def __init__(self, grid_info):
        self.GenNum = grid_info.GenNum
        self.GenRampUp = grid_info.GenRampUp
        self.GenRampDown = grid_info.GenRampDown
        self.GenRedispable = grid_info.GenRedispable
        self.Gen2Sub = grid_info.Gen2Sub
        self.gen_p_max = grid_info.gen_p_max
        self.gen_p_min = grid_info.gen_p_min

        self.adjust_num, self.adjust_id, self.adjust_mat, self.del_id= self.create_adjust_para()
        self.genid = list()
        for i in range(self.GenNum):
            self.genid.append(i)
        for i in sorted(self.del_id, reverse=True):
            del self.genid[i]

    def create_adjust_para(self):
        del_id = list()
        adjust_mat = np.array([int(x) for x in self.GenRedispable])
        for i in range(len(adjust_mat)):
            if adjust_mat[i] == 0:
                adjust_mat[i] = -1
                del_id.append(i)
        adjust_id = list()
        for i in range(len(adjust_mat)):
            if adjust_mat[i] == 1:
                adjust_id.append(self.Gen2Sub[i])
        for i in range(len(adjust_mat)):
            if adjust_mat[i] == 1:
                adjust_mat[i] = self.Gen2Sub[i]
            else:
                adjust_mat[i] = -100
        adjust_id = list(set(adjust_id))
        adjust_num = len(adjust_id)
        return adjust_num, adjust_id, adjust_mat, del_id

    def convert_act(self, a_hat, action_space, gen_p):
        assert len(a_hat[0]) == self.adjust_num
        overflow = 0
        act = 0.*np.zeros_like(self.adjust_mat)
        # 根据SAC输出正负，区分发电机正反向爬坡上限
        for i in range(self.adjust_num):
            idx = np.where(self.adjust_mat == self.adjust_id[i])[0].tolist()
            # TODO (act = a_hat[i]*GenRampUp/Down[] × k),k>1，防止sac输出边界值导致梯度消失，采取np.clip()裁剪输出
            if a_hat[0][i] > 0:
                act[idx] = a_hat[0][i]*self.GenRampUp[idx]     # 输出为正，RampUp
            else:
                act[idx] = a_hat[0][i]*self.GenRampDown[idx]   # 输出为负，RampDown
        # 检验发电机出力是否到达上边界/下边界
        pre_gen_p = np.array(gen_p) + act
        if any(pre_gen_p < self.gen_p_min):
            idx = pre_gen_p < self.gen_p_min
            act[idx] = self.gen_p_min[idx] - gen_p[idx]
            pre_gen_p = np.array(gen_p) + act
            overflow = -1   # 功率溢出标志，-1--下界溢出，1--上界溢出，0--无溢出

        if any(pre_gen_p > self.gen_p_max):
            idx = pre_gen_p > self.gen_p_max
            act[idx] = self.gen_p_max[idx] - gen_p[idx]
            pre_gen_p = np.array(gen_p) + act
            overflow = 1

        p_ramp = act.tolist()
        for i in sorted(self.del_id, reverse=True):
            del p_ramp[i]
        assert len(self.genid) == len(p_ramp)
        act_value = [(idn, dp) for idn, dp in zip(self.genid, p_ramp)]
        action = action_space({"redispatch": act_value})

        return action, overflow