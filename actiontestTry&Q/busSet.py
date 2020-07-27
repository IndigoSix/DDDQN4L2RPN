# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:22:01 2020

@author: IndigoSix
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
import operator
from functools import reduce
# import grid2op

# multimix_env = grid2op.make("l2rpn_neurips_2020_track2_small")
# gen_to_subid = multimix_env.action_space.gen_to_subid
# load_to_subid = multimix_env.action_space.load_to_subid
# line_or_to_subid = multimix_env.action_space.line_or_to_subid
# line_ex_to_subid = multimix_env.action_space.line_ex_to_subid

#读取各元素与变电站的连接关系（若在环境中，可直接使用上述注释的代码生成）
gen_to_subid=np.load("Gen.npy", allow_pickle=True)
load_to_subid=np.load("Load.npy", allow_pickle=True)
line_or_to_subid=np.load("Line_or.npy", allow_pickle=True)
line_ex_to_subid=np.load("Line_ex.npy", allow_pickle=True)


LineNum = 186
GenNum = 62
LoadNum = 99
SubNum = 118
ElementNum = 533
#生成各变电站连接元素的编号列表（使用了不同千位数相加保证不同类型元素索引的辨识度）
gen_action_id = []
load_action_id = []
line_action_id = []
for i in range(SubNum):
    Gindex = np.where(gen_to_subid == i)[0]
    Lindex = np.where(load_to_subid == i)[0]
    L_OR_index = np.where(line_or_to_subid == i)[0]
    L_EX_index = np.where(line_ex_to_subid == i)[0]
    if len(Gindex) > 0:
        for j in range(len(Gindex)):
           Gindex[j] = Gindex[j] + 8000
    Gindex = Gindex.tolist()
    if len(Lindex) > 0:
        for j in range(len(Lindex)):
           Lindex[j] = Lindex[j] + 6000
    Lindex = Lindex.tolist()
    if len(L_OR_index) > 0:
        for j in range(len(L_OR_index)):
            L_OR_index[j] = L_OR_index[j] + 1000
    L_OR_index =  L_OR_index.tolist()
    if len(L_EX_index) > 0:
        for j in range(len(L_EX_index)):
            L_EX_index[j] = L_EX_index[j] + 2000
    L_EX_index =  L_EX_index.tolist()
    Lineindex = L_OR_index + L_EX_index
    gen_action_id.append(Gindex)
    load_action_id.append(Lindex)
    line_action_id.append(Lineindex)


    
    
#build elements reconfiguration action vector
action_bus_num = 118
element_id = [[]] * action_bus_num
bus_actionIndex = []#总的母线动作索引集合
for i in range(action_bus_num):
    sub_bus_actionIndex = []#当前变电站母线动作索引集合
    element_id[i] = gen_action_id[i] + load_action_id[i] + line_action_id[i]
    MaxActElenum = int(len(element_id[i])//2)
    BusActCanlist = []
    if MaxActElenum > 1:#仅有连接元素大于等于4个元素的变电站方可进行母线分裂，否则会出现源荷孤立
        for j in range(MaxActElenum - 1):
            combins = [c for c in combinations(element_id[i],j+2)]
            BusActCanlist += combins
    for j in range(len(BusActCanlist)):
        lineChecklist = list(set(line_action_id[i]).intersection(set(list(BusActCanlist[j]))))
        if len(lineChecklist) >= 1:#只有当动作元素中涉及到不少于1条线路时，动作才可认为合理（还要进一步明确动作元素与非动作元素的线路包含关系才可进一步确定动作的合理性）
            bus_actionIndex.append(list(BusActCanlist[j]))
            sub_bus_actionIndex.append(list(BusActCanlist[j]))
    SubBusActionNum = len(sub_bus_actionIndex)
    sub_saved_actions = np.zeros((SubBusActionNum,1500))#1500为actionspace_size，对应动作顺序为[lineset linechange busset buschange redisp]
#进行当前变电站动作矩阵的生成
    for k in range(len(sub_bus_actionIndex)):
        gen_reconfig_id = list(set(sub_bus_actionIndex[k]).intersection(set(list(reduce(operator.add, gen_action_id)))))
        load_reconfig_id = list(set(sub_bus_actionIndex[k]).intersection(set(list(reduce(operator.add, load_action_id)))))
        line_reconfig_id = list(set(sub_bus_actionIndex[k]).intersection(set(list(reduce(operator.add, line_action_id)))))
        # append gen action
        if len(gen_reconfig_id) > 0:
            for j in range(len(gen_reconfig_id)):
                sub_saved_actions[k,2*LineNum + ElementNum + LoadNum + int(gen_reconfig_id[j]%1000)] = 1
        # append load action
        if len(load_reconfig_id) > 0:
            for j in range(len(load_reconfig_id)):
                sub_saved_actions[k,2*LineNum + ElementNum + int(load_reconfig_id[j]%1000)] = 1
        # append line action
        for j in range(len(line_reconfig_id)):
            if line_reconfig_id[j]//1000 == 1:
                sub_saved_actions[k, 2*LineNum + ElementNum + LoadNum + GenNum + int(line_reconfig_id[j]%1000)] = 1
            else:
                sub_saved_actions[k, 2*LineNum + ElementNum + LoadNum + GenNum + LineNum + int(line_reconfig_id[j]%1000)] = 1
    print("Number of bus actions in subID {}——{}".format(i,SubBusActionNum))
    np.save("Sub{}".format(i),sub_saved_actions)


# TotalBusActionNum = len(bus_actionIndex)
# saved_actions = np.zeros((TotalBusActionNum + 1,1500))#1500为actionspace_size
# for i in range(len(bus_actionIndex)):
    # gen_reconfig_id = list(set(bus_actionIndex[i]).intersection(set(list(reduce(operator.add, gen_action_id)))))
    # load_reconfig_id = list(set(bus_actionIndex[i]).intersection(set(list(reduce(operator.add, load_action_id)))))
    # line_reconfig_id = list(set(bus_actionIndex[i]).intersection(set(list(reduce(operator.add, line_action_id)))))
    # # append gen action
    # if len(gen_reconfig_id) > 0:
        # saved_actions[i,2*LineNum + ElementNum + LoadNum + gen_reconfig_id[0]] = 1
    # # append load action
    # if len(load_reconfig_id) > 0:
        # saved_actions[i,2*LineNum + ElementNum + load_reconfig_id[0]] = 1
    # # append line action
    # for j in range(len(line_reconfig_id)):
        # if line_reconfig_id[j]//1000 == 1:
            # saved_actions[i, 2*LineNum + ElementNum + LoadNum + GenNum + int(line_reconfig_id[j]%1000)] = 1
        # else:
            # saved_actions[i, 2*LineNum + ElementNum + LoadNum + GenNum + LineNum + int(line_reconfig_id[j]%1000)] = 1

