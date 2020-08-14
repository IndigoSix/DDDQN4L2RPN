import numpy as np
import itertools
import grid2op
import pandas as pd
from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Exceptions import AmbiguousAction, Grid2OpException
from grid2op.Space import SerializableSpace
from grid2op.Action.BaseAction import BaseAction

LineNum = 20
GenNum = 6
LoadNum = 11
SubNum = 14
ElementNum = 57

n_features = 3

env = grid2op.make("C:/Users/IndigoSix/data_grid2op/l2rpn_case14_sandbox")
subNum = 14
# CumEleNum = np.zeros((1,subNum+1))
# CumEleNum = CumEleNum[0]
CumEleNum = [0]
EleNum = env.action_space.sub_info
for sub_id, num_el in enumerate(env.action_space.sub_info):
    #CumEleNum[sub_id+1] = int(sum(EleNum[:sub_id+1]))
    CumEleNum.append(int(sum(EleNum[:sub_id+1])))
#CumEleNum = list(CumEleNum)
print(len(CumEleNum))

#进行各类设备编号向图数据中索引编号的转换
Line_or_to_Graph = np.full(shape=LineNum, fill_value=0, dtype=dt_int)
for i in range(LineNum):
    subID = env.action_space.line_or_to_subid[i]
    subInternalID = env.action_space.line_or_to_sub_pos[i]
    Line_or_to_Graph[i] = CumEleNum[subID] + subInternalID

Line_ex_to_Graph = np.full(shape=LineNum, fill_value=0, dtype=dt_int)
for i in range(LineNum):
    subID = env.action_space.line_ex_to_subid[i]
    subInternalID = env.action_space.line_ex_to_sub_pos[i]
    Line_ex_to_Graph[i] = CumEleNum[subID] + subInternalID

Gen_to_Graph = np.full(shape=GenNum, fill_value=0, dtype=dt_int)
for i in range(GenNum):
    subID = env.action_space.gen_to_subid[i]
    subInternalID = env.action_space.gen_to_sub_pos[i]
    Gen_to_Graph[i] = CumEleNum[subID] + subInternalID

Load_to_Graph = np.full(shape=LoadNum, fill_value=0, dtype=dt_int)
for i in range(LoadNum):
    subID = env.action_space.load_to_subid[i]
    subInternalID = env.action_space.load_to_sub_pos[i]
    Load_to_Graph[i] = CumEleNum[subID] + subInternalID

print(Line_or_to_Graph)
print(Line_ex_to_Graph)
print(Gen_to_Graph)
print(Load_to_Graph)
np.save("Line_or_to_Graph",Line_or_to_Graph)
np.save("Line_ex_to_Graph",Line_ex_to_Graph)
np.save("Gen_to_Graph",Gen_to_Graph)
np.save("Load_to_Graph",Load_to_Graph)

# GraphX = np.zeros((ElementNum,n_features))#初始化节点特征矩阵
# GraphX[Line_or_to_Graph, 0] = obs.p_or
# GraphX[Line_or_to_Graph, 1] = obs.q_or
# GraphX[Line_or_to_Graph, 2] = obs.v_or
# GraphX[Line_ex_to_Graph, 0] = obs.p_ex
# GraphX[Line_ex_to_Graph, 1] = obs.q_ex
# GraphX[Line_ex_to_Graph, 2] = obs.v_ex
# GraphX[Gen_to_Graph, 0] = obs.prod_p
# GraphX[Gen_to_Graph, 1] = obs.prod_q
# GraphX[Gen_to_Graph, 2] = obs.prod_v
# GraphX[Load_to_Graph, 0] = obs.load_p
# GraphX[Load_to_Graph, 1] = obs.load_q
# GraphX[Load_to_Graph, 2] = obs.load_v
# print(GraphX)

