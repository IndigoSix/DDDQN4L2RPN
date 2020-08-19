import os
import numpy as np
import itertools
import grid2op
import pandas as pd
from grid2op.dtypes import dt_int, dt_float, dt_bool
from grid2op.Exceptions import AmbiguousAction, Grid2OpException
from grid2op.Space import SerializableSpace
from grid2op.Action.BaseAction import BaseAction

from env_info import Env_Info
from config import Config

env = grid2op.make(Config.env_name)

CumEleNum = [0]
EleNum = env.action_space.sub_info
for sub_id, num_el in enumerate(env.action_space.sub_info):
    #CumEleNum[sub_id+1] = int(sum(EleNum[:sub_id+1]))
    CumEleNum.append(int(sum(EleNum[:sub_id+1])))
#CumEleNum = list(CumEleNum)
# print(len(CumEleNum))

grid_info = Env_Info(env)
LineNum = grid_info.LineNum
GenNum = grid_info.GenNum
LoadNum = grid_info.LoadNum
SubNum = grid_info.SubNum
ElementNum = grid_info.ElementNum

def get_all_unitary_topologies_set(action_space, CumEleNum):
    """
    This methods allows to compute and return all the unitary topological changes that can be performed on a
    powergrid.

    The changes will be performed using the "set_bus" method. The "do nothing" action will be counted once
    per substation in the grid.

    Parameters
    ----------
    action_space: :class:`grid2op.BaseAction.ActionHelper`
        The action space used.

    Returns
    -------
    res: ``list``
        The list of all the topological actions that can be performed.

    """
    if not os.path.exists("actions/"+Config.env_name + "/"):
        os.makedirs("actions/"+Config.env_name + "/")

    res = []
    S = [0, 1]
    for sub_id, num_el in enumerate(action_space.sub_info):
        
        tmp = []
        tmp_array = []
        new_topo = np.full(shape=num_el, fill_value=1, dtype=dt_int)
        # perform the action "set everything on bus 1"
        action = action_space({"set_bus": {"substations_id": [(sub_id, new_topo)]}})
        tmp.append(action)
        tmp_array.append(new_topo)

        powerlines_or_id = action_space.line_or_to_sub_pos[action_space.line_or_to_subid == sub_id]
        powerlines_ex_id = action_space.line_ex_to_sub_pos[action_space.line_ex_to_subid == sub_id]
        powerlines_id = np.concatenate((powerlines_or_id, powerlines_ex_id))

        # computes all the topologies at 2 buses for this substation
        for tup in itertools.product(S, repeat=num_el - 1):
            indx = np.full(shape=num_el, fill_value=False, dtype=dt_bool)
            tup = np.array((0, *tup)).astype(dt_bool)  # add a zero to first element -> break symmetry
            indx[tup] = True
            if np.sum(indx) >= 2 and np.sum(~indx) >= 2:
                # i need 2 elements on each bus at least (almost all the times, except when a powerline
                # is alone on its bus)
                new_topo = np.full(shape=num_el, fill_value=1, dtype=dt_int)
                new_topo[~indx] = 2

                if np.sum(indx[powerlines_id]) == 0 or np.sum(~indx[powerlines_id]) == 0:
                    # if there is a "node" without a powerline, the topology is not valid
                    continue

                action = action_space({"set_bus": {"substations_id": [(sub_id, new_topo)]}})
                tmp.append(action)
                tmp_array.append(new_topo)
            else:
                # i need to take into account the case where 1 powerline is alone on a bus too
                if np.sum(indx[powerlines_id]) >= 1 and np.sum(~indx[powerlines_id]) >= 1:
                    new_topo = np.full(shape=num_el, fill_value=1, dtype=dt_int)
                    new_topo[~indx] = 2
                    action = action_space({"set_bus": {"substations_id": [(sub_id, new_topo)]}})
                    tmp.append(action)
                    tmp_array.append(new_topo)


        if len(tmp) >= 2:
            # if i have only one single topology on this substation, it doesn't make any action
            # i cannot change the topology is there is only one.
            res += tmp
            total_col = LineNum * 2 + ElementNum * 2 + GenNum
            sub_saved_actions = np.zeros((len(tmp), total_col))
            for j in range(len(tmp)):
                sub_saved_actions[j,(2*LineNum + CumEleNum[sub_id]):(2*LineNum + CumEleNum[sub_id+1])] = tmp_array[j]
        else:
            sub_saved_actions = np.zeros((len(tmp)-1, total_col))
        print("Sub{} with {} actions saved".format(sub_id, sub_saved_actions.shape[0]))
        np.save("actions/"+Config.env_name+"/Sub{}".format(sub_id), sub_saved_actions)

    return res
    
# All_unitary_bus_sets = get_all_unitary_topologies_set(env.action_space, CumEleNum)

# action_mat 是矩阵,需要将 get_all_unitary_topologies_set() save 的变电站.npy文件读取到 action_mat 的对应位置中
def convert_act(a_i, action_space, action_mat, CumEleNumList = CumEleNum):#当前只能支持对1个动作的转换
    action_array = action_mat[a_i]
    # Transfer the l2rpn action array to a dict of grid2op
    action_dict = {}
    obj_id_dict = {}
    real_action = action_space(action_dict) # 首先初始时默认不动作
    # 当前只能支持对1个动作的转换，故线路/发电机的 id_list 长度应该只能为 1
    # line status set
    offset = 0
    set_lines_status_array = action_array[:LineNum] #LineNum = 20
    # 取 action_array 前 LineNum 个动作，因其为对应的线路 set 操作
    set_lines_id_list = list(np.where(set_lines_status_array != 0)[0])   
    if len(set_lines_id_list) != 0:
        print("line op!!!")
        for i in range(len(set_lines_id_list)):
            if set_lines_status_array[set_lines_id_list[i]] == -1:
                real_action = action_space.disconnect_powerline(line_id = set_lines_id_list[i])
            else:
                real_action = action_space.reconnect_powerline(line_id=set_lines_id_list[i], bus_ex=int(set_lines_status_array[set_lines_id_list[i]]//10), bus_or=int(set_lines_status_array[set_lines_id_list[i]]%10))
    
    # generator redipatch
    offset += 2*LineNum+2*ElementNum #2*LineNum+2*ElementNum = 154
    generator_redipatch_array = action_array[offset:]
    generator_redipatch_id_list = list(np.where(generator_redipatch_array != 0)[0])
    if len(generator_redipatch_id_list) != 0:
        print("Gen op!!!")
        for i in range(len(generator_redipatch_id_list)):
            print("Gen redispatch ID:",generator_redipatch_id_list[i],"Gen redispatch amount:",generator_redipatch_array[generator_redipatch_id_list[i]])
            real_action = action_space({"redispatch": [(generator_redipatch_id_list[i], generator_redipatch_array[generator_redipatch_id_list[i]])]})

    # load reconfigration
    offset -= ElementNum #ElementNum
    change_load_array = action_array[offset:offset+11]
    change_load_id_list = list(np.where(change_load_array == 1)[0])
    if len(change_load_id_list) != 0:
        obj_id_dict["loads_id"] = [i for i in change_load_id_list]

    # generator reconfiguration
    offset += LoadNum #LoadNum
    change_gen_array = action_array[offset:offset+6] 
    change_gen_id_list = list(np.where(change_gen_array == 1)[0])
    if len(change_load_id_list) != 0:
        obj_id_dict["generators_id"] = [i for i in change_gen_id_list]
    
    # line ox reconfiguration
    offset += GenNum #GenNum
    change_lines_or_array = action_array[offset:offset+20]
    change_or_id_list = list(np.where(change_lines_or_array == 1)[0])
    if len(change_or_id_list) != 0:
        obj_id_dict["lines_or_id"] = [i for i in change_or_id_list]

      #print(obj_id_dict["lines_or_id"])
    # line ex
    offset += LineNum #LineNum
    change_lines_ex_array = action_array[offset:offset+20]
    change_ex_id_list = list(np.where(change_lines_ex_array == 1)[0])
    if len(change_ex_id_list) != 0:
        obj_id_dict["lines_ex_id"] = [i for i in change_ex_id_list]
    
    if  len(obj_id_dict) != 0:
        action_dict["change_bus"] = obj_id_dict
        real_action = action_space(action_dict)
        
    #sub bus actions set
    offset = 2*LineNum
    bus_set_array = action_array[offset:offset+ElementNum]
    bus_set_id_list = list(np.where(bus_set_array > 0)[0])
    if len(bus_set_id_list) != 0:
        set_topo = bus_set_array[bus_set_id_list]
        set_sub_id = CumEleNumList.index(bus_set_id_list[0])
        real_action = action_space({"set_bus": {"substations_id": [(set_sub_id, set_topo)]}})
        
    return real_action