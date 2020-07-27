import numpy as np
import grid2op
from grid2op.Agent import DoNothingAgent
#在线运行测试及产生数据
class DummyAgent(DoNothingAgent):
    def __init__(self, action_space):
        super().__init__(action_space)

multimix_env = grid2op.make("l2rpn_neurips_2020_track2_small")
multimix_env.seed(seed=1)
MAX_STEPS = 8062
LineNum = 186
GenNum = 62
LoadNum = 99
SubNum = 118
ElementNum = 533
#my_agent = DummyAgent(multimix_env.action_space)

#进行场景运行及故障点分析的函数
def OPandAnalysis(env, beginT = 0, targetact= np.zeros((1,1500)), OPflag = 1, PreActlist = []):
    my_agent = DummyAgent(env.action_space)
    NUM_EPISODES = 1
    MAX_STEPS = 8062
    i = 0
    transitions = []
    j = 0
    while i < NUM_EPISODES:
        for mix in env:
            # Get number of chronics in this mix
            n_chronics_mix = len(mix.chronics_handler.subpaths)
            # Adjust iterations to limit to NB_EPISODES
            if (i + n_chronics_mix) > NUM_EPISODES:
                n_chronics_mix = NUM_EPISODES - i
            
            # Iterate over this mix scenarios
            for c in range(n_chronics_mix):
                mix.reset()
                #mix.fast_forward_chronics(beginT)#按照需要校验动作的介入时间前推环境
            
                # Print some info on current episode
                mix_name = mix.name
                chronic_name = mix.chronics_handler.get_name()
                print ("Episode [{}] - Mix [{}] - Chronic [{}]".format(i, mix_name, chronic_name))

                done = False
                obs = mix.current_obs
                reward = 0.0
                #step = beginT
                step = 0#若采用动作序列，则每次场景均要从初始时刻开始
                print("step——",step)
                while done is False and step < MAX_STEPS:
                    PreActionflag = 0
                    for xx in range(len(PreActlist)):#遍历历史动作列表中的元素，若当前时刻存在于动作列表，则应启用当前时刻对应的动作
                        if not isinstance(PreActlist[xx],int):
                            continue
                        elif PreActlist[xx] == step:
                            PreActionflag = 1
                            break
                    if step == beginT:#若为动作介入时刻，采取待校验动作，其余时刻采取donothing
                        agent_action = convert_act(env.action_space,targetact)
                        print("采取动作",agent_action)
                    elif PreActionflag == 1:#若为历史动作时刻，则采取选中的历史动作部署到系统
                        agent_action = convert_act(env.action_space,PreActlist[xx-1])
                    else:
                        agent_action = my_agent.act(obs, reward, done)
                    transitions.append(obs.rho)
                    obs, reward, done , info = mix.step(agent_action)
                    transitions.append(reward)
                    transitions.append(obs.rho)
                    transitions.append(done)
                    transitions.append(step)
                    step += 1
                i += 1
            if j == 0:
                print("GameOverStep：",step)
                break
    if OPflag:#仅有系统在序贯运行而非遍历动作时再保存故障录波数据
        np.save("Preinfo",transitions)

    #离线分析
    line_or_to_subid = env.action_space.line_or_to_subid
    line_ex_to_subid = env.action_space.line_ex_to_subid
    A=np.load("Preinfo.npy", allow_pickle=True)
    tstep = int(len(A)/5)
    Findex = np.where(A[-5]==0)[0]#记录故障前最后时刻断开线路的编号
    Histime = 10
    rhoHis = np.zeros((len(Findex),Histime))#记录最后断开线路在系统崩溃前的Histime个步长的负载率变化数据
    for i in range(Histime):
        rhoCurrent = A[(tstep - i -1)*5][Findex]
        rhoHis[:,Histime-i-1] = rhoCurrent
    Totalrho = np.sum(rhoHis,axis = 1)
    Totalrho[np.where(Totalrho==0)[0]] = 100#将负载率累加为0（即一直断开）的线路负载率置为大值，搜索总负载率最小的线路作为目标线路
    RiskLineID = Findex[np.argmin(Totalrho)]#结合线路负载率的变化趋势，找出最初断开的线路（源头线路）编号以进行针对性的处理
    print("目标线路对应变电站编号1：",line_or_to_subid[RiskLineID],"目标线路对应变电站编号2：",line_ex_to_subid[RiskLineID])
    Rlinerho = rhoHis[np.argmin(Totalrho),:]
    Risktime = np.where(Rlinerho > 1)[0]
    if len(Risktime) == 0:#若历史时刻中所有断开线路一直为断开状态，则在历史时刻刚开始就应进行动作部署及遍历
        Risktime = [0]
    Actrho = rhoHis[:,Risktime[0]]#得到部署动作时各重点线路的负载率
    ReConLine = Findex[np.where(Actrho==0)[0]]#得到部署动作时可进行重连尝试的线路编号
    Time2Act = step - Histime + Risktime[0]#得到该采取动作的时间（即故障录波中首次出现线路过载的时刻）
    if ((step-1) > (beginT+Histime)) & (beginT != 0):#只有当变电站母线动作使得系统能够撑过之前的故障时刻方可认为该动作有效——待进一步改进
        Valid = 1
    else:
        Valid = -1
    print("该动作是否有效：",Valid)
    return line_or_to_subid[RiskLineID],line_ex_to_subid[RiskLineID],Time2Act,Valid,step,ReConLine

#进行动作向量向环境可接受动作的转换
def convert_act(agent_action_space,action):#当前只能支持对1个动作的转换
    action_array = action
    # Transfer the l2rpn action array to a dict of grid2op
    action_dict = {}
    obj_id_dict = {}
    real_action = agent_action_space(action_dict)#首先初始时默认不动作


    #line status set
    offset = 0
    set_lines_status_array = action_array[:186]
    set_lines_id_list = list(np.where(set_lines_status_array != 0)[0])
    if len(set_lines_id_list) != 0:
        print("line op!!!")
        for i in range(len(set_lines_id_list)):
            if set_lines_status_array[set_lines_id_list[i]] == -1:
                real_action = agent_action_space.disconnect_powerline(line_id = set_lines_id_list[i])
            else:
                real_action = agent_action_space.reconnect_powerline(line_id=set_lines_id_list[i], bus_ex=int(set_lines_status_array[set_lines_id_list[i]]//10), bus_or=int(set_lines_status_array[set_lines_id_list[i]]%10))
    
    # generator redipatch
    offset += 1438
    generator_redipatch_array = action_array[offset:]
    generator_redipatch_id_list = list(np.where(generator_redipatch_array != 0)[0])
    if len(generator_redipatch_id_list) != 0:
        print("Gen op!!!")
        for i in range(len(generator_redipatch_id_list)):
            print("Gen redispatch ID:",generator_redipatch_id_list[i],"Gen redispatch amount:",generator_redipatch_array[generator_redipatch_id_list[i]])
            real_action = agent_action_space({"redispatch": [(generator_redipatch_id_list[i], generator_redipatch_array[generator_redipatch_id_list[i]])]})
    

    # load reconfigration
    offset += -533
    change_load_array = action_array[offset:offset+99]
    change_load_id_list = list(np.where(change_load_array == 1)[0])
    if len(change_load_id_list) != 0:
        obj_id_dict["loads_id"] = [i for i in change_load_id_list]

    # generator reconfiguration
    offset += 99
    change_gen_array = action_array[offset:offset+62]
    change_gen_id_list = list(np.where(change_gen_array == 1)[0])
    if len(change_load_id_list) != 0:
        obj_id_dict["generators_id"] = [i for i in change_gen_id_list]
    
    # line ox reconfiguration
    offset += 62
    change_lines_or_array = action_array[offset:offset+186]
    change_or_id_list = list(np.where(change_lines_or_array == 1)[0])
    if len(change_or_id_list) != 0:
        obj_id_dict["lines_or_id"] = [i for i in change_or_id_list]

      #print(obj_id_dict["lines_or_id"])
    # line ex
    offset += 186
    change_lines_ex_array = action_array[offset:offset+186]
    change_ex_id_list = list(np.where(change_lines_ex_array == 1)[0])
    if len(change_ex_id_list) != 0:
        obj_id_dict["lines_ex_id"] = [i for i in change_ex_id_list]
    
    if  len(obj_id_dict) != 0:
        action_dict["change_bus"] = obj_id_dict
        real_action = agent_action_space(action_dict)
        
    return real_action
    
    
TlineOR,TlineEX,Time2Act,Valid,Endstep,ReConLine = OPandAnalysis(multimix_env)#初始运行得到第一次的故障信息反馈
ValidAct=[]#初始化有效动作空间
PreActions=[]#初始化历史动作信息列表
#预先构建线路操作及发电机出力操作向量的基本模块
LineV = np.array([11,12,21,22],dtype=float)
gen_max_ramp_up = np.load("GenRamp.npy", allow_pickle=True)
        #num2discrete = 5
discretefactor = np.array([-0.95, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 0.95], dtype =float)#构建发电机出力的变更因子
gen_redisID = np.where(gen_max_ramp_up>0)[0]
DispGenNum = len(gen_redisID)
GenRedispSets = np.zeros((8*DispGenNum,1500))
BusTopo = np.load("BusTopo.npy", allow_pickle=True)#读取系统初始时各变电站拓扑关系以便后续进行邻域搜索
NeiBus = [[]] * SubNum
for i in range(SubNum):
    CuNeiBus = np.where(BusTopo[i] == 1)[0]
    CuNeiBus = CuNeiBus.tolist()
    CuNeiBus.remove(i)
    NeiBus[i] = NeiBus[i] + CuNeiBus
for i in range(len(gen_redisID)):
    GenRedispSets[(i*8):((i+1)*8), 2*LineNum+2*ElementNum+gen_redisID[i]] = gen_max_ramp_up[gen_redisID[i]]*discretefactor#NB only when ramp_up = ramo_down this equation can be applied
while Endstep < MAX_STEPS:
    CurrentBestValidAction = np.zeros((1,1500))#初始化当前动作向量
    CurrentBestValidAction = CurrentBestValidAction[0]
    BestEndTime = Endstep
    #首先尝试断开线路的重连
    if len(ReConLine) > 0:
        for i in range(len(ReConLine)):
            print("线路重连动作编号：",i)
            LineReconActSets = np.zeros((4,1500))
            LineReconActSets[:, i] = LineV
            for j in range(4):
                TlineOR,TlineEX,SearchTime2Act,Valid,Endstep,ReConLine = OPandAnalysis(multimix_env, beginT = Time2Act, targetact= LineReconActSets[j], OPflag = 0, PreActlist = PreActions)
                if Valid == 1:
                    ValidAct.append(LineReconActSets[j])
                    if Endstep > BestEndTime:#当目前动作检验后的系统终止时间长于之前的最优动作效果，更新当前最优动作及其对应的系统终止时间，下同
                        BestEndTime = Endstep
                        CurrentBestValidAction = LineReconActSets[j]
    #之后提取对应变电站动作集并部署
    Sub1Act=np.load("Sub{}.npy".format(TlineOR), allow_pickle=True)
    Sub2Act=np.load("Sub{}.npy".format(TlineEX), allow_pickle=True)
    SubAct = np.append(Sub1Act,Sub2Act,axis=0)#组合生成当前需要遍历的动作空间
    print("待遍历变电站动作个数：",SubAct.shape[0])
    for i in range(SubAct.shape[0]):
        print("遍历动作编号：",i)
        print("介入时间：",Time2Act)
        TlineOR,TlineEX,SearchTime2Act,Valid,Endstep,ReConLine = OPandAnalysis(multimix_env, beginT = Time2Act, targetact= SubAct[i], OPflag = 0, PreActlist = PreActions)
        if Valid == 1:
            print("该动作有效：",Valid)
            ValidAct.append(SubAct[i])
            if Endstep > BestEndTime:
                BestEndTime = Endstep
                CurrentBestValidAction = SubAct[i]
    if sum(CurrentBestValidAction) == 0:#当线路直连的拓扑操作均无效时，遍历2-hop的变电站动作
        ActNeiBus = NeiBus[TlineOR] + NeiBus[TlineEX]
        ActNeiBus.remove(TlineOR)
        ActNeiBus.remove(TlineEX)
        ActNeiBus = list(set(ActNeiBus))#防止有重复出现的母线加大计算量
        for i in range(len(ActNeiBus)):
            CSubAct=np.load("Sub{}.npy".format(ActNeiBus[i]), allow_pickle=True)
            if i == 0:
                NeiSubAct =  CSubAct
            else:
                NeiSubAct = np.append(NeiSubAct,CSubAct,axis=0)#组合生成当前需要遍历的动作空间
        for i in range(NeiSubAct.shape[0]):
            print("遍历动作编号：",i)
            print("介入时间：",Time2Act)
            TlineOR,TlineEX,SearchTime2Act,Valid,Endstep,ReConLine = OPandAnalysis(multimix_env, beginT = Time2Act, targetact= NeiSubAct[i], OPflag = 0, PreActlist = PreActions)
            if Valid == 1:
                print("该动作有效：",Valid)
                ValidAct.append(NeiSubAct[i])
                if Endstep > BestEndTime:
                    BestEndTime = Endstep
                    CurrentBestValidAction = NeiSubAct[i]
    if sum(CurrentBestValidAction) == 0:#当所有邻域拓扑操作均无效时，遍历发电机出力动作
        for i in range(GenRedispSets.shape[0]):
            TlineOR,TlineEX,SearchTime2Act,Valid,Endstep,ReConLine = OPandAnalysis(multimix_env, beginT = Time2Act, targetact= GenRedispSets[i], OPflag = 0, PreActlist = PreActions)
            if Valid == 1:
                print("该动作有效：",Valid)
                ValidAct.append(GenRedispSets[i])
                if Endstep > BestEndTime:
                    BestEndTime = Endstep
                    CurrentBestValidAction = GenRedispSets[i]
    if sum(CurrentBestValidAction) == 0:#当过载线路连接的变电站等多种可能操作动作均无法handle时，跳出遍历过程；否则，则选取当前最优的有效动作为实际部署的动作使系统继续运行
        print("current actions cannot handle the situation!!!")
        break
    else:
        HisActTime = Time2Act
        TlineOR,TlineEX,Time2Act,Valid,Endstep,ReConLine = OPandAnalysis(multimix_env, beginT = Time2Act, targetact= CurrentBestValidAction, PreActlist = PreActions)
        PreActions.append(CurrentBestValidAction)#当存在有效动作时，将该动作以及该动作部署时间记录入历史动作列表中，方便场景的连贯性推演
        PreActions.append(int(HisActTime))
np.save("SavedSubActs",ValidAct)





# while i < NUM_EPISODES:
    # for mix in multimix_env:
        # # Get number of chronics in this mix
        # n_chronics_mix = len(mix.chronics_handler.subpaths)
        # # Adjust iterations to limit to NB_EPISODES
        # if (i + n_chronics_mix) > NUM_EPISODES:
            # n_chronics_mix = NUM_EPISODES - i
        
        # # Iterate over this mix scenarios
        # for c in range(n_chronics_mix):
            # mix.reset()
        
            # # Print some info on current episode
            # mix_name = mix.name
            # chronic_name = mix.chronics_handler.get_name()
            # print ("Episode [{}] - Mix [{}] - Chronic [{}]".format(i, mix_name, chronic_name))

            # done = False
            # obs = mix.current_obs
            # reward = 0.0
            # step = 0
            # while done is False and step < MAX_STEPS:
                # print("step——",step)
                # agent_action = my_agent.act(obs, reward, done)
                # transitions.append(obs.rho)
                # obs, reward, done , info = mix.step(agent_action)
                # transitions.append(reward)
                # transitions.append(obs.rho)
                # transitions.append(done)
                # transitions.append(step)
                # step += 1
            # i += 1
        # if j == 0:
            # break





