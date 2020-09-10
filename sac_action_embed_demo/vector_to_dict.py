#首先make env，确定action space
self.agent_action_space = env.action_space

#找到vector的控制维度，一共是29个sub上的32个电机
self.gen_redis =  np.where(self.agent_action_space.gen_redispatchable)[0]
self.plant_id = [0]
n = 0
for i in range(1,len(self.agent_action_space.gen_to_subid[self.gen_redis])):
  if self.agent_action_space.gen_to_subid[self.gen_redis][i] != self.agent_action_space.gen_to_subid[self.gen_redis][i-1]:
    n+=1
  self.plant_id.append(n)
self.n_action = len(set(self.plant_id))

#将动作vector转化为dict
def select_act(self, action_vct, obs_dict):
  redis_list = []
  self.agent_action_space.gen_pmax[19] = 56
  self.agent_action_space.gen_pmax[51] = 90         
  self.agent_action_space.gen_pmax[53] = 56   
  self.agent_action_space.gen_pmax[56] = 150         
  self.agent_action_space.gen_pmax[6] = 90   
  self.agent_action_space.gen_pmax[42] = 100  
  self.agent_action_space.gen_pmax[45] = 100   
  self.agent_action_space.gen_pmax[10] = 120        
  self.agent_action_space.gen_pmax[37] = 200  
  self.agent_action_space.gen_pmax[38] = 100      
  self.agent_action_space.gen_pmax[10] = 90             
  self.agent_action_space.gen_pmin = [20]*len(self.agent_action_space.gen_pmin)

  for x, i in enumerate(self.plant_id):
    after_redis = obs_dict["prods"]["p"][self.gen_redis[x]]+action_vct[i]*self.agent_action_space.gen_max_ramp_up[self.gen_redis[x]]      
    if (self.agent_action_space.gen_pmin[self.gen_redis[x]] < after_redis and action_vct[i] < 0) or (self.agent_action_space.gen_pmax[self.gen_redis[x]]> after_redis and action_vct[i] > 0):
      continue
    else:
      redis_list.append((self.gen_redis[x], action_vct[i]*self.agent_action_space.gen_max_ramp_up[self.gen_redis[x]]))
  action = self.agent_action_space({'redispatch': redis_list})
  return action
 
