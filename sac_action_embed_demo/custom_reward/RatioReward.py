import numpy as np

from grid2op.Exceptions import Grid2OpException
from grid2op.Reward.BaseReward import BaseReward
from grid2op.dtypes import dt_float

class RatioReward(BaseReward):
    """
    This reward can be used for environments where redispatching is availble. It assigns a cost to redispatching action
    and penalizes with the losses.
    """
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = dt_float(0.0)
        self.reward_max = dt_float(1.0)


    def initialize(self, env):
        pass


    def __call__(self,  action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not is_done and not has_error:
            gen_p, *_ = env.backend.generators_info()
            load_p, *_ = env.backend.loads_info()
            res =  np.divide(sum(load_p), sum(gen_p), dtype=dt_float)
        else:
            # no more data to consider, no powerflow has been run, reward is what it is
            res = self.reward_min
        return res