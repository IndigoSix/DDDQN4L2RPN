import numpy as np

from grid2op.Exceptions import Grid2OpException
from grid2op.Reward.BaseReward import BaseReward
from grid2op.dtypes import dt_float

class BlackoutReward(BaseReward):
    """
    This reward can be used for environments where redispatching is availble. It assigns a cost to redispatching action
    and penalizes with the losses.
    """
    def __init__(self, beta_blackout=50.0):
        BaseReward.__init__(self)
        self.reward_min = None
        self.reward_max = None
        self.max_blackout = dt_float(0.0)
        self.beta_blackout = dt_float(beta_blackout)


    def initialize(self, env):
        if not env.redispatching_unit_commitment_availble:
            raise Grid2OpException("Impossible to use the RedispReward reward with an environment without generators"
                                   "cost. Please make sure env.redispatching_unit_commitment_availble is available.")
        worst_marginal_cost = np.max(env.gen_cost_per_MW)
        worst_load = dt_float(np.sum(env.gen_pmax))
        self.max_blackout = self.beta_blackout * worst_load * worst_marginal_cost
        self.reward_min = dt_float(-10.0)

        least_loads = dt_float(worst_load * 0.5)  # half the capacity of the grid
        least_redisp = dt_float(0.0)  # lower_bound is 0
        base_marginal_cost = np.min(env.gen_cost_per_MW[env.gen_cost_per_MW > 0.])
        min_blackout = self.beta_blackout * least_redisp * base_marginal_cost
        self.reward_max = dt_float((self.max_blackout - min_blackout) / least_loads)


    def __call__(self,  action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            res = self.reward_min
        else:
            # compute the losses
            #gen_p, *_ = env.backend.generators_info()
            load_p, *_ = env.backend.loads_info()
            #ratio = np.sum(load_p) / np.sum(gen_p)  

            # compute the marginal cost
            marginal_cost = np.max(env.gen_cost_per_MW[env.gen_activeprod_t > 0.])

            # blackout
            blackout_cost = self.beta_blackout * np.sum(load_p) * marginal_cost

            # compute reward
            reward = self.max_blackout - blackout_cost

            # divide it by load, to be less sensitive to load variation
            res = dt_float(reward / np.sum(load_p))

        return res