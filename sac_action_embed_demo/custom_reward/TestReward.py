import numpy as np

from grid2op.Exceptions import Grid2OpException
from grid2op.Reward.BaseReward import BaseReward
from grid2op.dtypes import dt_float

class TestReward(BaseReward):
    """
    This reward can be used for environments where redispatching is availble. It assigns a cost to redispatching action
    and penalizes with the losses.
    """
    def __init__(self, beta_blackout=50.0, max_step=8000,per_timestep=1):
        BaseReward.__init__(self)
        self.reward_min = None
        self.reward_max = None
        self.steps = dt_float(0.0)
        self.per_timestep = dt_float(per_timestep)
        self.max_blackout = dt_float(0.0)
        self.beta_blackout = dt_float(beta_blackout)
        self.max_step = dt_float(max_step)


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
        
        reward_max_1 = dt_float((self.max_blackout - min_blackout) / least_loads)
        reward_max_2 = env.chronics_handler.max_timestep() * self.per_timestep
        self.reward_max = max(reward_max_1, reward_max_2)
        
        self.steps = dt_float(env.nb_time_step * self.per_timestep)



    def __call__(self,  action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done and self.steps < self.max_step:
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
        elif not is_done and self.steps > self.max_step:
            gen_p, *_ = env.backend.generators_info()
            load_p, *_ = env.backend.loads_info()
            res =  np.divide(sum(load_p), sum(gen_p), dtype=dt_float)
        else:
            res = self.reward_min

        return res
        
    def set_range(self, reward_min, reward_max):
        """
        Setter function for the :attr:`BaseReward.reward_min` and :attr:`BaseReward.reward_max`.
        It is not recommended to override this function
        Parameters
        -------
        reward_min: ``float``
            The minimum reward, see :attr:`BaseReward.reward_min`
        reward_max: ``float``
            The maximum reward, see :attr:`BaseReward.reward_max`
        """
        self.reward_min = reward_min
        self.reward_max = reward_max