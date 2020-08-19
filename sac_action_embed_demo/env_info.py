import numpy as np

class Env_Info(object):
    def __init__(self, env):
        self.env = env
        self.Line_or = self.env.action_space.line_or_to_subid
        self.Line_ex = self.env.action_space.line_ex_to_subid
        self.LineNum = len(self.Line_or)
        self.GenNum = len(self.env.action_space.gen_to_subid)
        self.LoadNum = len(self.env.action_space.load_to_subid)
        self.SubNum = len(self.env.action_space.sub_info)
        self.ElementNum = np.sum(self.env.action_space.sub_info)
        self.GenRampUp = self.env.action_space.gen_max_ramp_up
