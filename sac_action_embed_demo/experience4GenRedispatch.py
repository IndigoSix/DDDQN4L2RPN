import numpy as np


class Expericence(object):
    def __init__(self, state_dim, action_dim, mem_size):
        self.states = np.zeros((mem_size, state_dim))
        self.action = np.zeros((mem_size, action_dim))
        self.next_states = np.zeros((mem_size, state_dim))
        self.rewards = np.zeros((mem_size,), dtype=np.float32)
        self.terminals = np.zeros((mem_size,), dtype=int)

        # 指针，cur表明当前存储经验所在的位置，cur_size表明当前经验池大小
        self.cur = 0
        self.cur_size = 0
        self.max_size = mem_size

    def store(self, state, action, next_state, reward, done):
        self.states[self.cur] = state
        self.action[self.cur] = action
        self.next_states[self.cur] = next_state
        self.rewards[self.cur] = reward
        self.terminals[self.cur] = done
        self.cur = (self.cur + 1) % self.max_size
        self.cur_size = self.cur_size + 1 if self.cur_size + 1 < self.max_size else self.max_size


    def sample(self, n):
        inds = np.random.choice(np.arange(self.cur_size), n)
        return self.states[inds], self.action[inds], \
               self.next_states[inds], self.rewards[inds], self.terminals[inds]

    def get_size(self):
        return self.cur_size