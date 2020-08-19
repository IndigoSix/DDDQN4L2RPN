import numpy as np


class Expericence(object):
    def __init__(self, state_dim, embed_dim, seq_len, mem_size):
        self.states = np.zeros((mem_size, state_dim))
        self.actions = np.zeros((mem_size,), dtype=int)
        self.action_embeddings = np.zeros((mem_size, embed_dim))
        self.next_states = np.zeros((mem_size, state_dim))
        self.rewards = np.zeros((mem_size,), dtype=np.float32)
        self.terminals = np.zeros((mem_size,), dtype=int)

        # 指针，cur表明当前存储经验所在的位置，cur_size表明当前经验池大小
        self.cur = 0
        self.cur_size = 0
        self.max_size = mem_size

        # 记录轨迹，每条轨迹长度为seq_len，最多记录seq_max条
        # seq_len_cur表明当前轨迹存储最新的单个经验所在位置，seq_cur表明当前所存储的轨迹为第几条
        # seq_cur_size表明当前轨迹总条数
        self.seq_max = mem_size // seq_len
        self.seq_states = np.zeros((self.seq_max, seq_len + 1, state_dim))
        self.seq_actions = np.zeros((self.seq_max, seq_len + 1), dtype=int)
        self.seq_lengths = np.zeros(self.seq_max, dtype=int)
        self.seq_cur = 0
        self.seq_len_cur = 0
        self.seq_cur_size = 0

    def store(self, state, action, action_embedding, next_state, reward, done):
        self.states[self.cur] = state
        self.actions[self.cur] = action
        self.action_embeddings[self.cur] = action_embedding
        self.next_states[self.cur] = next_state
        self.rewards[self.cur] = reward
        self.terminals[self.cur] = done
        self.cur = (self.cur + 1) % self.max_size
        self.cur_size = self.cur_size + 1 if self.cur_size + 1 < self.max_size else self.max_size

    def store_traj(self, state, action):
        self.seq_states[self.seq_cur][self.seq_len_cur] = state
        self.seq_actions[self.seq_cur][self.seq_len_cur] = action
        self.seq_len_cur += 1

    def finish(self):
        self.seq_lengths[self.seq_cur] = self.seq_len_cur
        self.seq_cur = (self.seq_cur + 1) % self.seq_max
        self.seq_cur_size = self.seq_cur_size + 1 if self.seq_cur_size + 1 < self.seq_max else self.seq_max
        self.seq_len_cur = 0

    def sample(self, n):
        inds = np.random.choice(np.arange(self.cur_size), n)
        return self.states[inds], self.actions[inds], self.action_embeddings[inds], \
               self.next_states[inds], self.rewards[inds], self.terminals[inds]

    def sample_traj(self, n):
        n = min(n, self.seq_cur_size)
        inds = np.random.choice(np.arange(self.seq_cur_size), n)
        return self.seq_states[inds], self.seq_actions[inds], self.seq_lengths[inds]

    def get_size(self):
        return self.cur_size

    def get_traj_size(self):
        return self.seq_cur_size


