import numpy as np
import random

class SumTree:
    """
    A binary tree data structure where the parent’s value is the sum of its children.
    Used here to sample transitions proportional to their priority.
    """
    def __init__(self, capacity):
        self.capacity = capacity  # maximum number of leaf nodes (transitions)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer with importance-sampling weights.
    Sample transitions with probability proportional to their priority.
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 1e-6  # small amount to avoid zero priority
        self.size = 0        # current number of transitions in buffer

    def _beta_by_frame(self):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))

    def push(self, transition):
        # Use max priority for new transitions so they are sampled at least once
        max_p = np.max(self.tree.tree[-self.capacity:])
        if max_p == 0:
            max_p = 1.0
        self.tree.add(max_p, transition)
        # update current size
        self.size = min(self.size + 1, self.capacity)
    def sample(self, batch_size):
        batch, idxs, priorities = [], [], []

        total = self.tree.total()
        if total <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")

        # each segment of the cumulative sum
        segment = total / batch_size

        # anneal beta
        self.beta = self._beta_by_frame()
        self.frame += 1

        for i in range(batch_size):
            # pick s in [i*segment, (i+1)*segment)
            a = segment * i
            s = random.random() * segment + a

            idx, p, data = self.tree.get(s)
            # if you ever hit a default‐inited slot (data is int 0), re‐draw
            while not isinstance(data, tuple):
                # uniform over [0, total)
                s = random.random() * total
                idx, p, data = self.tree.get(s)

            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        # compute normalized IS weights
        sampling_prob = np.array(priorities, dtype=np.float32) / total
        sampling_prob = np.maximum(sampling_prob, 1e-8)       # never zero
        is_weight = (self.capacity * sampling_prob) ** (-self.beta)

        # safe normalization
        max_w = np.nanmax(is_weight)
        if not np.isfinite(max_w) or max_w <= 0:
            is_weight = np.ones_like(is_weight, dtype=np.float32)
        else:
            is_weight = is_weight / max_w

        return batch, idxs, is_weight



    def update_priorities(self, idxs, td_errors):
        for idx, error in zip(idxs, td_errors):
            p = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        """Return current number of transitions stored"""
        return self.size
