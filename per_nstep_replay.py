# per_nstep_replay.py
import numpy as np
from collections import deque
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

class NStepReplayBuffer(ReplayBuffer):
    """
    N-step returns for SB3 DQN without touching the algorithm code.
    - Accumulates n rewards and early-terminates on done.
    - Stores (obs_t, a_t, R_t^n, obs_{t+n}, done_{t+n}) in the base buffer.
    """
    def __init__(self, *args, n_step=3, gamma=0.99, **kwargs):
        super().__init__(*args, **kwargs)       # SB3 base init (no gamma arg)
        self.n_step = n_step
        self.gamma = gamma
        # one deque per env to accumulate raw steps
        self._qs = [deque(maxlen=n_step) for _ in range(self.n_envs)]

    def _push_nstep(self, env_i, tr):
        """
        tr = (obs, action, reward, next_obs, done)
        Once we have n items, compute the aggregate tuple and push to base buffer.
        """
        q = self._qs[env_i]
        q.append(tr)
        if len(q) < self.n_step:
            return

        R, disc = 0.0, 1.0
        next_obs_n, done_n = q[-1][3], q[-1][4]
        for (o, a, r, no, d) in q:
            R += disc * float(r)
            disc *= self.gamma
            if d:
                done_n = True
                next_obs_n = no
                break
        obs0, action0 = q[0][0], q[0][1]
        # push into the *base* ring buffer
        super().add(obs0, next_obs_n, action0, R, done_n, infos=[{} for _ in range(self.n_envs)])

    def add(self, obs, next_obs, action, reward, done, infos):
        """
        SB3 vec-format add: arrays are (n_envs, ...).
        We intercept and do n-step aggregation before delegating to base .add().
        """
        for i in range(self.n_envs):
            self._push_nstep(i, (obs[i], action[i], reward[i], next_obs[i], done[i]))

    # sample(): just use the base implementation (uniform sampling, no IS weights)
    # def sample(...): return super().sample(...)
