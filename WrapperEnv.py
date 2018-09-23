from make_env import make_env
import numpy as np


def one_hot_action(n):
    oh = np.zeros(5)
    oh[n] = 1
    return oh

# TODO len_hist may be changed to 3 later for easier training.


class WrapperEnv(object):
    def __init__(self, scenario_name, max_steps=100, benchmark=False, pretrain=False):
        self.env = make_env(scenario_name, benchmark)
        self.max_steps = max_steps
        self.steps = 0

        self.n_agents = 2 # fixed
        self.histories = None
        print(hasattr(self.env.world, "len_hist"))
        self.len_hist = self.env.world.len_hist if hasattr(self.env.world, "len_hist") else 2
        self.dim_belief = (self.n_agents - 1) * self.max_steps * self.len_hist
        print(self.len_hist)
        self.pretrain = pretrain

        if self.len_hist == 2:
            self.dim_belief += 6

        self.dim_obs = 11 + self.dim_belief

        print("Initialized")

    def step(self, action_n): # action in the form like [1 , 4] for two agents.
        action = [one_hot_action(x) for x in action_n]
        obs, rewards, _, info = self.env.step(action)

        if not self.pretrain:
            self.set_histories(info)  # update the history before update the step

        obs = [np.append(obs[i], self.histories[1-i]) for i in range(self.n_agents)] # add the other agent's history

        self.steps += 1
        done = self.get_done()
        return obs, rewards, done, info

    def set_histories(self, info):
        start = self.steps * self.len_hist
        for i in range(self.n_agents):
            self.histories[i][start:start+self.len_hist] = info['n'][i]


    def reset(self):
        obs = self.env.reset()
        self.steps = 0
        self.histories = np.zeros((self.n_agents, self.max_steps*self.len_hist))
        obs = [np.append(obs[i], self.histories[1-i]) for i in range(self.n_agents)]
        return obs

    def get_done(self):
        return self.steps >= self.max_steps

    def render(self):
        self.env.render()


    @property
    def action_space(self):
        return self.env.action_space
