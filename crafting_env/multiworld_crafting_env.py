import numpy as np
from gym.spaces import Dict, Box
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from gym import  spaces
import sys
from six import StringIO, b
import copy
from gym import utils
from gym.envs.toy_text import discrete

from crafting_env.crafting_base import CraftingBase

class CraftingMultiworldEnv(CraftingBase, MultitaskEnv):
    """
    A MultiWorld API to the crafting environment.
    Goals are pairs of states, the goal is to reproduce the same change in state.
    """

    metadata = {'render.modes': ['rgb', 'ansi']}

    def __init__(self, **kwargs ):
        super().__init__(**kwargs)
        if self.state_obs:
            assert(self.few_obj)
            self.max_num_per_obj = 3
            self.state_space_size = len(OBJECTS)*2*self.max_num_per_obj+2+1+1
            self.obs_space = spaces.Box(low=0, high=self.nS, shape=(self.state_space_size,))
            self.state_space = self.observation_space
            self.goal_space =  spaces.Box(low=0, high=self.nS, shape=(self.state_space_size*2,))
        else:
            self.obs_space = spaces.Box(low=0, high=1., shape=((self.nrow+1)*self.res*self.ncol*self.res*3,))
            self.goal_space = spaces.Box(low=0, high=1., shape=((self.nrow+1)*self.res*self.ncol*self.res*3*2,))
            self.achieved_goal_space = self.goal_space
        self.observation_space = Dict([
            ('observation', self.obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.achieved_goal_space),
            ('state_observation', self.obs_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.achieved_goal_space),
        ])

    def reset(self, **kwargs):
        """ 
        init_from_state: If a dictionary state is passed here, the environment
        will reset to that state. Otherwise, the state is randomly initialized.
        """
        self._reset(**kwargs)
        self.goal = self.sample_goal()
        obs = self.get_env_obs()
        self.init_obs = obs.copy()
        obs = self.get_obs()
        return obs
        
    def step(self, a):
        r, d, info = self._step(a)
        obs = self.get_obs()
        return obs,r,d,info

    def get_obs(self):
        env_obs = self.get_env_obs().flatten()
        achieved_goal = np.concatenate([self.init_obs.flatten(), env_obs])
        obs = dict(
            observation=env_obs,
            desired_goal=self.goal,
            achieved_goal=achieved_goal,
        )
        return obs
        
    def compute_rewards(self, obs):
        batch_size = obs['achieved_goal'].shape[0]
        achieved_goals = obs['achieved_goal'].reshape((batch_size, 2,-1))
        achieved_goals = achieved_goals[:,1,:]-achieved_goal[:,0,:]
        desired_goals = obs['desired_goal'].reshape((batch_size, 2,-1))
        desired_goals = desired_goals[:,1,:]-desired_goals[:,0,:]
        if isinstance(achieved_goals,  np.ndarray):
            distances = np.linalg.norm(achieved_goals - desired_goals, axis=1)
        else:
            distances = torch.norm(achieved_goals - desired_goals, dim=1)

    def get_goal(self):
        return {
            'desired_goal': self.goal,
        }

    def sample_goals(self, batch_size):
        goals = np.random.uniform(
            self.goal_space.low,
            self.goal_space.high,
            size=(batch_size, self.goal_space.low.size),
        )
        return goals
    
    def sample_goal(self):
        goal = self.sample_goals(1)[0]
        return goal