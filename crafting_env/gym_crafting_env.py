import numpy as np

from gym import Env, spaces
from gym.utils import seeding
import sys
from six import StringIO, b
import copy
from gym import utils
from gym.envs.toy_text import discrete
from scipy.misc import imresize

from gridworld.envs.crafting_env.crafting_base import CraftingBase

class CraftingGymEnv(CraftingBase, Env):
    """
    A Gym API to the crafting environment.
    """

    metadata = {'render.modes': ['rgb', 'ansi']}

    def __init__(self, **kwargs ):
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        """ 
        init_from_state: If a dictionary state is passed here, the environment
        will reset to that state. Otherwise, the state is randomly initialized.
        """
        self._reset(**kwargs)
        obs = self.get_env_obs()
        self.init_obs = obs.copy()
        obs = self.get_obs()
        return obs
        
    def step(self, a):
        r,d, info = self._step(a)
        obs = self.get_obs()
        return obs,r,d,info
    
    
        
