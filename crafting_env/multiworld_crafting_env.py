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

class MultiworldCraftingEnv(CraftingBase, MultitaskEnv):
    """
    A MultiWorld API to the crafting environment.
    Goals are pairs of states, the goal is to reproduce the same change in state.
    """

    metadata = {'render.modes': ['rgb', 'ansi']}

    def __init__(self,append_init_state_to_goal=True, **kwargs ):
        super().__init__(**kwargs)
        self.append_init_state_to_goal = append_init_state_to_goal
        goalspace_multipler = 2 if append_init_state_to_goal else 1
        if self.state_obs:
            assert(self.few_obj, "state_obs can only be used with the few_obj mode.")
            self.max_num_per_obj = 3
            self.obs_space = spaces.Box(low=0, high=self.nrow, shape=(self.state_space_size,))
            self.state_space = self.observation_space
            self.goal_space =  spaces.Box(low=-2, high=2, shape=(self.state_space_size*goalspace_multipler,))
        else:
            self.obs_space = spaces.Box(low=0, high=1., shape=((self.nrow+1)*self.res*self.ncol*self.res*3,))
            self.goal_space = spaces.Box(low=0, high=1., shape=((self.nrow+1)*self.res*self.ncol*self.res*3*goalspace_multipler,))
            self.achieved_goal_space = self.goal_space
        self.observation_space = Dict([
            ('observation', self.obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.obs_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
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
        r = self.compute_reward(a, obs)
        info['goal_success'] = r> 0.5
        return obs,r,d,info

    def get_obs(self):
        env_obs = self.get_env_obs().flatten()
        if self.append_init_state_to_goal:
            achieved_goal = np.concatenate([self.init_obs.flatten(), env_obs])
        else:
            achieved_goal = env_obs
        obs = dict(
            observation=env_obs,
            desired_goal=self.goal,
            achieved_goal=achieved_goal,
        )
        return obs
    
    def compute_reward(self, action, obs):
        new_obs = {'achieved_goal': obs['achieved_goal'][np.newaxis,:],
                   'desired_goal': obs['desired_goal'][np.newaxis,:]
                  }
        return self.compute_rewards(action, new_obs)
        
    def compute_rewards(self, actions, obs):
        batch_size = obs['achieved_goal'].shape[0]
        achieved_goals = obs['achieved_goal'].reshape((batch_size, 2,-1))
        achieved_goals = achieved_goals[:,1,:]-achieved_goals[:,0,:]
        desired_goals = obs['desired_goal'].reshape((batch_size, 2,-1))
        desired_goals = desired_goals[:,1,:]-desired_goals[:,0,:]
        if isinstance(achieved_goals,  np.ndarray):
            distances = np.linalg.norm(achieved_goals - desired_goals, axis=1)
        else:
            distances = torch.norm(achieved_goals - desired_goals, dim=1)
        return -1*distances

    def get_goal(self):
        return {
            'desired_goal': self.goal,
        }

    def sample_goals(self, batch_size):
        batch_goals = []
        for i in range(batch_size):
            batch_goals.append(self.sample_goal)
        goals = np.concatenate(batch_goals, axis=0)
        return goals
    
    def sample_goal(self):
        num_tasks = 10
        task_id = np.random.randint(low=0, high=num_tasks, size=3)
        goal_state_dict = self.state
        for task in task_id:
            goal_state_dict = self.generate_goal_state(task, goal_state_dict)
        goal =  self.imagine_obs(goal_state_dict)[np.newaxis, :]
        if self.append_init_state_to_goal:
            curr_obs = self.get_env_obs()[np.newaxis, :]
            goal = np.concatenate([curr_obs, goal_obs], axis=1)
        return goal