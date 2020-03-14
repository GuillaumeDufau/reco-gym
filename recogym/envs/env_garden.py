from abc import ABC

import gym
import numpy as np
import pandas as pd
from gym.spaces import Discrete
from numpy.random.mtrand import RandomState
from scipy.special import expit as sigmoid
from tqdm import trange

from .configuration import Configuration
from .context_plant import Context_v1
from .features.time import DefaultTimeGenerator
from .observation import Observation
from .session_plant import OrganicSessions
from ..agents import Agent

# Arguments shared between all environments.

env_args = {
    'num_products': 10,
    'num_users': 100,
    'random_seed': np.random.randint(2 ** 31 - 1),
    # Markov State Transition Probabilities.
    'prob_leave_bandit': 0.01,
    'prob_leave_organic': 0.01,
    'prob_bandit_to_organic': 0.05,
    'prob_organic_to_bandit': 0.25,
    'normalize_beta': False,
    'with_ps_all': False
}


# Static function for squashing values between 0 and 1.
def f(mat, offset=5):
    """Monotonic increasing function as described in toy.pdf."""
    return sigmoid(mat - offset)


# Magic numbers for Markov states.
organic = 0
bandit = 1
stop = 2


class AbstractEnv(gym.Env, ABC):
    """
    This is the structure of one plant
    """

    def __init__(self):
        gym.Env.__init__(self)
        ABC.__init__(self)

        self.first_step = True
        self.config = None
        self.state = None
        self.current_plant_id = None
        self.current_time = None
        self.plant_type = 'Tomato'
        self.maturity = 0
        self.harvested = False
        self.water_level = 6
        self.empty_sessions = OrganicSessions()
        self.action_dict = ['water','harvest']
    

    def reset_random_seed(self, epoch=0):
        # Initialize Random State.
        assert (self.config.random_seed is not None)
        self.rng = RandomState(self.config.random_seed + epoch)

    def init_gym(self, args):   

        self.config = Configuration(args)

        # Defining Action Space. Gym spaces. Potential actions of the gardener -> randomness underlying those choses
        # due to the weather conditions and the actual state of the plant
        self.action_space = Discrete(3) #num_products

        #############
        if 'time_generator' not in args:
            self.time_generator = DefaultTimeGenerator(self.config)
        else:
            self.time_generator = self.config.time_generator
        ##############

        # Setting random seed for the first time.
        self.reset_random_seed()

        if 'agent' not in args:
            self.agent = None
        else:
            self.agent = self.config.agent

        # Setting any static parameters such as transition probabilities.
        self.set_static_params()

        # Set random seed for second time, ensures multiple epochs possible.
        self.reset_random_seed()

    def reset(self, plant_id=0):
        # Current state.
        self.first_step = True
        self.state = organic  # Manually set first state as Organic (to get observations-> equivalent of selecting arm None for the gardener)

        # set day time to zero
        self.time_generator.reset()
        # reset the policy
        if self.agent:
            self.agent.reset()

        self.current_time = self.time_generator.new_time()
        self.current_plant_id = plant_id

        # Record number of times each product seen for static policy calculation.
        self.organic_views = np.zeros(self.config.num_products)

    def generate_organic_sessions(self):
        
        # Initialize session.
        session = OrganicSessions()

        while self.state == organic:
            # Add next product view.
            self.update_product_view()
            session.next(
                Context_v1(self.current_time, 
                self.current_plant_id,
                self.maturity,
                self.water_level),
                self.product_view
            )
            # Update markov state.
            self.update_state()
        return session

    def step(self, action_id):
        """

        Parameters
        ----------
        action_id : int between 1 and num_products indicating which
                 product recommended (aka which ad shown)

        Returns
        -------
        observation, reward, done, info : tuple
            observation (tuple) :
                a tuple of values (is_organic, product_view)
                is_organic - True  if Markov state is `organic`,
                             False if Markov state `bandit` or `stop`.
                product_view - if Markov state is `organic` then it is an int
                               between 1 and P where P is the number of
                               products otherwise it is None.
            reward (float):
                if the previous state was
                    `bandit` - then reward is 1 if the user clicked on the ad
                               you recommended otherwise 0
                    `organic` - then reward is None
            done (bool) :
                whether it's time to reset the environment again.
                An episode is over at the end of a user's timeline (all of
                their organic and bandit sessions)
            info (dict) :
                 this is unused, it's always an empty dict
        """

        # No information to return.
        info = {}

        if self.first_step:
            assert (action_id is None)
            self.first_step = False
            sessions = self.generate_organic_sessions()
            return (
                Observation(
                    Context_v1(self.current_time, 
                    self.current_plant_id,
                    self.maturity,
                    self.water_level),
                    sessions
                ),
                None,
                self.state == stop,
                info
            )

        assert (action_id is not None)

        self.update_state()

        ## Need to update environment
        self.update_env(action_id)

        
        # Calculate reward from action.
        reward = self.draw_click(action_id)


        #if self.action_dict[action_id] == 'observe':
        #    self.state = organic  # After a click, Organic Events always follow.

        # Markov state dependent logic.
        if self.state == organic:
            sessions = self.generate_organic_sessions()
        else:
            sessions = self.empty_sessions

        return (
            Observation(
                Context_v1(self.current_time, 
                self.current_plant_id,
                self.maturity,
                self.water_level),
                sessions
            ),
            reward,
            (self.state == stop) | (self.harvested),
            info
        )

    # New stuff
    def update_env(self,action_id):
        actions = ['water','harvest']
        a = actions[action_id]
        if self.state == organic:
            pass
        elif a == 'water':
            self.water_level += 3
        elif a == 'harvest':
            self.harvested = True

        self.maturity += 0.1*self.water_level
        self.water_level -= 1



'''
    def step_offline(self, observation, reward, done):
        """Call step function wih the policy implemented by a particular Agent."""

        if self.first_step:
            action = None
        else:
            assert (hasattr(self, 'agent'))
            assert (observation is not None)
            if self.agent:
                action = self.agent.act(observation, reward, done)
            else:
                # Select a Product randomly.
                action = {
                    't': observation.context().time(),
                    'weather': observation.context().weather(),
                    'u': observation.context().user(),
                    'a': np.int16(self.rng.choice(self.config.num_products)),
                    'ps': 1.0 / self.config.num_products,
                    'ps-a': (
                        np.ones(self.config.num_products) / self.config.num_products
                        if self.config.with_ps_all else
                        ()
                    ),
                }

        if done:
            return (
                action,
                Observation(
                    DefaultContext(self.current_time, self.current_user_id),
                    self.empty_sessions
                ),
                0,
                done,
                None
            )
        else:
            observation, reward, done, info = self.step(
                action['a'] if action is not None else None
            )

            return action, observation, reward, done, info

    def generate_logs(
            self,
            num_offline_users: int,
            agent: Agent = None,
            num_organic_offline_users: int = 0
    ):
        """
        Produce logs of applying an Agent in the Environment for the specified amount of Users.
        If the Agent is not provided, then the default Agent is used that randomly selects an Action.
        """

        if agent:
            old_agent = self.agent
            self.agent = agent

        data = {
            't': [],
            'plant_id': [],
            'maturity': [],
            'water_level': [],
            'z': [],
            'action': [],
            'cost': [],
            'ps': [],
            'ps-a': [],
        }

        def _store_organic(observation):
            assert (observation is not None)
            assert (observation.sessions() is not None)
            for session in observation.sessions():
                data['t'].append(session['t'])
                data['plant_id'].append(session['plant_id'])
                data['maturity'].append(session['maturity'])
                data['water_level'].append(session['water_level'])
                data['z'].append('observation')
                data['action'].append(None)
                data['cost'].append(None)
                data['ps'].append(None)
                data['ps-a'].append(None)

        def _store_bandit(action, reward):
            if action:
                assert (reward is not None)
                data['t'].append(action['t'])
                data['plant_id'].append(action['plant_id'])
                data['maturity'].append(action['maturity'])
                data['water_level'].append(action['water_level'])
                data['z'].append('working_bandit')
                data['action'].append(action['action'])
                data['cost'].append(reward)
                data['ps'].append(action['ps'])
                data['ps-a'].append(action['ps-a'] if 'ps-a' in action else ())

        unique_user_id = 0
        for _ in trange(num_organic_offline_users, desc='Organic Users'):
            self.reset(unique_user_id)
            unique_user_id += 1
            observation, _, _, _ = self.step(None)
            _store_organic(observation)

        for _ in trange(num_offline_users, desc='Users'):
            self.reset(unique_user_id)
            unique_user_id += 1
            observation, reward, done, _ = self.step(None)

            while not done:
                _store_organic(observation)
                action, observation, reward, done, _ = self.step_offline(
                    observation, reward, done
                )
                _store_bandit(action, reward)

            _store_organic(observation)
            action, _, reward, done, _ = self.step_offline(
                observation, reward, done
            )
            assert done, 'Done must not be changed!'
            _store_bandit(action, reward)

        data['t'] = np.array(data['t'], dtype=np.float32)
        data['u'] = pd.array(data['u'], dtype=pd.UInt16Dtype())
        data['v'] = pd.array(data['v'], dtype=pd.UInt16Dtype())
        data['a'] = pd.array(data['a'], dtype=pd.UInt16Dtype())
        data['c'] = np.array(data['c'], dtype=np.float32)

        if agent:
            self.agent = old_agent

        return pd.DataFrame().from_dict(data)'''
