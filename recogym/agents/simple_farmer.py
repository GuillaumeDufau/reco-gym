import numpy as np
from numpy.random.mtrand import RandomState


from ..envs.configuration import Configuration
from ..envs.env_garden import env_args as garden_env_1_args

from .abstract import Agent
import random
class SimpleFarmerAgent(Agent):
    def __init__(self, config):
        # Set number of products as an attribute of the Agent.
        super(SimpleFarmerAgent, self).__init__(config)

        # Track number of times each item viewed in Organic session.
        #self.organic_views = np.zeros(self.config.num_products)
        self.num_products = self.config.num_products
        self.plant_state = np.zeros(5)
        self.rng = RandomState(config.random_seed)

    def train(self, observation, action, reward, done):
        """Train method learns from a tuple of data.
            this method can be called for offline or online learning"""


        # Simple Farmer doens't learn
        # Adding organic session to organic view counts.
        #if observation:
        #    for session in observation.sessions():
        #self.organic_views[session['v']] += 1
        if observation.sessions():
            features = ['water_level','fertilizer','maturity','day','forecast']
            #print(observation.sessions())
            last_session = observation.sessions()[-1]
            for i, f in enumerate(features):
                self.plant_state[i] = last_session[f]

    def act(self, observation, reward, done):
        """Act method returns an action based on current observation and past
            history"""

        # Choosing action randomly in proportion with number of views.
        #prob = self.organic_views / sum(self.organic_views)
        #action = choice(self.num_products, p = [1./self.num_products]*self.num_products)
        
        #actions are ['wait','water','harvest','fertilize']
        #features are ['water_level','fertilizer','maturity','day','forecast']

        #harvesting day
        if np.random.uniform() >= 0.99:
            action = self.rng.choice(4)
        else:
            if observation.sessions():
                features = ['water_level','fertilizer','maturity','day','forecast']
                #print(observation.sessions())
                last_session = observation.sessions()[-1]
                for i, f in enumerate(features):
                    self.plant_state[i] = last_session[f]

            #print(f'Plant state is {self.plant_state}')
            if self.plant_state[3] == (garden_env_1_args['harvest_period'] - 1):
                action = 2

            if self.plant_state[2] > 60:
                action = 2
            elif self.plant_state[0] <= 3:
                action = 1
            #no fertilizer
            elif self.plant_state[1] == 0:
                action = 3

            #no water

            else:
                action = 0

        action = random.randrange(self.config.num_products)

        return {
            **super().act(observation, reward, done),
            **{
                'action': action,
                'ps': 1./self.num_products
            }
        }

