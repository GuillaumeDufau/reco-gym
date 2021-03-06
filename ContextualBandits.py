#!/usr/bin/env python
# coding: utf-8

# # Step 1. Setup Env
# !pip install git+https://github.com/criteo-research/reco-gym.git@course

# In[1]:


from recogym.envs.session_plant import OrganicSessions
from recogym import garden_env_1_args

from numpy.random.mtrand import RandomState
from recogym import Configuration, Context_v1, Observation
from recogym.agents import Agent
from recogym.agents import organic_user_count_args, OrganicUserEventCounterAgent
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from recogym.evaluate_agent import verify_agents, plot_verify_agents
from recogym.agents import FeatureProvider, SimpleFarmerAgent
from recogym.agents import RandomAgent, random_args
import math
import gym
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['figure.figsize'] = [6, 3]


# In[2]:


def get_recogym_configuration(num_products, random_seed=42):
    return Configuration({
        **garden_env_1_args,
        'num_products': num_products, 
        'random_seed': random_seed,
        'num_products': num_products,
    })


def get_environement(num_products, random_seed=42, weather = None):
    env = gym.make('garden-gym-v1')
    env.init_gym(get_recogym_configuration(num_products, random_seed=random_seed).__dict__, weather = weather)
    
    return env

## Generating weather
mode = 0.2
weather = np.random.triangular(0.,mode,1.,size=garden_env_1_args['harvest_period'])


# # Step 2. Generate the training data and derive the raw features

# In[3]:

NUM_PLANTS = 1000
NUM_PRODUCTS = garden_env_1_args['num_products']


organic_counter_agent = SimpleFarmerAgent(Configuration({
           **organic_user_count_args,
           **get_recogym_configuration(NUM_PRODUCTS).__dict__,
           'select_randomly': False,
       }))


popularity_policy_logs = get_environement(NUM_PRODUCTS).generate_logs(NUM_PLANTS, organic_counter_agent)

# In[4]:


class ProductCountFeatureProvider(FeatureProvider):
    """This feature provider creates a user state based on viewed product count.
    Namely, the feature vector of shape (n_products, ) contains for each product how many times the
    user has viewed them organically.
    """

    '''This feature provider actually just gets us the features of the plant that day
    '''

    def __init__(self, config):
        super(ProductCountFeatureProvider, self).__init__(config)

        self.feature_data = np.zeros(5).astype(int)


    def observe(self, observation):
        features = ['water_level','fertilizer','maturity','day','forecast']

        try:
            last_session = observation.sessions()[-1]
        
            for i,f in enumerate(features):
                self.feature_data[i] = last_session[f]
        except:
            pass


    def features(self, observation):
        return self.feature_data.copy()

    def reset(self):
        self.feature_data[:] = 0


def build_rectangular_data(logs, feature_provider):
    """Create a rectangular feature set from the logged data.
    For each taken action, we compute the state in which the user was when the action was taken
    """
    user_states, actions, rewards, proba_actions = [], [], [], []
    
    current_plant = None
    for _, row in logs.iterrows():
        if current_plant != row['plant_id']:
            # Use has changed: start a new session and reset user state
            current_plant = row['plant_id']
            sessions = OrganicSessions()
            feature_provider.reset()
        
        context = Context_v1(row['t'], 
                    row['plant_id'],
                    row['maturity'],
                    row['water_level'],
                    row['forecast'],
                    row['day'],
                    row['fertilizer'])


        if row['z'] == 'observation':
            sessions.next(context, row['action'])
            
        else:
            # For each bandit event, generate one observation for the user state, the taken action
            # the obtained reward and the used probabilities
            feature_provider.observe(Observation(context, sessions))
            # just get the previous observations as features

            #print(row)
            user_states += [feature_provider.features(None)] 
            actions += [row['action']]
            rewards += [row['cost']]
            proba_actions += [row['ps']] 
            
            # Start a new organic session
            sessions = OrganicSessions()

    return np.array(user_states), np.array(actions).astype(int), np.array(rewards), np.array(proba_actions)


# In[5]:


# You can now see data that will be provided to our agents based on logistic regressions

feature_provider = ProductCountFeatureProvider(config=get_recogym_configuration(NUM_PRODUCTS))
user_states, actions, rewards, proba_actions = build_rectangular_data(popularity_policy_logs, feature_provider)


# In[6]:
# # Step 3.A. Define and train the bandit likelihood agent

# In order to be able to make the link between the state and the actions, we need to create cross-features 
#that show how good a certain pair of state,action is from the pv of predicting pClick (pReward)

# In[7]:

class LikelihoodAgent(Agent):
    def __init__(self, feature_provider, use_argmax=False, seed=43):
        self.feature_provider = feature_provider
        self.use_argmax = use_argmax
        self.random_state = RandomState(seed)
        self.model = None
        
    @property
    def num_products(self):
        return self.feature_provider.config.num_products
    
    def _create_features(self, user_state, action):
        """Create the features that are used to estimate the expected reward from the user state.
        """
        features = np.zeros(len(user_state) * self.num_products)
        features[action * len(user_state): (action + 1) * len(user_state)] = user_state
        
        return features
    
    def train(self, logs):
        user_states, actions, rewards, proba_actions = build_rectangular_data(logs, self.feature_provider)
        print(user_states)
      
        features = np.vstack([
            self._create_features(user_state, action) 
            for user_state, action in zip(user_states, actions) # should be the enumerate of action
        ])
        self.model = LogisticRegression()
        self.model.fit(features.astype(float), rewards.astype(int))



    
    def _score_products(self, user_state):
        all_action_features = np.array([
            self._create_features(user_state, action) 
            for action in range(self.num_products)
        ])
        temp = self.model.predict_proba(all_action_features)[:, 0]

        return temp
        
    def act(self, observation, reward, done):
        """Act method returns an action based on current observation and past history"""
        self.feature_provider.observe(observation)
        user_state = self.feature_provider.features(observation)
        prob = self._score_products(user_state)

        try:
            action = self.random_state.choice(self.num_products, p=prob/sum(prob))

            ps = prob[action]
            all_ps = prob.copy()
        except:
            action = np.argmax(prob)
            ps = 1.0
            all_ps = np.zeros(self.num_products)
            all_ps[action] = 1.0
        # ##epsilon greedy is working better, change it after tests

        return {
            **super().act(observation, reward, done),
            **{
                'action': action,
                'ps': ps,
                'ps-a': all_ps,
            }
        }

    def reset(self):
        self.feature_provider.reset()  

# Have a look at the feature vector used by the Likelihood agent
picked_sample = 10

count_product_views_feature_provider = ProductCountFeatureProvider(get_recogym_configuration(NUM_PRODUCTS))
likelihood_logreg = LikelihoodAgent(count_product_views_feature_provider)

likelihood_logreg = LikelihoodAgent(count_product_views_feature_provider)
likelihood_logreg.train(popularity_policy_logs)

# In[10]:plt.show()

# # Step 3.B. Define and train the Contextual Bandit agent
class PolicyAgent(LikelihoodAgent):
    def __init__(self, feature_provider, use_argmax=False, seed=43):
        LikelihoodAgent.__init__(self, feature_provider, use_argmax=use_argmax, seed=seed)
    
    def _create_features(self, user_state, action):
        """Create the features that are used to estimate the expected reward from the user state.
        """
        return user_state
    
    def train(self, reco_log):
        user_states, actions, rewards, proba_actions = build_rectangular_data(reco_log, feature_provider)
        
        features = np.vstack([
            self._create_features(user_state, action) 
            for user_state, action in zip(user_states, actions)
        ])        
        labels = actions
        weights = rewards / proba_actions
        
        self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=5000)        
        self.model.fit(features[weights != 0], labels[weights != 0], weights[weights != 0])
    
    def _score_products(self, user_state):
        return self.model.predict_proba(self._create_features(user_state, None).reshape(1, -1))[0, :]

# In[12]:


# In[13]:
policy_logreg = PolicyAgent(count_product_views_feature_provider)
policy_logreg.train(popularity_policy_logs)


result = verify_agents(get_environement(NUM_PRODUCTS,weather = weather),NUM_PLANTS,{'simple farmer': organic_counter_agent})

result = verify_agents(get_environement(NUM_PRODUCTS, weather = weather), NUM_PLANTS, {
    'likelihood logreg': likelihood_logreg, 
    # 'policy logreg': policy_logreg,
    
})
