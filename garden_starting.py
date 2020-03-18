#!/usr/bin/env python
# coding: utf-8

# ## What is RecoGym?
# 
# RecoGym is a Python [OpenAI Gym](https://gym.openai.com/) environment for testing recommendation algorithms.  It allows for the testing of both offline and reinforcement-learning based agents.  It provides a way to test algorithms in a toy environment quickly.
# 
# In this notebook, we will code a simple recommendation agent that suggests an item in proportion to how many times it has been viewed.  We hope to inspire you to create your own agents and test them against our baseline models.
# 
# In order to make the most out of RecoGym, we suggest you have some experience coding in Python, some background knowledge in recommender systems, and familiarity with the reinforcement learning setup.  Also, be sure to check out the python-based requirements in the README if something below errors.

# ## Reinforcement Learning Setup
# 
# RecoGym follows the usual reinforcement learning setup.  This means there are interactions between the environment (the user's behaviour) and the agent (our recommendation algorithm).  The agent receives a reward if the user clicks on the recommendation.

# <img src="images/rl-setup.png" alt="Drawing" style="width: 600px;"/>

# ## Organic and Bandit
# 
# Even though our focus is biased towards online advertising, we tried to make RecoGym universal to all types of recommendation.  Hence, we introduce the domain-agnostic terms Organic and Bandit sessions.  An Organic session is an observation of items the user interacts with.  For example, it could be views of products on an e-commerce website, listens to songs while streaming music, or readings of articles on an online newspaper.  A Bandit session is one where we have an opportunity to recommend the user an item and observe their behaviour.  We receive a reward if they click.
# 
# <img src="images/organic-bandit.png" alt="Drawing" style="width: 450px;"/>

# ## Offline and Online Learning
# 
# This project was born out of a desire to improve Criteo's recommendation system by exploring reinforcement learning algorithms. We quickly realised that we couldn't just blindly apply RL algorithms in a production system out of the box. The learning period would be too costly. Instead, we need to leverage the vast amounts of offline training examples we already to make the algorithm perform as good as the current system before releasing into the online production environment.
# 
# Thus, RecoGym follows a similar flow. An agent is first given access to many offline training examples produced from a fixed policy. Then, they have access to the online system where they choose the actions.

# <img src="images/two-steps.png" alt="Drawing" style="width: 450px;"/>

# ## Let's see some code - Interacting with the environment 
# 
# 
# The code snippet below shows how to initialise the environment and step through in an 'offline' manner (Here offline means that the environment is generating some recommendations for us).  We print out the results from the environment at each step.

# In[1]:


import gym, recogym

# env_0_args is a dictionary of default parameters (i.e. number of products)
from recogym import garden_env_1_args, Configuration

# You can overwrite environment arguments here:
garden_env_1_args['random_seed'] = 42

# Initialize the gym for the first time by calling .make() and .init_gym()
env = gym.make('garden-gym-v1')
env.init_gym(garden_env_1_args)

# .reset() env before each episode (one episode per user).
env.reset()
done = False


# In[2]:


# Create list of hard coded actions.
actions = [None] + [0,0,0,0,0,0,1,0,0,0]

# Reset env and set done to False.
env.reset()
done = False

# Counting how many steps.
i = 0

while not done and (i < len(actions)):
    action = actions[i]
    observation, reward, done, info = env.step(action)
    print(f"Step: {i} - Action: {action} - Observation: {observation.sessions()} - Reward: {reward}")
    print('')
    i += 1



# You'll notice that the offline and online APIs are nearly identical.  The only difference is that one calls either env.step_offline() or env.step(action).

# ## Creating our first agent
# 
# Now that we see have seen how the offline and online versions of the environment work, it is time to code our first recommendation agent!  Technically, an agent can be anything that produces actions for the environment to use.  However, we will show you the object-oriented way we like to create agents.
# 
# Below is the code for a very simple agent - the popularity based agent. The popularity based agent records merely how many times a user sees each product organically, then when required to make a recommendation, the agent chooses a product randomly in proportion with a number of times the user has viewed it.

# In[3]:


import numpy as np
from numpy.random import choice
from recogym.agents import Agent

# Define an Agent class.
class PopularityAgent(Agent):
    def __init__(self, config):
        # Set number of products as an attribute of the Agent.
        super(PopularityAgent, self).__init__(config)

        # Track number of times each item viewed in Organic session.
        self.organic_views = np.zeros(self.config.num_products)
        self.num_products = self.config.num_products

    def train(self, observation, action, reward, done):
        """Train method learns from a tuple of data.
            this method can be called for offline or online learning"""

        # Adding organic session to organic view counts.
        #if observation:
        #    for session in observation.sessions():
                #self.organic_views[session['v']] += 1

    def act(self, observation, reward, done):
        """Act method returns an action based on current observation and past
            history"""

        # Choosing action randomly in proportion with number of views.
        #prob = self.organic_views / sum(self.organic_views)
        action = choice(self.num_products, p = [1./self.num_products]*self.num_products)

        return {
            **super().act(observation, reward, done),
            **{
                'action': action,
                'ps': 1./self.num_products
            }
        }




# The `PopularityAgent` class above demonstrates our preferred way to create agents for RecoGym. Notice how we have both a `train` and `act` method present. The `train` method is designed to take in training data from the environments `step_offline` method and thus has nothing to return, while the `act` method must return an action to pass back into the environment. 
# 
# The code below highlights how one would use this agent for first offline training and then using the learned knowledge to make recommendations online.

# In[4]:


# Instantiate instance of PopularityAgent class.
num_products = garden_env_1_args['num_products']
agent = PopularityAgent(Configuration({
    **garden_env_1_args,
    'num_products': num_products,
}))

# Resets random seed back to 42, or whatever we set it to in env_0_args.
env.reset_random_seed()

# Train on 1000 users offline.
num_offline_users = 1000
"""
for _ in range(num_offline_users):

    # Reset env and set done to False.
    env.reset()
    done = False

    observation, reward, done = None, 0, False
    while not done:
        old_observation = observation
        action, observation, reward, done, info = env.step_offline(observation, reward, done)
        agent.train(old_observation, action, reward, done)
"""
# Train on 100 users online and track click through rate.
num_online_users = 100
num_clicks, num_events = 0, 0

## Need to generate weather forecasts, but consistent for all days of the harvest.
# mode dictates "how rainy" the season is
mode = 0.2
weather = np.random.triangular(0.,mode,1.,size=garden_env_1_args['harvest_period'])

for plant_id in range(num_online_users):

    # Reset env and set done to False.
    env.reset(plant_id,weather)
    observation, _, done, _ = env.step(None)
    reward = None
    done = None
    while not done:
        action = agent.act(observation, reward, done)
        observation, reward, done, info = env.step(action['action'])

        # Used for calculating click through rate.
        num_clicks += 1 if reward == 1 and reward is not None else 0
        num_events += 1

ctr = num_clicks / num_events

print(f"Click Through Rate: {ctr:.4f}")


# ## Testing our first agent
# 
# Now we have created our popularity based agent, and we should test it against an even simpler baseline - one that performs no learning and recommends products uniformly at random. To do this, we will first load a more complex version of the toy data environment called `reco-gym-v1`.
# 
# Next, we will load another agent for our agent to compete against each other. Here you can see we make use of the `RandomAgent` and create an instance of it in addition to our `PopularityAgent`.

# In[5]:


import gym, recogym
#from recogym import env_1_args

from copy import deepcopy

#env_1_args['random_seed'] = 42

env = gym.make('garden-gym-v1')
env.init_gym(garden_env_1_args)

# Import the random agent.
from recogym.agents import RandomAgent, random_args, SimpleFarmerAgent,  WaitAgent

random_args['num_products'] = garden_env_1_args['num_products']
# Create the two agents.
num_products = garden_env_1_args['num_products'] ################
popularity_agent = PopularityAgent(Configuration(garden_env_1_args))
agent_rand = RandomAgent(Configuration({
    **garden_env_1_args,
    **random_args
}))

simple_agent = SimpleFarmerAgent(Configuration(garden_env_1_args))

wait_agent = WaitAgent(Configuration(garden_env_1_args))


# Now we have instances of our two agents. We can use the `test_agent` method from RecoGym and compare there performance.
# 
# To use `test_agent`, one must provide a copy of the current env, a copy of the agent class, the number of training users and the number of testing users. 

# In[6]:


# Credible interval of the CTR median and 0.025 0.975 quantile.
agent_success = recogym.test_agent(deepcopy(env), deepcopy(agent_rand), 1000, 1000)
print(f"RandomAgent success is {agent_success}")



# In[7]:

agent_success = recogym.test_agent(deepcopy(env), deepcopy(wait_agent), 1000, 1000) 
print(f"WaitAgent success is {agent_success}")






agent_success, plots = recogym.test_agent(deepcopy(env), deepcopy(simple_agent), 1000, 1000, plotting = False)
print(f"SimpleFarmerAgent success is {agent_success}")

plots = np.array(plots)
print(plots)