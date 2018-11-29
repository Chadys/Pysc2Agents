# reference used : https://itnext.io/refine-your-sparse-pysc2-agent-a3feb189bc68

import os

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import normalize

from pysc2.agents import base_agent
from pysc2.lib import actions, units

MIN_DIST = 1
MAX_DIST = 15
STEP_DIST = 3

DATA_FILE = 'defeat_roaches_agent_data'

ACTION_DO_NOTHING = 'nope'
ACTION_RETREAT = 'retreat_5'
ACTION_ATTACK = 'attack'


# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions_list, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions_list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.disallowed_actions = {}

    def choose_action(self, observation):
        self.check_state_exist(observation)

        state_action = self.q_table.ix[observation, :]

        if np.random.uniform() < self.epsilon:
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            action = np.random.choice(state_action.index)

        return action

    def learn(self, s, a, r, s_):
        if s == s_:
            return

        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]

        s_rewards = self.q_table.ix[s_, :]

        if s_ != 'terminal':
            q_target = r + self.gamma * s_rewards.max()
        else:
            q_target = r  # next state is terminal

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class DefeatRoachesAgent(base_agent.BaseAgent):

    def __init__(self):
        super().__init__()

        self.smart_actions = [
            ACTION_DO_NOTHING,
            ACTION_ATTACK,
            ACTION_RETREAT
        ]

        # for dist_i in range(MIN_DIST, MAX_DIST, STEP_DIST):
        #     self.smart_actions.append(ACTION_RETREAT + '_' + str(dist_i))

        self.qlearn = QLearningTable(actions_list=list(range(len(self.smart_actions))))

        self.previous_action = None
        self.previous_state = None
        self.marines_number = 0
        self.roaches_number = 0

        self.attack_cooldown = None
        self.action_to_do = None
        # self.failed_action = False

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)
        if "feature_units" not in obs_spec:
            raise Exception("This agent requires the feature_units observation.")
        self.init_values()
        self.marines_number = 9
        self.roaches_number = 4

    def reset(self):
        super().reset()
        self.init_values()
        self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

    def init_values(self):
        self.attack_cooldown = None

        self.action_to_do = None
        # self.failed_action = False

    @staticmethod
    def get_units_by_type(obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    @staticmethod
    def can_do(obs, action):
        return action in obs.observation.available_actions

    def split_action(self, action_id):
        smart_action = self.smart_actions[action_id]

        dist = 0
        # if '_' in smart_action:
        #     smart_action, dist = smart_action.split('_')

        return smart_action, dist

    def step(self, obs):
        if self.attack_cooldown is not None:
            self.attack_cooldown += 1

        if self.action_to_do is not None:
            return self.play_action(obs)

        roaches = [(unit.x, unit.y) for unit in self.get_units_by_type(obs, units.Zerg.Roach)]
        marines = [(unit.x, unit.y) for unit in self.get_units_by_type(obs, units.Terran.Marine)]

        # reward = self.get_reward(len(roaches), len(marines))
        reward = obs.reward

        if not roaches or not marines:
            self.attack_cooldown = None
            dist = 84 if not roaches else 0
            current_state = [dist, self.attack_cooldown]
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal' if obs.last() else str(current_state))
            self.previous_state = current_state
            self.previous_action = 0
            return actions.FUNCTIONS.no_op()

        (min_dist_indexes, min_dists) = pairwise_distances_argmin_min(marines, roaches)
        min_index = np.argmin(min_dists)
        min_marine, min_roach = marines[min_index], roaches[min_dist_indexes[min_index]]
        min_dist = round(min_dists[min_index], 2)

        current_state = [min_dist, self.attack_cooldown]
        if self.previous_action is not None:
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal' if obs.last() else str(current_state))

        direction = normalize(np.subtract(min_marine, min_roach).reshape(1, -1)).ravel()
        # excluded_actions = []
        # for dist_i in range(MIN_DIST, MAX_DIST):
        #     potential_new_pos = np.add(min_marine, int(dist) * direction)
        #     if any(potential_new_pos >= 84) or any(potential_new_pos <= 0):
        #         excluded_actions = range(dist_i + 2, MAX_DIST + 3)  # account for ACTION_DO_NOTHING and ACTION_ATTACK
        #         break

        action_id = self.qlearn.choose_action(str(current_state))
        print('current state : ', current_state)
        print('action choosen : ', self.smart_actions[action_id])
        self.previous_state = current_state
        self.previous_action = action_id
        return self.use_action(action_id, min_roach, min_marine, direction)

    def use_action(self, action_index, min_roach, min_marine, direction):
        action, dist = self.split_action(action_index)
        if action == ACTION_ATTACK:
            self.action_to_do = (actions.FUNCTIONS.Attack_screen, ["now", min_roach])
            self.attack_cooldown = -1
        elif action == ACTION_RETREAT:
            self.action_to_do = (
                actions.FUNCTIONS.Move_screen, ["now", self.get_coord_in_dir(min_marine, direction, dist)])
        else:
            return actions.FUNCTIONS.no_op()

        return actions.FUNCTIONS.select_army("select")

    def play_action(self, obs):
        action, args = self.action_to_do
        self.action_to_do = None
        if not self.can_do(obs, action.id):
            # self.failed_action = True
            print("FAILED")
            return actions.FUNCTIONS.no_op()
        return action(*args)

    # def get_reward(self, n_roaches, n_marines):
    #     reward = 0
    #     if self.failed_action:
    #         self.failed_action = False
    #         reward -= 1
    #     roaches_killed = self.roaches_number - n_roaches
    #     if roaches_killed > 0:
    #         reward += roaches_killed * 2
    #     # roaches regularly appear so don't use appearing roaches to calculate reward, only killed ones
    #     self.roaches_number = n_roaches
    #     reward += (n_marines - self.marines_number) * 2
    #     return reward

    @staticmethod
    def get_coord_in_dir(min_marine, direction, dist):
        coord =  np.add(min_marine, int(dist) * direction)
        coord = np.minimum(coord, [0, 0])
        coord = np.maximum(coord, [83, 83])
        return coord
