# reference used :
# https://itnext.io/refine-your-sparse-pysc2-agent-a3feb189bc68
# https://itnext.io/build-a-zerg-bot-with-pysc2-2-0-295375d2f58e

import random
import os
import enum
from absl import app

import numpy as np
import pandas as pd
# from collections import Counter

from pysc2.agents import base_agent
from pysc2.lib import actions, features, units, named_array
from pysc2.env import sc2_env

# game state on start
# MineralField750 x4
# VespeneGeyser x2
# MineralField x4
# Hatchery
# Overlord
# Larva x3
# Drone x12

MINIMAP_SIZE = 64

DATA_FILE = 'simple_zerg_agent_data'


# TODO check cando for every action


class State(enum.IntEnum):
    num_base = 0
    num_spawning_pool = 1
    num_queen = 2
    num_extractor = 3
    num_roach_warren = 4
    num_larvae = 5
    army_count = 6
    idle_worker_count = 7
    enemy_race = 8
    hot_squares = 9
    green_square = 13


class Race(enum.Enum):
    Terran = units.Terran
    Zerg = units.Zerg
    Protoss = units.Protoss
    Unknown = None


class Action(enum.IntEnum):
    ACTION_DO_NOTHING = 0
    ACTION_TRAIN_DRONE = 1
    ACTION_TRAIN_ZERGLING = 2
    ACTION_TRAIN_ROACH = 3
    ACTION_TRAIN_OVERLORD = 4
    ACTION_ATTACK = 5
    ACTION_BUILD_HATCHERY = 9
    ACTION_BUILD_SPAWNING_POOL = 10
    ACTION_TRAIN_QUEEN = 11
    ACTION_BUILD_EXTRACTOR = 12
    ACTION_BUILD_ROACH_WARREN = 13
    ACTION_BUILD_LAIR = 14
    ACTION_UPGRADE_ZERGLINGS = 15
    ACTION_UPGRADE_ROACHES = 16
    ACTION_SPAWN_LARVA = 17
    ACTION_SCOUT_ENEMY = 18
    ACTION_IDENTIFY_ENEMY = 19
    ACTION_STOP_SCOUTING = 20


class PatrolState(enum.Enum):
    NOPE = enum.auto()
    ENROUTE = enum.auto()
    RETURNING = enum.auto()


smart_actions = [Action.ACTION_DO_NOTHING.name, Action.ACTION_TRAIN_DRONE.name, Action.ACTION_TRAIN_ZERGLING.name,
                 Action.ACTION_TRAIN_ROACH.name, Action.ACTION_TRAIN_OVERLORD.name]

for mm_x in range(MINIMAP_SIZE // 4, MINIMAP_SIZE, MINIMAP_SIZE // 2):
    for mm_y in range(MINIMAP_SIZE // 4, MINIMAP_SIZE, MINIMAP_SIZE // 2):
        smart_actions.append('{}_{}_{}'.format(Action.ACTION_ATTACK.name, mm_x, mm_y))


# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, possible_actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = possible_actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.disallowed_actions = {}

    def choose_action(self, observation, excluded_actions=None):
        if excluded_actions is None:
            excluded_actions = []
        self.check_state_exist(observation)

        self.disallowed_actions[observation] = excluded_actions

        state_action = self.q_table.ix[observation, :]

        for excluded_action in excluded_actions:
            del state_action[excluded_action]

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

        if s_ in self.disallowed_actions:
            for excluded_action in self.disallowed_actions[s_]:
                del s_rewards[excluded_action]

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


class SparseZergAgent(base_agent.BaseAgent):
    center = MINIMAP_SIZE / 2
    top_left_pos = (12, 16)
    bot_right_pos = (49, 49)
    resource_loc = {
        units.Neutral.MineralField750.value: np.array(
            [(0, 13), (0, 23), (0, 29), (0, 40), (2, 16), (2, 19), (2, 34), (2, 36), (2, 44), (6, 13), (6, 23), (6, 29),
             (6, 40), (6, 55), (6, 57), (6, 65), (6, 68), (7, 16), (7, 19), (7, 34), (7, 36), (7, 44), (7, 61), (7, 62),
             (11, 13), (11, 23), (11, 29), (11, 40), (11, 55), (11, 57), (11, 65), (11, 68), (13, 16), (13, 19),
             (13, 34), (13, 36), (13, 44), (13, 61), (13, 62), (16, 13), (16, 23), (16, 29), (16, 40), (16, 55),
             (16, 57), (16, 65), (16, 68), (18, 16), (18, 19), (18, 34), (18, 36), (18, 44), (18, 61), (18, 62),
             (21, 13), (21, 19), (21, 23), (21, 29), (21, 40), (21, 42), (21, 44), (21, 55), (21, 57), (21, 65),
             (21, 68), (23, 16), (23, 34), (23, 36), (23, 61), (23, 62), (24, 29), (24, 55), (25, 32), (27, 13),
             (27, 23), (27, 40), (27, 57), (27, 65), (27, 68), (28, 16), (28, 26), (28, 34), (28, 36), (28, 51),
             (28, 61), (28, 62), (31, 34), (31, 36), (31, 62), (32, 13), (32, 23), (32, 25), (32, 40), (32, 57),
             (32, 65), (32, 68), (34, 16), (34, 26), (34, 51), (34, 61), (35, 23), (35, 68), (37, 13), (37, 40),
             (37, 57), (37, 65), (38, 43), (39, 16), (39, 26), (39, 43), (39, 51), (39, 61), (42, 13), (42, 16),
             (42, 22), (42, 40), (42, 48), (42, 57), (42, 61), (42, 65), (44, 26), (44, 43), (44, 51), (45, 40),
             (45, 65), (46, 21), (48, 13), (48, 22), (48, 48), (48, 57), (49, 15), (49, 26), (49, 41), (49, 43),
             (49, 51), (52, 2), (52, 5), (52, 30), (52, 47), (53, 2), (53, 5), (53, 13), (53, 22), (53, 30), (53, 47),
             (53, 48), (53, 57), (55, 15), (55, 26), (55, 41), (55, 43), (55, 51), (56, 13), (56, 57), (58, 2), (58, 5),
             (58, 22), (58, 30), (58, 47), (58, 48), (59, 9), (59, 54), (60, 9), (60, 15), (60, 26), (60, 41), (60, 43),
             (60, 51), (60, 54), (63, 2), (63, 5), (63, 20), (63, 22), (63, 30), (63, 47), (63, 48), (63, 64), (65, 9),
             (65, 15), (65, 26), (65, 41), (65, 43), (65, 51), (65, 54), (69, 2), (69, 5), (69, 20), (69, 22), (69, 30),
             (69, 47), (69, 48), (69, 64), (70, 9), (70, 15), (70, 26), (70, 41), (70, 43), (70, 51), (70, 54), (74, 2),
             (74, 5), (74, 20), (74, 22), (74, 30), (74, 47), (74, 48), (74, 64), (76, 9), (76, 15), (76, 26), (76, 41),
             (76, 43), (76, 51), (76, 54), (79, 2), (79, 5), (79, 20), (79, 22), (79, 30), (79, 47), (79, 48), (79, 64),
             (81, 9), (81, 15), (81, 26), (81, 41), (81, 43), (81, 51), (81, 54)]),
        units.Neutral.MineralField.value: np.array(
            [(0, 15), (0, 30), (0, 37), (0, 40), (0, 41), (2, 16), (2, 26), (2, 51), (4, 20), (4, 36), (4, 62), (4, 64),
             (6, 15), (6, 30), (6, 37), (6, 40), (6, 41), (6, 65), (6, 75), (7, 16), (7, 26), (7, 51), (7, 61), (9, 20),
             (9, 36), (9, 62), (9, 64), (11, 15), (11, 30), (11, 37), (11, 40), (11, 41), (11, 65), (11, 75), (13, 16),
             (13, 26), (13, 51), (13, 61), (14, 20), (14, 36), (14, 62), (14, 64), (16, 15), (16, 30), (16, 37),
             (16, 40), (16, 41), (16, 65), (16, 75), (18, 16), (18, 26), (18, 51), (18, 61), (20, 20), (20, 36),
             (20, 62), (20, 64), (21, 15), (21, 26), (21, 30), (21, 37), (21, 40), (21, 41), (21, 51), (21, 65),
             (21, 75), (23, 16), (23, 61), (24, 15), (24, 41), (25, 20), (25, 36), (25, 39), (25, 46), (25, 62),
             (25, 64), (27, 30), (27, 37), (27, 40), (27, 65), (27, 75), (28, 16), (28, 28), (28, 61), (30, 20),
             (30, 36), (30, 62), (30, 64), (31, 22), (31, 48), (32, 22), (32, 30), (32, 37), (32, 40), (32, 48),
             (32, 65), (32, 75), (34, 16), (34, 61), (35, 2), (35, 20), (35, 30), (35, 36), (35, 37), (35, 40),
             (35, 47), (35, 62), (35, 64), (35, 65), (35, 75), (37, 22), (37, 48), (38, 20), (38, 64), (39, 16),
             (39, 61), (41, 2), (41, 36), (41, 47), (41, 62), (42, 22), (42, 25), (42, 48), (44, 16), (44, 61),
             (45, 19), (45, 44), (46, 2), (46, 19), (46, 36), (46, 44), (46, 47), (46, 62), (48, 22), (48, 48), (49, 1),
             (49, 8), (49, 16), (49, 27), (49, 34), (49, 36), (49, 43), (49, 61), (49, 62), (51, 2), (51, 19), (51, 44),
             (51, 47), (52, 16), (52, 61), (53, 22), (53, 48), (55, 1), (55, 8), (55, 27), (55, 34), (55, 43), (56, 2),
             (56, 19), (56, 44), (56, 47), (58, 22), (58, 48), (59, 23), (59, 68), (60, 1), (60, 8), (60, 23), (60, 27),
             (60, 34), (60, 43), (60, 68), (62, 2), (62, 19), (62, 44), (62, 47), (63, 13), (63, 22), (63, 48),
             (63, 57), (65, 1), (65, 8), (65, 23), (65, 27), (65, 34), (65, 43), (65, 68), (67, 2), (67, 19), (67, 44),
             (67, 47), (69, 13), (69, 22), (69, 48), (69, 57), (70, 1), (70, 8), (70, 23), (70, 27), (70, 34), (70, 43),
             (70, 68), (72, 2), (72, 19), (72, 44), (72, 47), (74, 13), (74, 22), (74, 48), (74, 57), (76, 1), (76, 8),
             (76, 23), (76, 27), (76, 34), (76, 43), (76, 68), (77, 2), (77, 19), (77, 44), (77, 47), (79, 13),
             (79, 22), (79, 48), (79, 57), (81, 1), (81, 8), (81, 23), (81, 27), (81, 34), (81, 43), (83, 2), (83, 19),
             (83, 44), (83, 47)]),
        units.Neutral.VespeneGeyser.value: np.array(
            [(0, 40), (0, 65), (2, 13), (2, 57), (4, 5), (4, 30), (4, 48), (6, 40), (6, 65), (7, 13), (7, 57), (9, 5),
             (9, 30), (9, 48), (11, 40), (11, 65), (13, 13), (13, 57), (14, 5), (14, 30), (14, 48), (15, 26), (15, 51),
             (16, 26), (16, 40), (16, 51), (16, 65), (18, 13), (18, 57), (20, 5), (20, 30), (20, 48), (21, 26),
             (21, 40), (21, 51), (21, 65), (22, 5), (22, 30), (22, 43), (23, 13), (23, 43), (23, 56), (23, 57),
             (25, 48), (27, 26), (27, 40), (27, 51), (27, 65), (28, 13), (28, 43), (28, 57), (30, 48), (32, 26),
             (32, 40), (32, 51), (32, 65), (33, 48), (34, 13), (34, 43), (34, 57), (37, 26), (37, 40), (37, 51),
             (37, 65), (39, 13), (39, 43), (39, 57), (42, 26), (42, 40), (42, 51), (42, 65), (44, 13), (44, 43),
             (44, 57), (48, 26), (48, 40), (48, 51), (48, 65), (49, 13), (49, 43), (49, 57), (50, 16), (50, 79),
             (51, 16), (51, 79), (53, 26), (53, 40), (53, 51), (53, 65), (55, 13), (55, 43), (55, 57), (56, 16),
             (56, 79), (58, 21), (58, 26), (58, 40), (58, 51), (58, 65), (60, 13), (60, 43), (60, 57), (61, 34),
             (61, 40), (61, 65), (61, 78), (62, 16), (62, 34), (62, 78), (62, 79), (63, 26), (63, 51), (65, 13),
             (65, 43), (65, 57), (67, 16), (67, 34), (67, 78), (67, 79), (68, 13), (68, 57), (69, 26), (69, 51),
             (70, 43), (72, 16), (72, 34), (72, 78), (72, 79), (74, 26), (74, 51), (76, 43), (77, 16), (77, 34),
             (77, 78), (77, 79), (79, 26), (79, 51), (81, 43), (83, 16), (83, 34)])
    }

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)
        if "feature_units" not in obs_spec[0]:
            raise Exception("This agent requires the feature_units observation.")

        self.qlearn = QLearningTable(possible_actions=list(range(len(smart_actions))))

        self.init_values()

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def reset(self):
        super().reset()

        self.init_values()

        self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
        # TODO update graph

    def init_values(self):
        self.move_number = 0
        self.previous_action = None
        self.previous_state = None
        self.hardcoded_action = False
        self.enemy_race = Race.Unknown
        self.upgrade_zergling = False
        self.upgrade_roaches = False
        self.patrol_state = PatrolState.NOPE

    def transform_distance(self, x, x_distance, y, y_distance):
        return [x - x_distance, y - y_distance] if not self.base_top_left else [x + x_distance, y + y_distance]

    def transform_location(self, x, y):
        if not self.base_top_left:
            return [MINIMAP_SIZE - x, MINIMAP_SIZE - y]

        return [x, y]

    @staticmethod
    def split_action(action_id):
        smart_action = smart_actions[action_id]

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return smart_action, x, y

    def init_starting_pos(self, obs):
        player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                              features.PlayerRelative.SELF).nonzero()
        xmean = player_x.mean()
        ymean = player_y.mean()

        if xmean < self.center and ymean < self.center:
            self.attack_coordinates = self.bot_right_pos
            self.base_top_left = True
        else:
            self.attack_coordinates = self.top_left_pos
            self.base_top_left = False

    @staticmethod
    def get_units_by_type(obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    @staticmethod
    def can_do(obs, action):
        return action in obs.observation.available_actions

    @staticmethod
    def get_unit_type_name(unit_type):
        unit_enum = SparseZergAgent.get_unit_type(unit_type)
        return unit_enum.name if unit_enum is not None else 'unknown enum {}'.format(unit_type)

    @staticmethod
    def get_unit_type(unit_type):
        try:
            return units.Neutral(unit_type)
        except ValueError:
            try:
                return units.Zerg(unit_type)
            except ValueError:
                try:
                    return units.Terran(unit_type)
                except ValueError:
                    try:
                        return units.Protoss(unit_type)
                    except ValueError:
                        return None

    @staticmethod
    def process_obs(obs, display=False):
        if not display:
            return
        print('player_idle')
        print(obs.observation.player.idle_worker_count)
        print('player_army')
        print(obs.observation.player.army_count)
        print('player_larva')
        print(obs.observation.player.larva_count)
        print('player_vespene')
        print(obs.observation.player.vespene)
        print('player_minerals')
        print(obs.observation.player.minerals)
        print('player_food_used')
        print(obs.observation.player.food_used)
        print('player_food_cap')
        print(obs.observation.player.food_cap)
        print('player_food_army')
        print(obs.observation.player.food_army)
        print('player_food_workers')
        print(obs.observation.player.food_workers)

    def get_alliance_squares(self, obs, alliance):
        hot_squares = np.zeros(4)
        enemies_y, enemies_x = (obs.observation.feature_minimap.player_relative == alliance).nonzero()
        for enemy_x, enemy_y in zip(enemies_x, enemies_y):
            hot_squares[enemy_x // 32 + (enemy_y // 32) * 2] = 1
        if not self.base_top_left:
            hot_squares = hot_squares[::-1]
        return hot_squares

    def get_current_state(self, obs):
        base = len(self.get_units_by_type(obs, units.Zerg.Hatchery))
        base = base if base > 0 else len(self.get_units_by_type(obs, units.Zerg.Lair)) * 2
        current_state = named_array.NamedNumpyArray([
            base,
            len(self.get_units_by_type(obs, units.Zerg.SpawningPool)),
            len(self.get_units_by_type(obs, units.Zerg.Queen)),
            len(self.get_units_by_type(obs, units.Zerg.Extractor)),
            len(self.get_units_by_type(obs, units.Zerg.RoachWarren)),
            len(self.get_units_by_type(obs, units.Zerg.Larva)),  # TODO replace with obs.observation.player.larva_count?
            obs.observation.player.army_count,
            obs.observation.player.idle_worker_count,
            self.enemy_race.value,
            *self.get_alliance_squares(obs, features.PlayerRelative.ENEMY),
            *self.get_alliance_squares(obs, features.PlayerRelative.SELF)
        ], names=State, dtype=np.int32)

        return current_state

    def get_excluded_actions(self, obs):
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)

        excluded_actions = []
        # actions excluded because of space
        if not free_supply:
            excluded_actions.append(Action.ACTION_BUILD_DRONE.value)
            excluded_actions.append(Action.ACTION_BUILD_ZERGLING.value)
            excluded_actions.append(Action.ACTION_BUILD_ROACH.value)
        elif free_supply < 2:
            excluded_actions.append(Action.ACTION_BUILD_ROACH.value)

        # actions excluded because of army
        if not obs.observation.player.army_count:
            for i in range(Action.ACTION_ATTACK.value, Action.ACTION_ATTACK.value + 4):
                excluded_actions.append(i)

        # actions excluded because of resources
        if obs.observation.player.minerals < 25:
            excluded_actions.append(Action.ACTION_BUILD_ZERGLING.value)
            excluded_actions.append(Action.ACTION_BUILD_DRONE.value)
            excluded_actions.append(Action.ACTION_BUILD_ROACH.value)
            excluded_actions.append(Action.ACTION_BUILD_OVERLORD.value)
        elif obs.observation.player.minerals < 50:
            excluded_actions.append(Action.ACTION_BUILD_DRONE.value)
            excluded_actions.append(Action.ACTION_BUILD_ROACH.value)
            excluded_actions.append(Action.ACTION_BUILD_OVERLORD.value)
        elif obs.observation.player.minerals < 75:
            excluded_actions.append(Action.ACTION_BUILD_ROACH.value)
            excluded_actions.append(Action.ACTION_BUILD_OVERLORD.value)
        elif obs.observation.player.minerals < 100:
            excluded_actions.append(Action.ACTION_BUILD_OVERLORD.value)
        elif obs.observation.player.vespene < 25:
            excluded_actions.append(Action.ACTION_BUILD_ROACH.value)

        # actions excluded because of larvae
        larvae = self.get_units_by_type(obs, units.Zerg.Larva)
        if not len(larvae):
            excluded_actions.append(Action.ACTION_BUILD_ZERGLING.value)
            excluded_actions.append(Action.ACTION_BUILD_DRONE.value)
            excluded_actions.append(Action.ACTION_BUILD_ROACH.value)
            excluded_actions.append(Action.ACTION_BUILD_OVERLORD.value)

        return excluded_actions

    def get_hardcoded_action(self, obs, current_state):
        if self.enemy_race == Race.Unknown:
            if self.patrol_state == PatrolState.NOPE:
                return Action.ACTION_SCOUT_ENEMY
            # else self.patrol_state == PatrolState.ENROUTE
            if obs.observation.feature_minimap.visibility_map.item(
                    self.attack_coordinates) == features.Visibility.VISIBLE:
                return Action.ACTION_IDENTIFY_ENEMY
        if self.patrol_state == PatrolState.RETURNING:
            overlord = [o for o in self.get_units_by_type(obs, units.Zerg.Overlord) if o.order_length > 0]
            if len(overlord) > 0:
                return Action.ACTION_STOP_SCOUTING

        if not current_state.num_base:
            if obs.observation.player.minerals >= 300:
                return Action.ACTION_BUILD_HATCHERY
            return None
        if not current_state.num_spawning_pool:
            if obs.observation.player.minerals >= 200:
                return Action.ACTION_BUILD_SPAWNING_POOL
            return None
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if not current_state.num_queen:
            if obs.observation.player.minerals >= 150 and free_supply >= 2:
                return Action.ACTION_TRAIN_QUEEN
            return None
        queen = self.get_units_by_type(obs, units.Zerg.Queen)
        if len(queen) > 0:
            queen = queen[0]
            if queen.energy >= 25:
                return Action.ACTION_SPAWN_LARVA
        if not self.upgrade_zergling:
            if obs.observation.player.minerals >= 100 and obs.observation.player.vespene >= 100:
                return Action.ACTION_UPGRADE_ZERGLINGS
        if current_state.num_extractor < 2:
            if obs.observation.player.minerals >= 25:
                return Action.ACTION_BUILD_EXTRACTOR
            return None
        if not current_state.num_roach_warren:
            if obs.observation.player.minerals >= 150:
                return Action.ACTION_BUILD_ROACH_WARREN
            return None
        if current_state.num_base < 2:
            if obs.observation.player.minerals >= 150 and obs.observation.player.vespene >= 100:
                return Action.ACTION_BUILD_LAIR
            return None
        if not self.upgrade_roaches:
            if obs.observation.player.minerals >= 100 and obs.observation.player.vespene >= 100:
                return Action.ACTION_UPGRADE_ROACHES
        return None

    def step(self, obs):
        super().step(obs)

        self.process_obs(obs, True)

        if obs.first():
            self.init_starting_pos(obs)

        if obs.last():
            reward = obs.reward
            if self.hardcoded_action:
                self.previous_action = Action.ACTION_DO_NOTHING.name
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
            return actions.FUNCTIONS.no_op()

        if self.move_number == 0:
            current_state = self.get_current_state(obs)

            if self.previous_action is not None and not self.hardcoded_action:
                self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))

            rl_action = self.get_hardcoded_action(obs, current_state)
            if rl_action is not None:
                rl_action = rl_action.name
                self.hardcoded_action = True
            else:
                rl_action = self.qlearn.choose_action(str(current_state), self.get_excluded_actions(obs))
                self.hardcoded_action = False

            self.previous_state = current_state
            self.previous_action = rl_action

            return self.get_first_action(obs)

        if self.move_number == 1:
            return self.get_second_action(obs)

    @staticmethod
    def get_default_action():
        actions.FUNCTIONS.no_op()

    def get_first_action(self, obs):
        smart_action, x, y = self.split_action(self.previous_action)
        if smart_action == Action.ACTION_DO_NOTHING.name:
            return actions.FUNCTIONS.no_op()
        self.move_number += 1
        if smart_action == Action.ACTION_TRAIN_DRONE.name or smart_action == Action.ACTION_TRAIN_ZERGLING.name or smart_action == Action.ACTION_TRAIN_ROACH.name or smart_action == Action.ACTION_TRAIN_OVERLORD.name or smart_action == Action.ACTION_TRAIN_QUEEN.name:
            # TODO not sure it works since larva_count stays zero ?
            return actions.FUNCTIONS.select_larva()
        elif smart_action == Action.ACTION_ATTACK.name:
            return actions.FUNCTIONS.select_army(actions.SelectAdd.select.name)
        elif smart_action == Action.ACTION_BUILD_HATCHERY.name or smart_action == Action.ACTION_BUILD_SPAWNING_POOL.name or smart_action == Action.ACTION_BUILD_EXTRACTOR.name or smart_action == Action.ACTION_BUILD_ROACH_WARREN.name:
            return actions.FUNCTIONS.select_idle_worker(actions.SelectWorker.select.name)
        elif smart_action == Action.ACTION_BUILD_LAIR.name:
            hatchery = self.get_units_by_type(obs, units.Zerg.Hatchery)
            if len(hatchery) > 0:
                hatchery = hatchery[0]
                return actions.FUNCTIONS.select_point(actions.SelectPointAct.select.name, (hatchery.x, hatchery.y))
        elif smart_action == Action.ACTION_UPGRADE_ZERGLINGS.name:
            s_pool = self.get_units_by_type(obs, units.Zerg.SpawningPool)
            if len(s_pool) > 0:
                s_pool = s_pool[0]
                return actions.FUNCTIONS.select_point(actions.SelectPointAct.select.name, (s_pool.x, s_pool.y))
        elif smart_action == Action.ACTION_UPGRADE_ROACHES.name:
            r_warren = self.get_units_by_type(obs, units.Zerg.RoachWarren)
            if len(r_warren) > 0:
                r_warren = r_warren[0]
                return actions.FUNCTIONS.select_point(actions.SelectPointAct.select.name, (r_warren.x, r_warren.y))
        elif smart_action == Action.ACTION_SPAWN_LARVA.name:
            queen = self.get_units_by_type(obs, units.Zerg.Queen)
            if len(queen) > 0:
                queen = queen[0]
                return actions.FUNCTIONS.select_point(actions.SelectPointAct.select.name, (queen.x, queen.y))
        elif smart_action == Action.ACTION_SCOUT_ENEMY.name:
            overlord = self.get_units_by_type(obs, units.Zerg.Overlord)
            if len(overlord) > 0:
                overlord = overlord[0]
                return actions.FUNCTIONS.select_point(actions.SelectPointAct.select.name, (overlord.x, overlord.y))
        elif smart_action == Action.ACTION_IDENTIFY_ENEMY.name:
            return actions.FUNCTIONS.move_camera(self.attack_coordinates)
        elif smart_action == Action.ACTION_STOP_SCOUTING.name:
            overlord = [o for o in self.get_units_by_type(obs, units.Zerg.Overlord) if o.order_length > 0]
            if len(overlord) > 0:
                overlord = overlord[0]
                if not overlord.is_selected:
                    return actions.FUNCTIONS.select_point(actions.SelectPointAct.select.name, (overlord.x, overlord.y))
                self.move_number = 0
                return actions.FUNCTIONS.Stop_quick(actions.Queued.now.name)
        # impossible action selected
        self.move_number = 0
        return self.get_default_action()

    def get_second_action(self, obs):
        smart_action, x, y = self.split_action(self.previous_action)
        self.move_number = 0

        if smart_action == Action.ACTION_TRAIN_DRONE.name:
            return actions.FUNCTIONS.Train_Drone_quick(actions.Queued.now.name)
        elif smart_action == Action.ACTION_TRAIN_ZERGLING.name:
            return actions.FUNCTIONS.Train_Zergling_quick(actions.Queued.now.name)
        elif smart_action == Action.ACTION_TRAIN_ROACH.name:
            return actions.FUNCTIONS.Morph_Lair_quick(actions.Queued.now.name)
        elif smart_action == Action.ACTION_TRAIN_OVERLORD.name:
            return actions.FUNCTIONS.Train_Overlord_quick(actions.Queued.now.name)
        elif smart_action == Action.ACTION_TRAIN_QUEEN.name:
            return actions.FUNCTIONS.Train_Queen_quick(actions.Queued.now.name)
        elif smart_action == Action.ACTION_ATTACK.name:
            x_offset = random.randint(-1, 1)
            y_offset = random.randint(-1, 1)
            return actions.FUNCTIONS.Attack_minimap(actions.Queued.now.name,
                                                    self.transform_location(int(x) + x_offset * 8,
                                                                            int(y) + y_offset * 8))

        # TODO choose coordinates
        # elif smart_action == Action.ACTION_BUILD_HATCHERY.name:
        # elif smart_action == Action.ACTION_BUILD_SPAWNING_POOL.name:
        # elif smart_action == Action.ACTION_BUILD_EXTRACTOR.name:
        # elif smart_action == Action.ACTION_BUILD_ROACH_WARREN.name:
        elif smart_action == Action.ACTION_BUILD_LAIR.name:
            return actions.FUNCTIONS.Morph_Lair_quick(actions.Queued.now.name)
        elif smart_action == Action.ACTION_UPGRADE_ZERGLINGS.name:
            return actions.FUNCTIONS.Research_ZerglingMetabolicBoost_quick(actions.Queued.now.name)
        elif smart_action == Action.ACTION_UPGRADE_ROACHES.name:
            return actions.FUNCTIONS.Research_GlialRegeneration_quick(actions.Queued.now.name)
        elif smart_action == Action.ACTION_SPAWN_LARVA.name:
            hatchery = self.get_units_by_type(obs, units.Zerg.Hatchery)
            if len(hatchery) > 0:
                hatchery = hatchery[0]
                return actions.FUNCTIONS.Effect_InjectLarva_screen(actions.Queued.now.name, (hatchery.x, hatchery.y))
        elif smart_action == Action.ACTION_SCOUT_ENEMY.name:
            self.patrol_state = PatrolState.ENROUTE
            return actions.FUNCTIONS.Patrol_minimap(actions.Queued.now.name, self.attack_coordinates)
        elif smart_action == Action.ACTION_IDENTIFY_ENEMY.name:
            enemy_units = [unit for unit in obs.observation.feature_units
                           if unit.alliance == features.PlayerRelative.ENEMY]
            if len(enemy_units) > 0:
                self.patrol_state = PatrolState.RETURNING
                enemy_unit_type = self.get_unit_type(enemy_units[0].unit_type)
                self.enemy_race = Race(enemy_unit_type.__class__) if enemy_unit_type is not None else Race.Unknown
                return actions.FUNCTIONS.move_camera([18, 22] if self.base_top_left else [40, 46])
        elif smart_action == Action.ACTION_STOP_SCOUTING.name:
            self.patrol_state = PatrolState.NOPE
            return actions.FUNCTIONS.Stop_quick(actions.Queued.now.name)

        # impossible action selected
        return self.get_default_action()


def main(_):
    agent = SparseZergAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="Simple64",
                    players=[sc2_env.Agent(sc2_env.Race.zerg),
                             sc2_env.Bot(sc2_env.Race.random,
                                         sc2_env.Difficulty.very_easy)],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=84, minimap=MINIMAP_SIZE),
                        use_feature_units=True,
                        hide_specific_actions=False),
                    step_mul=16,
                    game_steps_per_episode=0,
                    visualize=True) as env:

                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass


def enable_remote_debug():
    import sys
    sys.path.append("pycharm-debug-py3k.egg")
    import pydevd
    pydevd.settrace('192.168.0.10', port=51234, stdoutToServer=True, stderrToServer=True, suspend=False)


if __name__ == "__main__":
    enable_remote_debug()
    app.run(main)
