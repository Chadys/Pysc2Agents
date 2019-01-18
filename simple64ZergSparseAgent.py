# reference used :
# https://itnext.io/refine-your-sparse-pysc2-agent-a3feb189bc68
# https://itnext.io/build-a-zerg-bot-with-pysc2-2-0-295375d2f58e

# import random
# import math
import os
import enum
from absl import app

import numpy as np
import pandas as pd
# from collections import Counter

from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
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

class Race(enum.Enum):
    Terran = units.Terran
    Zerg = units.Zerg
    Protoss = units.Protoss
    Unknown = None

class Action(enum.Enum):
    ACTION_DO_NOTHING = enum.auto()
    ACTION_BUILD_DRONE = enum.auto()
    ACTION_BUILD_ZERGLING = enum.auto()
    ACTION_BUILD_ROACH = enum.auto()
    ACTION_BUILD_OVERLORD = enum.auto()
    ACTION_ATTACK_ZERGLINGS = enum.auto()
    ACTION_ATTACK_ROACHES = enum.auto()
    ACTION_GATHER_GAZ_MAIN = enum.auto()
    ACTION_GATHER_GAZ_SECOND = enum.auto()
    ACTION_GATHER_GAZ_ENNEMY = enum.auto()
    ACTION_GATHER_MINERAL_MAIN = enum.auto()
    ACTION_GATHER_MINERAL_SECOND = enum.auto()
    ACTION_GATHER_MINERAL_ENNEMY = enum.auto()


smart_actions = [a.name for a in Action if a != Action.ACTION_ATTACK_ZERGLINGS and a != Action.ACTION_ATTACK_ROACHES]

for mm_x in range(MINIMAP_SIZE//4, MINIMAP_SIZE, MINIMAP_SIZE//2):
    for mm_y in range(MINIMAP_SIZE//4, MINIMAP_SIZE, MINIMAP_SIZE//2):
        smart_actions.append('{}_{}_{}'.format(Action.ACTION_ATTACK_ZERGLINGS.name, mm_x, mm_y))
        smart_actions.append('{}_{}_{}'.format(Action.ACTION_ATTACK_ROACHES.name, mm_x, mm_y))


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

    # def setup(self, obs_spec, action_spec):
    # self.resource_loc = {units.Neutral.MineralField.value: set(),
    #                      units.Neutral.MineralField750.value: set(),
    #                      units.Neutral.VespeneGeyser.value: set()}
    # self.MAX_X_SIZE = obs_spec[0]['feature_minimap'][2]  # x
    # self.MAX_Y_SIZE = obs_spec[0]['feature_minimap'][1]  # y
    # self.screen_x = 0
    # self.screen_y = 0


    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)
        if "feature_units" not in obs_spec[0]:
            raise Exception("This agent requires the feature_units observation.")

        self.qlearn = QLearningTable(possible_actions=list(range(len(smart_actions))))

        self.previous_action = None
        self.previous_state = None

        self.cc_y = None
        self.cc_x = None

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def reset(self):
        super().reset()

        self.move_number = 0
        self.enemy_race = Race.Unknown

        # self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

    def transform_distance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

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
        # units_for_display = []
        # for unit in obs.observation.feature_units:
        #     units_for_display.append(
        #         (self.get_unit_type_name(unit.unit_type), features.PlayerRelative(unit.alliance).name))
        #     ressource_set = self.resource_loc.get(unit.unit_type, None)
        #     if ressource_set is not None:
        #         if 0 <= unit.x < 84 and unit.y >= 0 and unit.y < 84:
        #             ressource_set.add((unit.x, unit.y))
        # if not display:
        #     return
        # print('units')
        # print(Counter(units_for_display))
        # print('resource_loc')
        # plan = []
        # for i in range(84):
        #     plan.append([0] * 84)
        # for resource, loc in self.resource_loc.items():
        #     loc_sorted = list(loc)
        #     loc_sorted.sort()
        #     print('{}: {}'.format(units.Neutral(resource).name, loc_sorted))
        #     for l in loc:
        #         plan[l[0]][l[1]] = resource
        # print(plan)

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

    def step(self, obs):
        super().step(obs)

        self.process_obs(obs, True)
        # self.screen_x += 1
        # if self.screen_x >= self.MAX_X_SIZE:
        #     self.screen_x = 0
        #     self.screen_y += 12
        #     if self.screen_y >= self.MAX_Y_SIZE:
        #         self.process_obs(obs, True)
        #         self.screen_y = 0
        # print('cam at {}, {}'.format(int(self.screen_x), int(self.screen_y)))
        # return actions.FUNCTIONS.move_camera([self.screen_x, self.screen_y])

        if obs.first():
            self.init_starting_pos(obs)

        if self.move_number == 0:
            overlord = self.get_units_by_type(obs, units.Zerg.Overlord)
            if len(overlord) > 0:
                self.move_number += 1
                overlord = overlord[0]
                return actions.FUNCTIONS.select_point(actions.SelectPointAct.select.name, (overlord.x, overlord.y))
        if self.move_number == 1:
            self.move_number += 1
            return actions.FUNCTIONS.Patrol_minimap(actions.Queued.now.name, self.attack_coordinates)
        if self.move_number == 2 and obs.observation.feature_minimap.visibility_map.item(
                self.attack_coordinates) == features.Visibility.VISIBLE:
            self.move_number += 1
            return actions.FUNCTIONS.move_camera(self.attack_coordinates)
        if self.move_number == 3:
            enemy_units = [unit for unit in obs.observation.feature_units
                           if unit.alliance == features.PlayerRelative.ENEMY]
            if len(enemy_units) > 0:
                enemy_unit_type = self.get_unit_type(enemy_units[0].unit_type)
                self.enemy_race = Race(enemy_unit_type.__class__) if enemy_unit_type is not None else Race.Unknown
                self.move_number += 1
                return actions.FUNCTIONS.move_camera(
                    [18, 22] if self.base_top_left else [40, 46])

        if self.move_number == 4:
            overlord = self.get_units_by_type(obs, units.Zerg.Overlord)
            if len(overlord) > 0:
                overlord = overlord[0]
                if not overlord.is_selected:
                    return actions.FUNCTIONS.select_point(actions.SelectPointAct.select.name, (overlord.x, overlord.y))
                return actions.FUNCTIONS.Stop_quick(actions.Queued.now.name)

        return actions.FUNCTIONS.no_op()


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
                        use_feature_units=True),
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
