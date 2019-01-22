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
    hot_squares0 = 9
    hot_squares1 = 10
    hot_squares2 = 11
    hot_squares3 = 12
    green_square0 = 13
    green_square1 = 14
    green_square2 = 15
    green_square3 = 16


class Race(enum.Enum):
    Unknown = 0
    Terran = 1
    Zerg = 2
    Protoss = 3


race_dict = {units.Terran: Race.Terran, units.Zerg: Race.Zerg, units.Protoss: Race.Protoss, None: Race.Unknown}


class Action(enum.IntEnum):
    DO_NOTHING = 0
    TRAIN_DRONE = 1
    TRAIN_ZERGLING = 2
    TRAIN_ROACH = 3
    TRAIN_OVERLORD = 4
    ATTACK = 5
    BUILD_HATCHERY = 9
    BUILD_SPAWNING_POOL = 10
    TRAIN_QUEEN = 11
    BUILD_EXTRACTOR = 12
    BUILD_ROACH_WARREN = 13
    BUILD_LAIR = 14
    UPGRADE_ZERGLINGS = 15
    UPGRADE_ROACHES = 16
    SPAWN_LARVA = 17
    SCOUT_ENEMY = 18
    IDENTIFY_ENEMY = 19
    STOP_SCOUTING = 20
    HARVEST_GAZ = 21


class PatrolState(enum.Enum):
    NOPE = 0
    ENROUTE = 1
    RETURNING = 2


smart_actions = [Action.DO_NOTHING.name, Action.TRAIN_DRONE.name, Action.TRAIN_ZERGLING.name,
                 Action.TRAIN_ROACH.name, Action.TRAIN_OVERLORD.name]

for mm_x in range(MINIMAP_SIZE // 4, MINIMAP_SIZE, MINIMAP_SIZE // 2):
    for mm_y in range(MINIMAP_SIZE // 4, MINIMAP_SIZE, MINIMAP_SIZE // 2):
        smart_actions.append('{},{},{}'.format(Action.ATTACK.name, mm_x, mm_y))


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
        self.base_coord = [0, 0]

    def transform_distance(self, x, x_distance, y, y_distance):
        return [x - x_distance, y - y_distance] if not self.base_top_left else [x + x_distance, y + y_distance]

    def transform_location(self, x, y):
        if not self.base_top_left:
            return [MINIMAP_SIZE - x, MINIMAP_SIZE - y]

        return [x, y]

    def parse_action(self, action_id):
        smart_action = smart_actions[action_id] if not self.hardcoded_action else action_id.name

        x = 0
        y = 0
        if ',' in smart_action:
            smart_action, x, y = smart_action.split(',')

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

    def process_obs(self, obs, display=False):
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
        print('action_result')
        print(obs.observation.action_result)
        print('last_actions')
        print(obs.observation.last_actions)
        print('single_select')
        print(obs.observation.single_select)
        print('multi_select')
        print(obs.observation.multi_select)
        hatchery = self.get_units_by_type(obs, units.Zerg.Hatchery)
        if len(hatchery):
            print('hatchery')
            print('{}, {}, rad : {}'.format(hatchery[0].x, hatchery[0].y, hatchery[0].radius))
        creep_min = [83, 83]
        creep_max = [0, 0]
        for creep_x, creep_y in zip(*obs.observation.feature_screen.creep.nonzero()):
            creep_min[0] = min(creep_min[0], creep_x)
            creep_min[1] = min(creep_min[1], creep_y)
            creep_max[0] = max(creep_max[0], creep_x)
            creep_max[1] = max(creep_max[1], creep_y)
        print('creep')
        print('{}, {}'.format(creep_min, creep_max))

    def get_alliance_squares(self, obs, alliance):
        hot_squares = [0] * 4
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
        ], names=State)

        return current_state

    def get_excluded_actions(self, current_state, obs):
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)

        excluded_actions = set()
        # actions excluded because of space
        if not free_supply:
            excluded_actions.add(Action.TRAIN_DRONE.value)
            excluded_actions.add(Action.TRAIN_ZERGLING.value)
            excluded_actions.add(Action.TRAIN_ROACH.value)
        elif free_supply < 2:
            excluded_actions.add(Action.TRAIN_ROACH.value)

        # actions excluded because of army
        if not obs.observation.player.army_count:
            for i in range(Action.ATTACK.value, Action.ATTACK.value + 4):
                excluded_actions.add(i)

        # actions excluded because of resources
        if obs.observation.player.minerals < 25:
            excluded_actions.add(Action.TRAIN_ZERGLING.value)
            excluded_actions.add(Action.TRAIN_DRONE.value)
            excluded_actions.add(Action.TRAIN_ROACH.value)
            excluded_actions.add(Action.TRAIN_OVERLORD.value)
        elif obs.observation.player.minerals < 50:
            excluded_actions.add(Action.TRAIN_DRONE.value)
            excluded_actions.add(Action.TRAIN_ROACH.value)
            excluded_actions.add(Action.TRAIN_OVERLORD.value)
        elif obs.observation.player.minerals < 75:
            excluded_actions.add(Action.TRAIN_ROACH.value)
            excluded_actions.add(Action.TRAIN_OVERLORD.value)
        elif obs.observation.player.minerals < 100:
            excluded_actions.add(Action.TRAIN_OVERLORD.value)
        elif obs.observation.player.vespene < 25:
            excluded_actions.add(Action.TRAIN_ROACH.value)

        # actions excluded because of larvae
        # num_larvae = obs.observation.player.larva_count  # can't use, always zero
        if not current_state.num_larvae:
            excluded_actions.add(Action.TRAIN_ZERGLING.value)
            excluded_actions.add(Action.TRAIN_DRONE.value)
            excluded_actions.add(Action.TRAIN_ROACH.value)
            excluded_actions.add(Action.TRAIN_OVERLORD.value)

        # actions excluded because of building requirement
        if not current_state.num_spawning_pool:
            excluded_actions.add(Action.TRAIN_ZERGLING.value)
        if not current_state.num_roach_warren:
            excluded_actions.add(Action.TRAIN_ROACH.value)

        return excluded_actions

    def get_hardcoded_action(self, obs, current_state):
        if self.enemy_race == Race.Unknown:
            if self.patrol_state == PatrolState.NOPE:
                return Action.SCOUT_ENEMY
            # else self.patrol_state == PatrolState.ENROUTE
            if obs.observation.feature_minimap.visibility_map.item(
                    self.attack_coordinates) == features.Visibility.VISIBLE:
                return Action.IDENTIFY_ENEMY
        if self.patrol_state == PatrolState.RETURNING:
            overlord = [o for o in self.get_units_by_type(obs, units.Zerg.Overlord) if o.order_length > 0]
            if len(overlord) > 0:
                return Action.STOP_SCOUTING

        if not current_state.num_base:
            if obs.observation.player.minerals >= 300 and obs.observation.player.idle_worker_count > 0:
                return Action.BUILD_HATCHERY
            return None
        if not current_state.num_spawning_pool:
            if obs.observation.player.minerals >= 200 and obs.observation.player.idle_worker_count > 0:
                return Action.BUILD_SPAWNING_POOL
            return None
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        if not current_state.num_queen:
            if obs.observation.player.minerals >= 150 and free_supply >= 2 and current_state.num_larvae > 0:
                return Action.TRAIN_QUEEN
            return None
        queen = self.get_units_by_type(obs, units.Zerg.Queen)
        if len(queen) > 0:
            queen = queen[0]
            if queen.energy >= 25:
                return Action.SPAWN_LARVA
        if not self.upgrade_zergling:
            if obs.observation.player.minerals >= 100 and obs.observation.player.vespene >= 100:
                return Action.UPGRADE_ZERGLINGS
        if current_state.num_extractor < 2:
            if obs.observation.player.minerals >= 25 and obs.observation.player.idle_worker_count > 0:
                return Action.BUILD_EXTRACTOR
            return None
        extractor = [e for e in self.get_units_by_type(obs, units.Zerg.Extractor) if e.assigned_harvesters < 3]
        if len(extractor) > 0 and obs.observation.player.idle_worker_count > 0:
            return Action.HARVEST_GAZ.name
        if not current_state.num_roach_warren:
            if obs.observation.player.minerals >= 150 and obs.observation.player.idle_worker_count > 0:
                return Action.BUILD_ROACH_WARREN
            return None
        if current_state.num_base < 2:
            if obs.observation.player.minerals >= 150 and obs.observation.player.vespene >= 100:
                return Action.BUILD_LAIR
            return None
        if not self.upgrade_roaches:
            if obs.observation.player.minerals >= 100 and obs.observation.player.vespene >= 100:
                return Action.UPGRADE_ROACHES
        return None

    def step(self, obs):
        super().step(obs)

        self.process_obs(obs, True)
        action = self.get_default_action()

        if obs.first():
            self.init_starting_pos(obs)

        if obs.last():
            reward = obs.reward
            if self.hardcoded_action:
                self.previous_action = 0
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
            return actions.FUNCTIONS.no_op()

        if self.move_number == 0:
            current_state = self.get_current_state(obs)
            print('state : {}'.format(current_state))

            if self.previous_action is not None and not self.hardcoded_action:
                self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))

            rl_action = self.get_hardcoded_action(obs, current_state)
            if rl_action is not None:
                rl_action = rl_action
                self.hardcoded_action = True
            else:
                rl_action = self.qlearn.choose_action(str(current_state), self.get_excluded_actions(current_state, obs))
                self.hardcoded_action = False

            self.previous_state = current_state
            self.previous_action = rl_action

            action = self.get_first_action(obs)

        elif self.move_number == 1:
            action = self.get_second_action(obs)

        print('chosen action : {}'.format(action))
        return action

    @staticmethod
    def get_default_action():
        actions.FUNCTIONS.no_op()

    def get_first_action(self, obs):
        smart_action, x, y = self.parse_action(self.previous_action)
        print('smart_action : {}'.format(smart_action))
        if smart_action == Action.DO_NOTHING.name:
            return actions.FUNCTIONS.no_op()
        self.move_number += 1
        if smart_action == Action.TRAIN_DRONE.name or smart_action == Action.TRAIN_ZERGLING.name or \
                smart_action == Action.TRAIN_ROACH.name or smart_action == Action.TRAIN_OVERLORD.name or \
                smart_action == Action.TRAIN_QUEEN.name:
            # return actions.FUNCTIONS.select_larva()  # doesn't works because obs.observation.player.larva_count is always 0
            larvae = self.get_units_by_type(obs, units.Zerg.Larva)
            if len(larvae) > 0:
                larva = random.choice(larvae)
                return actions.FUNCTIONS.select_point(actions.SelectPointAct.select.name, (larva.x, larva.y))
        elif smart_action == Action.ATTACK.name:
            return actions.FUNCTIONS.select_army(actions.SelectAdd.select.name)
        elif smart_action == Action.BUILD_HATCHERY.name or smart_action == Action.BUILD_SPAWNING_POOL.name or \
                smart_action == Action.BUILD_EXTRACTOR.name or smart_action == Action.BUILD_ROACH_WARREN.name or \
                smart_action == Action.HARVEST_GAZ.name:
            return actions.FUNCTIONS.select_idle_worker(actions.SelectWorker.select.name)
        elif smart_action == Action.BUILD_LAIR.name:
            if self.base_coord[0] > 0:
                return actions.FUNCTIONS.select_point(actions.SelectPointAct.select.name, self.base_coord)
            hatchery = self.get_units_by_type(obs, units.Zerg.Hatchery)
            if len(hatchery) > 0:
                hatchery = hatchery[0]
                self.base_coord = [hatchery.x, hatchery.y]
                return actions.FUNCTIONS.select_point(actions.SelectPointAct.select.name, self.base_coord)
        elif smart_action == Action.UPGRADE_ZERGLINGS.name:
            s_pool = self.get_units_by_type(obs, units.Zerg.SpawningPool)
            if len(s_pool) > 0:
                s_pool = s_pool[0]
                return actions.FUNCTIONS.select_point(actions.SelectPointAct.select.name, (s_pool.x, s_pool.y))
        elif smart_action == Action.UPGRADE_ROACHES.name:
            r_warren = self.get_units_by_type(obs, units.Zerg.RoachWarren)
            if len(r_warren) > 0:
                r_warren = r_warren[0]
                return actions.FUNCTIONS.select_point(actions.SelectPointAct.select.name, (r_warren.x, r_warren.y))
        elif smart_action == Action.SPAWN_LARVA.name:
            queen = self.get_units_by_type(obs, units.Zerg.Queen)
            if len(queen) > 0:
                queen = queen[0]
                return actions.FUNCTIONS.select_point(actions.SelectPointAct.select.name, (queen.x, queen.y))
        elif smart_action == Action.SCOUT_ENEMY.name:
            overlord = self.get_units_by_type(obs, units.Zerg.Overlord)
            if len(overlord) > 0:
                overlord = overlord[0]
                return actions.FUNCTIONS.select_point(actions.SelectPointAct.select.name, (overlord.x, overlord.y))
        elif smart_action == Action.IDENTIFY_ENEMY.name:
            return actions.FUNCTIONS.move_camera(self.attack_coordinates)
        elif smart_action == Action.STOP_SCOUTING.name:
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
        smart_action, x, y = self.parse_action(self.previous_action)
        self.move_number = 0

        if smart_action == Action.TRAIN_DRONE.name:
            return actions.FUNCTIONS.Train_Drone_quick(actions.Queued.now.name)
        elif smart_action == Action.TRAIN_ZERGLING.name:
            return actions.FUNCTIONS.Train_Zergling_quick(actions.Queued.now.name)
        elif smart_action == Action.TRAIN_ROACH.name:
            return actions.FUNCTIONS.Morph_Lair_quick(actions.Queued.now.name)
        elif smart_action == Action.TRAIN_OVERLORD.name:
            return actions.FUNCTIONS.Train_Overlord_quick(actions.Queued.now.name)
        elif smart_action == Action.TRAIN_QUEEN.name:
            return actions.FUNCTIONS.Train_Queen_quick(actions.Queued.now.name)
        elif smart_action == Action.ATTACK.name:
            x_offset = random.randint(-1, 1)
            y_offset = random.randint(-1, 1)
            return actions.FUNCTIONS.Attack_minimap(actions.Queued.now.name,
                                                    self.transform_location(int(x) + x_offset * 8,
                                                                            int(y) + y_offset * 8))
        elif smart_action == Action.HARVEST_GAZ.name:
            extractor = [e for e in self.get_units_by_type(obs, units.Zerg.Extractor) if e.assigned_harvesters < 3][0]
            return actions.FUNCTIONS.Harvest_Gather_Drone_screen(actions.Queued.now.name, (extractor.x, extractor.y))

        # TODO choose coordinates
        elif smart_action == Action.BUILD_HATCHERY.name:
            return actions.FUNCTIONS.Build_Hatchery_screen(actions.Queued.now.name, self.base_coord)
        elif smart_action == Action.BUILD_SPAWNING_POOL.name:
            return actions.FUNCTIONS.Build_SpawningPool_screen(actions.Queued.now.name,
                                                               self.transform_distance(self.base_coord[0], 18,
                                                                                       self.base_coord[1], 0))
        elif smart_action == Action.BUILD_ROACH_WARREN.name:
            return actions.FUNCTIONS.Build_RoachWarren_screen(actions.Queued.now.name,
                                                              self.transform_distance(self.base_coord[0], 0,
                                                                                      self.base_coord[1], 18))

        elif smart_action == Action.BUILD_EXTRACTOR.name:
            vespene = self.get_units_by_type(obs, units.Neutral.VespeneGeyser)
            if len(vespene) > 0:
                vespene = vespene[0]
                return actions.FUNCTIONS.Build_Extractor_screen(actions.Queued.now.name, (vespene.x, vespene.y))
        elif smart_action == Action.BUILD_LAIR.name:
            return actions.FUNCTIONS.Morph_Lair_quick(actions.Queued.now.name)
        elif smart_action == Action.UPGRADE_ZERGLINGS.name:
            return actions.FUNCTIONS.Research_ZerglingMetabolicBoost_quick(actions.Queued.now.name)
        elif smart_action == Action.UPGRADE_ROACHES.name:
            return actions.FUNCTIONS.Research_GlialRegeneration_quick(actions.Queued.now.name)
        elif smart_action == Action.SPAWN_LARVA.name:
            hatchery = self.get_units_by_type(obs, units.Zerg.Hatchery)
            if len(hatchery) > 0:
                hatchery = hatchery[0]
                return actions.FUNCTIONS.Effect_InjectLarva_screen(actions.Queued.now.name, (hatchery.x, hatchery.y))
        elif smart_action == Action.SCOUT_ENEMY.name:
            self.patrol_state = PatrolState.ENROUTE
            return actions.FUNCTIONS.Patrol_minimap(actions.Queued.now.name, self.attack_coordinates)
        elif smart_action == Action.IDENTIFY_ENEMY.name:
            enemy_units = [unit for unit in obs.observation.feature_units
                           if unit.alliance == features.PlayerRelative.ENEMY]
            if len(enemy_units) > 0:
                self.patrol_state = PatrolState.RETURNING
                enemy_unit_type = self.get_unit_type(enemy_units[0].unit_type)
                self.enemy_race = race_dict[enemy_unit_type.__class__ if enemy_unit_type is not None \
                    else enemy_unit_type]
                return actions.FUNCTIONS.move_camera([18, 22] if self.base_top_left else [40, 46])
        elif smart_action == Action.STOP_SCOUTING.name:
            self.patrol_state = PatrolState.NOPE
            hatchery = self.get_units_by_type(obs, units.Zerg.Hatchery)
            if len(hatchery) > 0:
                self.base_coord = [hatchery[0].x, hatchery[0].y]
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
