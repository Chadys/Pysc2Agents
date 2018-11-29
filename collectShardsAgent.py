# Improvement over base scripted agent in https://github.com/deepmind/pysc2/blob/master/pysc2/agents/scripted_agent.py

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
import numpy

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS

MAX = 1000000


class CollectMineralShardsFeatureUnits(base_agent.BaseAgent):
    """An agent for solving the CollectMineralShards map with feature units.
  Controls the two marines independently:
  - select marine
  - move to nearest mineral shard that wasn't the previous target and that isn't closer to the other marine
  - swap marine and repeat
  """

    def setup(self, obs_spec, action_spec):
        super(CollectMineralShardsFeatureUnits, self).setup(obs_spec, action_spec)
        if "feature_units" not in obs_spec:
            raise Exception("This agent requires the feature_units observation.")

    def reset(self):
        super(CollectMineralShardsFeatureUnits, self).reset()
        self._marine_selected = False
        self._previous_mineral_xy = [-1, -1]

    def step(self, obs):
        super(CollectMineralShardsFeatureUnits, self).step(obs)
        marines = [unit for unit in obs.observation.feature_units
                   if unit.alliance == _PLAYER_SELF]
        if len(marines) < 2:
            return FUNCTIONS.no_op()

        index_marine_to_play = marines[1].is_selected == self._marine_selected
        marine_to_play = marines[index_marine_to_play]
        other_marine = marines[not index_marine_to_play]


        if not marine_to_play.is_selected:
            # Nothing selected or the wrong marine is selected.
            self._marine_selected = True
            return FUNCTIONS.select_point("select", (marine_to_play.x, marine_to_play.y))

        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            # Find and move to the nearest mineral.
            minerals = [[unit.x, unit.y] for unit in obs.observation.feature_units
                        if unit.alliance == _PLAYER_NEUTRAL]

            if self._previous_mineral_xy in minerals:
                # Don't go for the same mineral shard as other marine.
                minerals.remove(self._previous_mineral_xy)

            if minerals:
                # Find the closest.
                distances = numpy.linalg.norm(
                    numpy.array(minerals) - numpy.array((marine_to_play.x, marine_to_play.y)), axis=1)
                distances_other = numpy.linalg.norm(
                    numpy.array(minerals) - numpy.array((other_marine.x, other_marine.y)), axis=1)
                print("DIST : ")
                print(distances)

                for index, d in numpy.ndenumerate(distances):
                    if distances_other[index] < d:
                        d += MAX
                closest_mineral_xy = minerals[numpy.argmin(distances)]

                # Swap to the other marine.
                self._marine_selected = False
                self._previous_mineral_xy = closest_mineral_xy
                return FUNCTIONS.Move_screen("now", closest_mineral_xy)

        return FUNCTIONS.no_op()
