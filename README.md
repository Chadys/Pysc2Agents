# demoAgent.py
Just an agent that move the camera.

# collectShardsAgent.py
Improvement over [base scripted agent from official API](https://github.com/deepmind/pysc2/blob/master/pysc2/agents/scripted_agent.py).
Instead of just going for the closest mineral, go for the closest mineral that isn't closer to the other marine.

# defeatRoachesSmartAgent.py
With the help from this [pysc2 Q learning tutorial](https://itnext.io/refine-your-sparse-pysc2-agent-a3feb189bc68).
Using the distance between the closest Marine and Roach, and the time since the last attack, learns when to attack or retreat.

# simple64ZergSparseAgent.py
With the help from this [pysc2 Q learning tutorial](https://itnext.io/refine-your-sparse-pysc2-agent-a3feb189bc68).
The state of the game for learning is defined by the buildings constructed, the number of larvae, army units and idle workers, the quarters of the map where ourselves or the enemy is present and the race of the enemy.
There are predefined actions :
- Send an Overlord on a patrol to identify the race of the enemy
- Send some workers to scout the map (the area are scouted in the order getting farther from the base) if they are idle
- Build, in that order, a Hatchery, Spawning Pool, two Extractors, a Roach Warren and a Lair when resources are sufficient
- Produce a Queen and regularly cast Spawn Larva
- Do the speed upgrade for Zerglings and Roaches
- Send six workers to harvest Vespene Gaz

Since these actions can fail (for example with a building started on top of some units), these predefined actions are only called every two steps, so as not to prevent the learned actions to influence the game.
For the learned actions, the impossible ones are excluded (like producing a roach when there is no Spawning Pool) to reduce the action space.
The actions that can be learned are :
- Do nothing
- Produce a Drone
- Produce two Zerglings
- Produce a Roach
- Produce an Overlord
- Send all army to attack one of the four quarters of the map

The goal is that the agent should learn what unit to produce and when to attack depending on who the enemy is.