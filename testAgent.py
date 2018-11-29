from pysc2.agents import base_agent
from pysc2.lib import actions

class SimpleAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.episodes = -1
        self.steps_episode = -1
        self.steps_game = -1
        self.i = 0
        self.j = 0
        self.i_max = 0
        self.j_max = 0
        print("-- __init__")

    def __del__(self):
        print("-- __del__")

    def setup(self, obs_spec, action_spec):
        self.step_cam = obs_spec['feature_screen'][0]
        self.i_max = obs_spec['feature_minimap'][1]
        self.j_max = obs_spec['feature_minimap'][2]

        self.MAX_X_SIZE = obs_spec['feature_minimap'][2]  # x
        self.MAX_Y_SIZE = obs_spec['feature_minimap'][1]  # y
        self.screen_x = self.MAX_X_SIZE
        self.screen_y = self.MAX_Y_SIZE
        # TODO find dynamic values
        self.CAM_DX = 20
        self.CAM_DY = 20

        self.nb_scan = 0

        print("obs_spec :")
        print(obs_spec)
        print("action_spec :")
        print(action_spec)

    def reset(self):
        self.steps_episode = -1
        self.episodes += 1
        print("-- -- reset")

    # def step(self, obs):
    #     self.steps_episode += 1
    #     self.steps_game += 1
    #     if((self.steps_game % 1000) == 0):
    #         print("-- -- step %d episode %d (%d)"%(self.steps_game, self.episodes, self.steps_episode))
    #     dest = [self.i, self.j]
    #     self.update_i_j()
    #     return actions.FUNCTIONS.move_camera(dest)
        # return actions.FUNCTIONS.no_op()

    # def update_i_j(self):
    #     self.i += self.step_cam
    #     if self.i >= self.i_max:
    #         self.i = 0
    #         self.j += self.step_cam
    #         if self.j >= self.j_max:
    #             self.j = 0

    def step(self, obs):
        super().step(obs)
        if self.screen_x < self.MAX_X_SIZE - self.CAM_DX:
            self.screen_x = self.screen_x + self.CAM_DX
        else:
            if self.screen_y < self.MAX_Y_SIZE - self.CAM_DY:
                self.screen_x = 12
                self.screen_y = self.screen_y + self.CAM_DY
            else:
                self.screen_x = 12
                self.screen_y = 12
                self.nb_scan = self.nb_scan + 1
        return actions.FUNCTIONS.move_camera([int(self.screen_x), int(self.screen_y)])
