import copy
import time
import pygame
from gym_cooking.utils.replay import Replay
from gym_cooking.misc.game.game import Game

class GamePlayReply(Game):
    def __init__(self, env, replay: Replay):
        Game.__init__(self, env, play=True)
        self.env = env
        self.replay = replay

        self.speedup = 2.0

    def on_execute(self):
        if self.on_init() == False:
            exit()

        # simulate
        last_t = time.time()
        self.on_render(paused=False, replay=True)

        for record in self.replay:
            if record['name'] == 'env.step':
                sleep_time = max(record['args']['passed_time'] / self.speedup - (time.time() - last_t), 0)
                last_t = time.time()
                time.sleep(sleep_time)

                ret = self.env.step(**record['args'])

            elif record['name'] == 'on_render':
                args = copy.deepcopy(record['args'])
                self.on_render(replay=True, **args)
            
            else:
                print(record['time'], " :", record['name'])

            for event in pygame.event.get():
                pass

        # clean up
        self.on_cleanup()
