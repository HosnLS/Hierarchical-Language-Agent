# modules for game
from gym_cooking.misc.game.game import Game
from gym_cooking.misc.game.utils import *
from gym_cooking.utils.gui import popup_text

# helpers
import pygame
import threading
import time
import queue

from copy import deepcopy as dcopy


class GamePlay(Game):
    def __init__(self, env):
        Game.__init__(self, env, play=True)
        # fps
        self.fps = 10

        self._success = False

        self._q_control = queue.Queue()
        self._q_env = queue.Queue()

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._q_control.put(('Quit', {}))

        elif event.type == pygame.KEYDOWN:
            if event.key in KeyToTuple.keys():
                # Control
                action_dict = {agent.name: (0, 0) for agent in self.sim_agents}
                action = KeyToTuple[event.key]
                action_dict[self.current_agent.name] = action
                self._q_env.put(('Action', {"agent": "1", "action": action}))

            if event.key in KeyToTuple2.keys():
                # Control
                action_dict = {agent.name: (0, 0) for agent in self.sim_agents}
                action = KeyToTuple2[event.key]
                action_dict[self.current_agent.name] = action
                self._q_env.put(('Action', {"agent": "2", "action": action}))

            if pygame.key.name(event.key) == "space":
                self._q_env.put(('Pause', {}))

                s = popup_text("Say:")

                if s is not None:
                    self._q_env.put(('ChatIn', {"chat": s, "mode": "text"}))

                self._q_env.put(('Continue', {}))

    def _run_env(self):
        seconds_per_step = 1 / self.fps
        last_t = time.time()
        action_dict = {agent.name: None for agent in self.sim_agents}
        chat = ''
        paused = False

        self.on_render(paused=paused)

        while True:
            while not self._q_env.empty():
                event = self._q_env.get_nowait()
                event_type, args = event
                if event_type == 'Action':
                    if args['agent'] == "1":
                        action_dict[self.sim_agents[0].name] = args['action']
                    elif args['agent'] == "2":
                        action_dict[self.sim_agents[1].name] = args['action']
                elif event_type == 'Pause':
                    paused += 1
                elif event_type == 'Continue':
                    paused -= 1
                elif event_type == 'ChatIn':
                    chat = args['chat']

            if not paused:
                ad = {k: v if v is not None else (
                    0, 0) for k, v in action_dict.items()}
                _, _, done, _ = self.env.step(ad, passed_time=seconds_per_step)
                if done:
                    self._success = True
                    self._q_control.put(('Quit', {}))
                    return

                action_dict = {agent.name: None for agent in self.sim_agents}

            sleep_time = max(seconds_per_step - (time.time() - last_t), 0)
            last_t = time.time()
            time.sleep(sleep_time)

            self.on_render(paused=paused, chat=chat)

    def _run_human(self):
        while True:
            for event in pygame.event.get():
                self.on_event(event)
            if not self._q_control.empty():
                event, args = self._q_control.get_nowait()
                if event == 'Quit':
                    return

    def on_execute(self):
        if self.on_init() == False:
            exit()

        thread_env = threading.Thread(target=self._run_env, daemon=True)
        thread_env.start()

        self._run_human()

        # clean up
        self.on_cleanup()

        return self._success
