from agent.mind.agent import AgentSetting
from agent.gameplay import GamePlay

from gym_cooking.utils.gui import *
from gym_cooking.utils.replay import Replay
from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.play_test import MAP_SETTINGS
from copy import deepcopy

import os
import argparse
from datetime import datetime
from pathlib import Path



def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked argument parser")

    parser.add_argument(
        "--map", type=str,
        choices=['ring', 'bottleneck', 'partition', 'quick'], default='ring'
    )
    parser.add_argument(
        "--agent", type=str,
        choices=['HLA', 'SMOA', 'FMOA', 'NEA'], default='HLA'
    )

    return parser.parse_args()


def init_env_replay(map_name, agent_name):
    map_set = MapSetting(**MAP_SETTINGS[map_name])
    agent_set = AgentSetting(agent_name, speed=2.5 if map_name != 'quick' else 3.5)
    
    replay = Replay()

    env = OvercookedEnvironment(map_set)
    env.reset()

    game = GamePlay(env, replay, agent_set)

    replay['set_map'] = deepcopy(map_set)
    replay['set_agent'] = deepcopy(agent_set)
    replay['order_rand'] = deepcopy(env.order_scheduler.rand_recipe_list)
    replay['chg_rand'] = deepcopy(env.chg_rand_list)

    return game, env, replay


if __name__ == '__main__':
    arglist = parse_arguments()

    # initialize replay
    game, env, replay = init_env_replay(arglist.map, arglist.agent)

    # play
    ok = game.on_execute()
    
    print(replay['order_result'])
    repdir = Path(__file__).resolve().parent / 'replay'
    replay.save(repdir / f'{arglist.map}-{arglist.agent}-{datetime.now().strftime("%Y%m%d_%H%M%S")}.rep')

    # record
    if ok is True:
        popup_box("Game End!")
    else:
        popup_box("Game Failed!")
