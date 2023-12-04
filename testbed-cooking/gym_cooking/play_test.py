from gym_cooking.misc.game.gameplay import GamePlay
from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked argument parser")
    parser.add_argument(
        "--map", type=str,
        choices=['ring', 'bottleneck', 'partition', 'quick'], default='ring'
    )

    return parser.parse_args()


MAP_SETTINGS = dict(
    ring=dict(level="new1",),
    bottleneck=dict(level="new3",),
    partition=dict(level="new2"),
    quick=dict(level="new5", max_num_orders=4,),
)

if __name__ == '__main__':
    arglist = parse_arguments()

    map_set = MapSetting(**MAP_SETTINGS[arglist.map])
    env = OvercookedEnvironment(map_set)
    env.reset()

    game = GamePlay(env)

    ok = game.on_execute()
    print(ok)
