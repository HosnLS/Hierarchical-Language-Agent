from gym.envs.registration import register
from . import envs, misc, recipe_planner, utils
from pathlib import Path


__all__ = [
    "envs",
    "misc",
    "recipe_planner",
    "utils",
]

register(
    id="overcookedEnv-v0",
    entry_point="gym_cooking.envs:OvercookedEnvironment",
)
