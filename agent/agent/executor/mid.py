from collections import namedtuple, defaultdict
from copy import copy, deepcopy
import random

from gym_cooking.utils.core import GridSquare, Object, mergeable, \
    GRIDSQUARES, PUTTABLE_GRIDSQUARES, FOOD_TILE, \
    FRESH_FOOD, CHOPPING_FOOD, CHOPPED_FOOD, COOKING_FOOD, COOKED_FOOD, CHARRED_FOOD, \
    ASSEMBLE_CHOPPED_FOOD, ASSEMBLE_CHOPPED_PLATE_FOOD, ASSEMBLE_COOKING_FOOD, ASSEMBLE_COOKING_PLATE_FOOD, \
    ASSEMBLE_COOKED_FOOD, ASSEMBLE_COOKED_PLATE_FOOD, ASSEMBLE_CHARRED_FOOD, ASSEMBLE_CHARRED_PLATE_FOOD
from gym_cooking.utils.world import World
from gym_cooking.utils.agent import SimAgent
from gym_cooking.utils.order_schedule import OrderScheduler
from gym_cooking.utils.event import Event

from agent.executor.low import LTApproach, LTInteract, EnvState, match_any, fname

# Mid Task: doing simple jobs like cook, chop, putout, etc.
#     input: the variable of the task
#     prep: the condition of the task
#     done: the state of the environment after the task
#     error: possible error of the task

class MidTask:
    Success = 1
    Working = 0
    Failed = -1

    def can_begin(self, env: EnvState):
        return True

    def __call__(self, env: EnvState):
        ret = MTChop.Success, (0, 0), f"Init"
        return ret


# input: obj: chopping food
# prep: hold nothing
#       Cutboard with chopping food
# done: Cutboard with chopped food
# error: no match pos, no match hold, chop complete on approaching
class MTChop(MidTask):
    # cook
    # 1 seek
    #   prep: no reachable cutboard + chopping   (stop)
    #         holding something             (stop)
    #   ret:                                (goto 2)
    # 2 approach
    #   prep: pos is not cutboard + chopping     (goto 1)
    #         holding something             (stop)
    #   ret:  working                       act
    #         block                         act, (goto 1)
    #         unreachable                   (goto 1)
    #         done                          (goto 3)
    # 3 interact: chop
    #   prep: holding something                     (goto 1)
    #         pos is cutboard + chopped             success
    #         pos is not cutboard + chopping        fail
    #   error: toofar                               (goto 2)
    #          success                              act
    def __init__(self, obj: str = None):
        self.OBJ = obj

        self._task = None
        self._stage = 1
        self._pos = None

    def can_begin(self, env: EnvState):
        if self.OBJ is None:
            self._task = CHOPPING_FOOD
        else:
            if self.OBJ not in CHOPPING_FOOD: return False
            self._task = self.OBJ
        pos = env.navigate_pos_by_obj_gs(obj=self._task, gs='Cutboard')
        if pos is None:
            return False
        if not match_any(fname(env.hold), "Nothing"):
            return False
        return True

    def __call__(self, env: EnvState):
        if self._task is None and not self.can_begin(env):
            return self.Failed, (0, 0), f"Can't perform chop"

        while True:
            if self._stage == 1:
                pos = env.navigate_pos_by_obj_gs(obj=self._task, gs='Cutboard')
                if pos is None:
                    return MTChop.Failed, (0, 0), "No reachable Cutboard"
                if not match_any(fname(env.hold), "Nothing"):
                    return MTChop.Failed, (0, 0), f"Holding {fname(env.hold)}"
                self._pos = pos
                self._stage = 2
            elif self._stage == 2:
                if not env.check_pos_by_obj_gs(pos=self._pos, obj=self._task, gs='Cutboard'):
                    self._stage = 1
                    continue
                status, move = LTApproach(self._pos)(env)
                if status == LTApproach.Working:
                    return MTChop.Working, move, "Approaching"
                elif status == LTApproach.DestBlock:
                    self._stage = 1
                    return MTChop.Working, move, "Blocked by other agent"
                elif status == LTApproach.DestUnreachable:
                    self._stage = 1
                    continue
                elif status == LTApproach.Success:
                    self._stage = 3
                    continue
            elif self._stage == 3:
                if env.check_pos_by_obj_gs(self._pos, obj=CHOPPED_FOOD, gs='Cutboard'):
                    return MTChop.Success, (0, 0), f"Successfully chop"
                if not env.check_pos_by_obj_gs(self._pos, obj=self._task, gs='Cutboard'):
                    return MTChop.Failed, (0, 0), f"No food being chopped on Cutboard"
                status, move = LTInteract(self._pos)(env)
                if status == LTInteract.DestTooFar:
                    self._stage = 2
                    continue
                elif status == LTInteract.Success:
                    return MTChop.Working, move, "Chopping"
        return MTChop.Failed, (0, 0), f"Unknown error"


# prep: hold fire extinguisher
#       Pot with fire
# done: Pot without fire
class MTPutout(MidTask):
    # cook
    # 1 seek
    #   prep: no reachable fire             stop
    #         not holding extinguisher      stop
    #   ret:                                (goto 2)
    # 2 approach
    #   prep: pos is not fire               (goto 1)
    #         not holding extinguisher      stop
    #   ret:  working                       act
    #         block                         act, (goto 1)
    #         unreachable                   (goto 1)
    #         done                          (goto 3)
    # 3 interact-1: putout
    #   prep: pos is not fire               success
    #         not holding extinguisher      stop
    #   ret:  toofar                        (goto 2)
    #         success                       act
    def __init__(self):
        self.stage = 1
        self.pos = None

    def can_begin(self, env: EnvState):
        pos = env.navigate_pos_by_obj_gs(obj='Fire', gs=None, inner=True)
        if pos is None:
            return False
        if not match_any(fname(env.hold), "FireExtinguisher"):
            return False
        return True

    def __call__(self, env: EnvState):
        while True:
            if self.stage == 1:
                pos = env.navigate_pos_by_obj_gs(obj='Fire', gs=None, inner=True)
                if pos is None:
                    return MTPutout.Failed, (0, 0), "No reachable fire"
                if not match_any(fname(env.hold), "FireExtinguisher"):
                    return MTPutout.Failed, (0, 0), f"Not holding FireExtinguisher"
                self.pos = pos
                self.stage = 2
            elif self.stage == 2:
                if not env.check_pos_by_obj_gs(self.pos, obj='Fire', gs=None, inner=True):
                    self.stage = 1
                    continue
                status, move = LTApproach(self.pos)(env)
                if status == LTApproach.Working:
                    return MTPutout.Working, move, "Approaching"
                elif status == LTApproach.DestBlock:
                    self.stage = 1
                    return MTPutout.Working, move, "Blocked by other agent"
                elif status == LTApproach.DestUnreachable:
                    self.stage = 1
                    continue
                elif status == LTApproach.Success:
                    self.stage = 3
                    continue
            elif self.stage == 3:
                if not env.check_pos_by_obj_gs(self.pos, obj='Fire', gs=None, inner=True):
                    return MTPutout.Success, (0, 0), f"Successfully putout fire"
                status, move = LTInteract(self.pos)(env)
                if status == LTInteract.DestTooFar:
                    self.stage = 2
                    continue
                elif status == LTInteract.Success:
                    return MTPutout.Working, move, "Interacting"


# error: no match pos, no match holding, result wrong
# Note: because many tasks share a common procedure, such as puck, put, etc. (they all require an one-time interaction)
#       We define a common task here. And the specific task is defined based on it.
class MTDoOnce(MidTask):
    # 1 seek
    #   prep: no reachable empty prep_targ  (stop)
    #         not holding prep_hold         (stop)
    #   ret:                                (goto 2)
    # 2 approach
    #   prep: pos is not empty prep_targ    (goto 1)
    #         not holding prep_hold         (stop)
    #   ret:  working                       act
    #         block                         act, (goto 1)
    #         unreachable                   (goto 1)
    #         done                          (goto 3)
    # 3 interact-1: put
    #   prep: pos is not empty prep_targ    (goto 1)
    #         not holding prep_hold         (stop)
    #   ret:  toofar                        (goto 2)
    #         success                       act, (goto 4)
    # 4 judge
    #   prep: holding after_hold and pos is after_targ  success
    #         else                                      fail
    AvailTask = [
        {
            "prep": {
                "targ": {
                    "obj": "None",
                    "gs": PUTTABLE_GRIDSQUARES
                },
                "hold": "None"
            },
            "after": {
                "targ": {
                    "obj": "None",
                    "gs": PUTTABLE_GRIDSQUARES
                },
                "hold": "None"
            }
        },
    ]
    Desc = {"n": "interaction", "v": "interact", "adj": "interactable"}

    def __init__(self, gs: str = None, obj: str = None):
        self.PREP_GS = gs
        self.PREP_OBJ = obj

        self._task = None
        self._stage = 1
        self._pos = None

    def can_begin(self, env: EnvState) -> bool:
        for task in self.AvailTask:
            task = deepcopy(task)
            # narrow down condition
            if self.PREP_GS is not None:
                if not match_any(self.PREP_GS, task['prep']['targ']['gs']): continue
                task['prep']['targ']['gs'] = self.PREP_GS
            if self.PREP_OBJ is not None:
                if not match_any(self.PREP_OBJ, task['prep']['targ']['obj']): continue
                task['prep']['targ']['obj'] = self.PREP_OBJ
            # match
            pos = env.navigate_pos_by_obj_gs(obj=task['prep']['targ']['obj'], gs=task['prep']['targ']['gs'])
            if pos is None:
                continue
            if not match_any(fname(env.hold), task['prep']['hold']):
                continue
            if not mergeable(env.hold, env.pos_obj[pos]):
                continue
            self._task = task
            break
        return self._task is not None

    def __call__(self, env: EnvState):
        if self._task is None and not self.can_begin(env):
            return self.Failed, (0, 0), f"Can't perform {self.Desc['n']}"

        while True:
            if self._stage == 1:
                pos = env.navigate_pos_by_obj_gs(obj=self._task['prep']['targ']['obj'],
                                                 gs=self._task['prep']['targ']['gs'])
                if pos is None:
                    msg = f"No matching {self.Desc['n']} point"
                    if self.PREP_GS is not None: msg += f" on {self.PREP_GS}"
                    if self.PREP_OBJ is not None: msg += f" with {self.PREP_OBJ}"
                    return self.Failed, (0, 0), f"No matching {self.Desc['n']} point"
                if not match_any(fname(env.hold), self._task['prep']['hold']):
                    return self.Failed, (0, 0), f"{fname(env.hold)} is not {self.Desc['adj']}"
                self._pos = pos
                self._stage = 2
            elif self._stage == 2:
                if not env.check_pos_by_obj_gs(self._pos, obj=self._task['prep']['targ']['obj'],
                                               gs=self._task['prep']['targ']['gs']):
                    self._stage = 1
                    continue
                status, move = LTApproach(self._pos)(env)
                if status == LTApproach.Working:
                    return self.Working, move, "Approaching"
                elif status == LTApproach.DestBlock:
                    self._stage = 1
                    return self.Working, move, "Blocked by other agent"
                elif status == LTApproach.DestUnreachable:
                    self._stage = 1
                    continue
                elif status == LTApproach.Success:
                    self._stage = 3
                    continue
            elif self._stage == 3:
                if not env.check_pos_by_obj_gs(self._pos, obj=self._task['prep']['targ']['obj'],
                                               gs=self._task['prep']['targ']['gs']):
                    self._stage = 1
                    continue
                status, move = LTInteract(self._pos)(env)
                if status == LTInteract.DestTooFar:
                    self._stage = 2
                    continue
                elif status == LTInteract.Success:
                    self._stage = 4
                    return self.Working, move, "Interacting"
            elif self._stage == 4:
                if match_any(fname(env.hold), self._task['after']['hold']) \
                        and env.check_pos_by_obj_gs(self._pos, obj=self._task['after']['targ']['obj'],
                                                    gs=self._task['after']['targ']['gs']):
                    return self.Success, (0, 0), f"Successfully {self.Desc['v']}"
                else:
                    return self.Failed, (0, 0), f"Can't carry out {self.Desc['v']} near {self.PREP_GS}"
        return self.Failed, (0, 0), f"Unknown error"


# prep: hold somgthing
# done: hold nothing
class MTPut(MTDoOnce):
    AvailTask = [
        {  # hold something, put on empty counter
            "prep": {
                "targ": {
                    "obj": "Nothing",
                    "gs": "Counter"
                },
                "hold": FRESH_FOOD + ASSEMBLE_CHOPPED_FOOD + ASSEMBLE_CHOPPED_PLATE_FOOD +
                        ASSEMBLE_COOKED_PLATE_FOOD + ASSEMBLE_CHARRED_PLATE_FOOD + ['FireExtinguisher', 'Plate']
            },
            "after": {
                "targ": {
                    "obj": FRESH_FOOD + ASSEMBLE_CHOPPED_FOOD + ASSEMBLE_CHOPPED_PLATE_FOOD +
                           ASSEMBLE_COOKED_PLATE_FOOD + ASSEMBLE_CHARRED_PLATE_FOOD + ['FireExtinguisher', 'Plate'],
                    "gs": PUTTABLE_GRIDSQUARES
                },
                "hold": "Nothing"
            }
        },
        {  # hold fresh food, put on empty cutboard
            "prep": {
                "targ": {
                    "obj": "Nothing",
                    "gs": "Cutboard"
                },
                "hold": FRESH_FOOD + ASSEMBLE_CHOPPED_FOOD
            },
            "after": {
                "targ": {
                    "obj": CHOPPING_FOOD + ASSEMBLE_CHOPPED_FOOD,
                    "gs": PUTTABLE_GRIDSQUARES
                },
                "hold": "Nothing"
            }
        },
        {  # hold plate, put on plate tile
            "prep": {
                "targ": {
                    "obj": "Nothing",
                    "gs": "PlateTile"
                },
                "hold": "Plate"
            },
            "after": {
                "targ": {
                    "obj": "Nothing",
                    "gs": "PlateTile"
                },
                "hold": "Nothing"
            }
        },
    ]
    Desc = {"n": "put", "v": "put", "adj": "puttable"}

    def __init__(self, gs: str = None):
        super().__init__(gs=gs)


# prep: hold somgthing
# done: hold nothing
class MTDrop(MTDoOnce):
    AvailTask = [
        {
            "prep": {
                "targ": {
                    "obj": "Nothing",
                    "gs": "Bin"
                },
                "hold": FRESH_FOOD
                        + ASSEMBLE_CHOPPED_FOOD + ASSEMBLE_CHOPPED_PLATE_FOOD
                        + ASSEMBLE_COOKED_PLATE_FOOD
                        + ASSEMBLE_CHARRED_PLATE_FOOD
            },
            "after": {
                "targ": {
                    "obj": "Nothing",
                    "gs": "Bin"
                },
                "hold": ["Nothing", "Plate"]
            }
        },
    ]
    Desc = {"n": "drop", "v": "drop", "adj": "droppable"}

    def __init__(self):
        super().__init__()


# prep: hold somgthing
# done: hold nothing/plate
class MTDeliver(MTDoOnce):
    AvailTask = [
        {
            "prep": {
                "targ": {
                    "obj": "Nothing",
                    "gs": "Delivery"
                },
                "hold": ASSEMBLE_CHOPPED_PLATE_FOOD + ASSEMBLE_COOKED_PLATE_FOOD
            },
            "after": {
                "targ": {
                    "obj": "Nothing",
                    "gs": "Delivery"
                },
                "hold": ["Nothing", "Plate"]
            }
        },
    ]
    Desc = {"n": "delivery", "v": "deliver", "adj": "deliverable"}

    def __init__(self):
        super().__init__()


# prep: hold somgthing
# done: hold nothing/plate
class MTCook(MTDoOnce):
    AvailTask = [
        {
            "prep": {
                "targ": {
                    "obj": "Nothing",
                    "gs": "Pot"
                },
                "hold": ASSEMBLE_CHOPPED_FOOD + ASSEMBLE_CHOPPED_PLATE_FOOD
            },
            "after": {
                "targ": {
                    "obj": ASSEMBLE_COOKING_FOOD + ASSEMBLE_COOKED_FOOD,
                    "gs": "Pot"
                },
                "hold": ["Nothing", "Plate"]
            }
        },
    ]
    Desc = {"n": "cook", "v": "cook", "adj": "cookable"}

    def __init__(self):
        super().__init__()


# prep: hold somgthing
# done: hold nothing
# a special case is not considered: hold(plate + veg1) + (plate + veg2) -> hold(plate) + (plate + veg1 + veg2)
class MTAssemble(MTDoOnce):
    # not holding plate
    AvailTask = [
        {  # hold chopped food, assemble from puttable gridsquare
            "prep": {
                "targ": {
                    "obj": ASSEMBLE_CHOPPED_FOOD + ASSEMBLE_CHOPPED_PLATE_FOOD + ["Plate"],
                    "gs": PUTTABLE_GRIDSQUARES
                },
                "hold": ASSEMBLE_CHOPPED_FOOD
            },
            "after": {
                "targ": {
                    "obj": ASSEMBLE_CHOPPED_FOOD + ASSEMBLE_CHOPPED_PLATE_FOOD,
                    "gs": PUTTABLE_GRIDSQUARES
                },
                "hold": "Nothing"
            }
        },
    ]
    Desc = {"n": "assemble", "v": "assemble", "adj": "assemblable"}

    def __init__(self, obj: str, gs: str = None):
        super().__init__(obj=obj, gs=gs)


# prep: hold nothing/plate
# done: hold something
# special:  hold(chopped) + platetile -> hold(chopped + plate) is belonged to 'pick' task
#           hold() + tile -> hold(something). the arg obj is None or "Nothing"
class MTPick(MTDoOnce):
    AvailTask = [
        {  # hold nothing, pick from food/plate tile
            "prep": {
                "targ": {
                    "obj": "Nothing",
                    "gs": FOOD_TILE + ['PlateTile']
                },
                "hold": "Nothing"
            },
            "after": {
                "targ": {
                    "obj": "Nothing",
                    "gs": FOOD_TILE + ['PlateTile']
                },
                "hold": FRESH_FOOD + ['Plate']
            }
        },
        {  # hold nothing, pick from puttable gridsquare
            "prep": {
                "targ": {
                    "obj": FRESH_FOOD + ASSEMBLE_CHOPPED_FOOD + ASSEMBLE_CHOPPED_PLATE_FOOD
                           + ASSEMBLE_COOKED_PLATE_FOOD + ASSEMBLE_CHARRED_PLATE_FOOD + ['FireExtinguisher', 'Plate'],
                    "gs": PUTTABLE_GRIDSQUARES
                },
                "hold": "Nothing"
            },
            "after": {
                "targ": {
                    "obj": "Nothing",
                    "gs": PUTTABLE_GRIDSQUARES
                },
                "hold": FRESH_FOOD + ASSEMBLE_CHOPPED_FOOD + ASSEMBLE_CHOPPED_PLATE_FOOD
                        + ASSEMBLE_COOKED_PLATE_FOOD + ASSEMBLE_CHARRED_PLATE_FOOD + ['FireExtinguisher', 'Plate']
            }
        },
        {  # hold plate, pick from pot
            "prep": {
                "targ": {
                    "obj": ASSEMBLE_COOKING_FOOD + ASSEMBLE_COOKED_FOOD + ASSEMBLE_CHARRED_FOOD,
                    "gs": "Pot"
                },
                "hold": "Plate"
            },
            "after": {
                "targ": {
                    "obj": "Nothing",
                    "gs": "Pot"
                },
                "hold": ASSEMBLE_COOKED_PLATE_FOOD + ASSEMBLE_CHARRED_PLATE_FOOD
            }
        },
        {  # hold plate + chopped, pick from puttable gridsquare
            "prep": {
                "targ": {
                    "obj": ASSEMBLE_CHOPPED_FOOD,
                    "gs": PUTTABLE_GRIDSQUARES
                },
                "hold": ASSEMBLE_CHOPPED_PLATE_FOOD + ['Plate']
            },
            "after": {
                "targ": {
                    "obj": "Nothing",
                    "gs": PUTTABLE_GRIDSQUARES
                },
                "hold": ASSEMBLE_CHOPPED_PLATE_FOOD
            }
        },
        {  # hold chopped, pick from plate tile
            "prep": {
                "targ": {
                    "obj": "Nothing",
                    "gs": "PlateTile"
                },
                "hold": ASSEMBLE_CHOPPED_FOOD
            },
            "after": {
                "targ": {
                    "obj": "Nothing",
                    "gs": "PlateTile"
                },
                "hold": ASSEMBLE_CHOPPED_PLATE_FOOD
            }
        },
    ]
    Desc = {"n": "pick", "v": "pick", "adj": "pickable"}

    def __init__(self, obj: str | list[str] = None, gs: str | list[str] = None):
        super().__init__(obj=obj, gs=gs)


# prep: none
# done: time reached
class MTWait(MidTask):
    def __init__(self, obj: str = None, gs: str = None, timeout: int = 5):
        self.obj = obj
        self.gs = gs
        self.timeout = timeout

        self._stage = 1

    def can_begin(self, env: EnvState):
        return True

    def __call__(self, env: EnvState):
        if (self.obj is not None or self.gs is not None) and env.navigate_pos_by_obj_gs(self.obj, self.gs):
            return self.Success, (0, 0), "Completed"
        if self._stage >= self.timeout:
            return self.Success, (0, 0), "Completed"
        self._stage += 1

        self_pos = env.agents[env.agent_idx].location
        other_pos = env.agents[1 - env.agent_idx].location
        if abs(self_pos[0] - other_pos[0]) + abs(self_pos[1] - other_pos[1]) <= 1:
            possible_moves = []
            for move in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_pos = (self_pos[0] + move[0], self_pos[1] + move[1])
                if env.check_pos_by_obj_gs(pos=(self_pos[0] + move[0], self_pos[1] + move[1]), gs='Floor') \
                        and new_pos != other_pos:
                    possible_moves.append(move)
            return self.Working, random.choice(possible_moves), "Waiting"

        else:
            return self.Working, (0, 0), "Waiting"


class MTFail(MidTask):

    def __init__(self, msg: str = 'Failed'):
        self.msg = msg

    def can_begin(self, env: EnvState):
        return True

    def __call__(self, env: EnvState):
        return self.Failed, (0, 0), self.msg


class MTSuccess(MidTask):
    def __init__(self, msg: str = 'Success'):
        self.msg = msg

    def can_begin(self, env: EnvState):
        return True

    def __call__(self, env: EnvState):
        return self.Success, (0, 0), self.msg
