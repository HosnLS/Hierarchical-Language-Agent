from collections import defaultdict

from gym_cooking.utils.config import *
from gym_cooking.utils.core import GridSquare, Object, mergeable, add_something, \
    GRIDSQUARES, PUTTABLE_GRIDSQUARES, FOOD_TILE, \
    FRESH_FOOD, CHOPPING_FOOD, CHOPPED_FOOD, COOKING_FOOD, COOKED_FOOD, CHARRED_FOOD, \
    ASSEMBLE_CHOPPED_FOOD, ASSEMBLE_CHOPPED_PLATE_FOOD, ASSEMBLE_COOKING_FOOD, ASSEMBLE_COOKING_PLATE_FOOD, \
    ASSEMBLE_COOKED_FOOD, ASSEMBLE_COOKED_PLATE_FOOD, ASSEMBLE_CHARRED_FOOD, ASSEMBLE_CHARRED_PLATE_FOOD
from agent.executor.low import bfs_search, match_any, fname, fname_content, EnvState
from agent.executor.mid import MidTask, MTChop, MTPutout, MTPut, MTDrop, MTDeliver, MTCook, MTAssemble, MTPick, MTWait, MTFail, MTSuccess

OBJ_TO_GOODS_GS = defaultdict(lambda: "Unknown")
OBJ_TO_GOODS_GS.update({
    'FireExtinguisher': 'Fire Extinguisher',
    'Plate': 'Plate',

    'FreshLettuce': 'Fresh Lettuce',
    'FreshOnion': 'Fresh Onion',
    'FreshTomato': 'Fresh Tomato',

    'ChoppingLettuce': 'Chopping Lettuce',
    'ChoppingOnion': 'Chopping Onion',
    'ChoppingTomato': 'Chopping Tomato',

    'ChoppedLettuce': 'Chopped Lettuce',
    'ChoppedOnion': 'Chopped Onion',
    'ChoppedTomato': 'Chopped Tomato',

    'ChoppedLettuce-ChoppedOnion': 'Alice Ingredients',
    'ChoppedLettuce-ChoppedTomato': 'Bob Ingredients',
    'ChoppedOnion-ChoppedTomato': 'Cathy Ingredients',
    'ChoppedLettuce-ChoppedOnion-ChoppedTomato': 'David Ingredients',

    'ChoppedLettuce-ChoppedOnion-Plate': 'Alice Ingredients with Plate',
    'ChoppedLettuce-ChoppedTomato-Plate': 'Bob Ingredients with Plate',
    'ChoppedOnion-ChoppedTomato-Plate': 'Cathy Ingredients with Plate',
    'ChoppedLettuce-ChoppedOnion-ChoppedTomato-Plate': 'David Ingredients with Plate',

    'CookedLettuce-CookedOnion-Plate': 'Plated Alice Soup',
    'CookedLettuce-CookedTomato-Plate': 'Plated Bob Soup',
    'CookedOnion-CookedTomato-Plate': 'Plated Cathy Soup',
    'CookedLettuce-CookedOnion-CookedTomato-Plate': 'Plated David Soup',

    'CharredLettuce-CharredOnion-Plate': 'Plated Charred Alice Soup',
    'CharredLettuce-CharredTomato-Plate': 'Plated Charred Bob Soup',
    'CharredOnion-CharredTomato-Plate': 'Plated Charred Cathy Soup',
    'CharredLettuce-CharredOnion-CharredTomato-Plate': 'Plated Charred David Soup',
})

OBJ_TO_GOODS_POT = {
    'CookingLettuce': 'Cooking Lettuce',
    'CookingOnion': 'Cooking Onion',
    'CookingTomato': 'Cooking Tomato',
    'CookedLettuce': 'Cooked Lettuce',
    'CookedOnion': 'Cooked Onion',
    'CookedTomato': 'Cooked Tomato',
    'CharredLettuce': 'Charred Lettuce',
    'CharredOnion': 'Charred Onion',
    'CharredTomato': 'Charred Tomato',
    'CharredLettuce-Fire': 'Charred Lettuce',
    'CharredOnion-Fire': 'Charred Onion',
    'CharredTomato-Fire': 'Charred Tomato',

    'CookingLettuce-CookingOnion': 'Alice Soup',
    'CookingLettuce-CookingTomato': 'Bob Soup',
    'CookingOnion-CookingTomato': 'Cathy Soup',
    'CookingLettuce-CookingOnion-CookingTomato': 'David Soup',

    'CookedLettuce-CookedOnion': 'Alice Soup',
    'CookedLettuce-CookedTomato': 'Bob Soup',
    'CookedOnion-CookedTomato': 'Cathy Soup',
    'CookedLettuce-CookedOnion-CookedTomato': 'David Soup',

    'CharredLettuce-CharredOnion': 'Alice Soup',
    'CharredLettuce-CharredTomato': 'Bob Soup',
    'CharredOnion-CharredTomato': 'Cathy Soup',
    'CharredLettuce-CharredOnion-CharredTomato': 'David Soup',

    'CharredLettuce-CharredOnion-Fire': 'Alice Soup',
    'CharredLettuce-CharredTomato-Fire': 'Bob Soup',
    'CharredOnion-CharredTomato-Fire': 'Cathy Soup',
    'CharredLettuce-CharredOnion-CharredTomato-Fire': 'David Soup',
}
ALL_FRESH_FOOD = ['Tomato', 'Lettuce', 'Onion']
ALL_ASSEMBLE = ['Alice Ingredients', 'Bob Ingredients',
                'Cathy Ingredients', 'David Ingredients']
ALL_SOUP = ['Alice Soup', 'Bob Soup', 'Cathy Soup', 'David Soup']
GOODS_TO_OBJ_GS = defaultdict(lambda: "Unknown")
GOODS_TO_OBJ_GS.update({v: k for k, v in OBJ_TO_GOODS_GS.items()})
GOODS_TO_OBJ_GS.update({
    "Alice Soup": "CookedLettuce-CookedOnion",
    "Bob Soup": "CookedLettuce-CookedTomato",
    "Cathy Soup": "CookedOnion-CookedTomato",
    "David Soup": "CookedLettuce-CookedOnion-CookedTomato",
})
HT_MAP = {
    'Chop': 'Chop',
    'Assemble': 'Prepare',
    'Cook': 'Cook',
    'Putout': 'Putout',
    'Pick': 'Plate',
    'Serve': 'Serve',
    'Drop': 'Drop',
}

# High Task: Macro Action

# Chop [FOOD_INGREDIENT]                : Tomato, Lettuce, Onion
# Assemble [ASSEMBLED_CHOPPED_FOOD]     : ASSEMBLE_CHOPPED_FOOD
# Put out fire
# Cook [SOUP]                           : ASSEMBLE_COOKED_FOOD
# Pick up [SOUP] with a plate
# Serve [SOUP]
# Drop                                  : should be executed when there is no empty counter

class HighTask:
    MAX_TRY_TIMES = 3
    Success = 1
    Working = 0
    Failed = -1

    def __init__(self):
        self._last_task = False
        self._try_times = 0
        self._task = []

    def can_begin(self, env: EnvState):
        return True, "Can begin", [], 0

    def _get_subtask(self, env: EnvState):
        return

    def __call__(self, env: EnvState):
        # start
        if not self._task:
            self._get_subtask(env)

        while not self._last_task or self._task:
            state, move, msg = self._task[0](env)
            if state == self.Working:  # working
                return self.Working, move, msg
            elif state == MidTask.Failed:  # reassign task
                self._try_times += 1
                if self._try_times >= self.MAX_TRY_TIMES:
                    return self.Failed, move, msg
                self._task = []
            else:
                self._task.pop(0)
            if not self._task:
                self._get_subtask(env)

        return self.Success, (0, 0), "Completed"

    def __str__(self):
        return 'HighTask'

    def __eq__(self, other):
        return str(self) == str(other)


# if reachable empty counter <= 1, clean a counter by dropping all thing it holds
class HTClean_(HighTask):
    #   (empty counter <= 1)
    #       (hold plate) put on platetile
    #       (hold something) drop
    #       pickup something
    #   success
    def __init__(self):
        super().__init__()

    def _get_subtask(self, env: EnvState):
        if self._last_task:
            self._task = []
            return

        res = env.get_all_grid_info()

        # chopped vegetables
        order_fname = [a[0].full_name.replace("-Plate", "") for a in env.order.current_orders]
        req = {a: 0 for a in ALL_FRESH_FOOD}
        for o in order_fname:
            fns = o.split('-')
            for f in fns:
                req[f.replace("Cooked", "")] += 1
        for o in env.all_obj_a:
            fns = fname(o).split('-')
            for f in fns:
                f = f.replace("Cooked", "").replace("Cooking", "").replace("Chopped", "")
                if f in req: req[f] -= 1
        USELESS_ASSEMBLES = [k for k, v in req.items() if v <= -1]
        USELESS_ASSEMBLES = [f"Chopped{x}" for x in USELESS_ASSEMBLES]
        USELESS_ASSEMBLES = USELESS_ASSEMBLES if USELESS_ASSEMBLES else ASSEMBLE_CHOPPED_FOOD

        if len([e for e in res if e['gs'].name == 'Counter' and e['obj'] is None and e['rch']]) <= 1:
            if match_any(fname(env.hold), 'Plate'):
                self._task = [MTPut(gs='PlateTile')]
                return
            if not match_any(fname(env.hold), "Nothing"):
                self._task = [MTDrop()]
                return
            if env.navigate_pos_by_obj_gs(obj='Plate', gs='Counter') is not None:
                self._task = [MTPick(obj='Plate', gs='Counter')]
            elif env.navigate_pos_by_obj_gs(obj=ASSEMBLE_CHARRED_PLATE_FOOD, gs='Counter') is not None:
                self._task = [MTPick(obj=ASSEMBLE_CHARRED_PLATE_FOOD, gs='Counter')]
            elif env.navigate_pos_by_obj_gs(obj=FRESH_FOOD, gs='Counter') is not None:
                self._task = [MTPick(obj=FRESH_FOOD, gs='Counter')]
            elif env.navigate_pos_by_obj_gs(obj=USELESS_ASSEMBLES, gs='Counter') is not None:
                self._task = [MTPick(obj=USELESS_ASSEMBLES, gs='Counter')]
            elif env.navigate_pos_by_obj_gs(obj=ASSEMBLE_CHOPPED_FOOD, gs='Counter') is not None:
                self._task = [MTPick(obj=ASSEMBLE_CHOPPED_FOOD, gs='Counter')]
            elif env.navigate_pos_by_obj_gs(obj=ASSEMBLE_CHOPPED_PLATE_FOOD, gs='Counter') is not None:
                self._task = [MTPick(obj=ASSEMBLE_CHOPPED_PLATE_FOOD, gs='Counter')]
            elif env.navigate_pos_by_obj_gs(obj=ASSEMBLE_COOKED_PLATE_FOOD, gs='Counter') is not None:
                self._task = [MTPick(obj=ASSEMBLE_COOKED_PLATE_FOOD, gs='Counter')]
            else:
                self._task = [MTPick(gs='Counter')]
            return
        self._task = [MTSuccess()]
        self._last_task = True
        return

    def __str__(self):
        return "Clean_"


# Lettuce, Onion, Tomato
class HTChop(HighTask):
    #   (no empty counter) HTClean
    #   (no desired cutboard, regarding agent reachable)
    #       (all cutboard cutting)
    #           (hold something) put on empty counter
    #           finish cut
    #       (no empty cutboard)
    #           (hold something) put on empty counter
    #           pickup one, put to counter
    #       (hold other) put on empty counter
    #       (hold none)
    #           (has single vegetable) pickup vegetable from counter
    #           (no single vegetable) pickup vegetable from tile
    #       put to empty cutboard
    #   (hold something) put on empty counter
    #   chop desired cutboard, pick, put
    def __init__(self, obj: str):
        super().__init__()

        # obj: Tomato, Lettuce, Onion
        assert obj in ['Tomato', 'Lettuce', 'Onion']
        self.ingre = obj

    def _get_subtask(self, env: EnvState):
        if self._last_task:
            self._task = []
            return

        if env.navigate_pos_by_obj_gs(obj='Nothing', gs='Counter') is None:
            self._task = [HTClean_()]
            return

        if env.navigate_pos_by_obj_gs(obj='Chopping' + self.ingre, gs='Cutboard', agent=True) is None:
            if all([fname(env.pos_obj[o.location]) in CHOPPING_FOOD for o in env.rch_grid if o.name == 'Cutboard']):
                if not match_any(fname(env.hold), "Nothing"):
                    self._task = [MTPut(gs='Counter')]
                    return
                self._task = [MTChop()]
                return
            if env.navigate_pos_by_obj_gs(obj='Nothing', gs='Cutboard') is None:
                if not match_any(fname(env.hold), "Nothing"):
                    self._task = [MTPut(gs='Counter')]
                    return
                self._task = [MTPick(gs='Cutboard'), MTPut(gs='Counter')]
                return
            if not match_any(fname(env.hold), ['Fresh' + self.ingre, 'Nothing']):
                self._task = [MTPut(gs='Counter')]
                return
            if match_any(fname(env.hold), "Nothing"):
                if env.navigate_pos_by_obj_gs(obj='Fresh' + self.ingre, gs='Counter') is not None:
                    self._task = [MTPick(obj='Fresh' + self.ingre)]
                    return
                else:
                    self._task = [MTPick(gs='Fresh' + self.ingre + 'Tile')]
                    return
            self._task = [MTPut(gs='Cutboard')]
            return
        if not match_any(fname(env.hold), "Nothing"):
            self._task = [MTPut(gs='Counter')]
            return
        self._task = [MTChop(obj='Chopping' + self.ingre), MTPick(obj='Chopped' + self.ingre, gs='Cutboard'),
                      MTPut(gs='Counter')]
        self._last_task = True
        return

    def can_begin(self, env: EnvState):
        order_fname = [a[0].full_name.replace("-Plate", "") for a in env.order.current_orders]
        req = {a: 0 for a in ALL_FRESH_FOOD}
        for o in order_fname:
            fns = o.split('-')
            for f in fns:
                req[f.replace("Cooked", "")] += 1
        for o in env.all_obj_a:
            fns = fname(o).split('-')
            for f in fns:
                f = f.replace("Cooked", "").replace("Cooking", "").replace("Chopped", "")
                if f in req: req[f] -= 1

        score = 0.5 * (req[self.ingre] > 0)
        score = max(score, 0.1)

        can_begin = False
        if env.navigate_pos_by_obj_gs(gs="Fresh" + self.ingre + "Tile") is not None:
            can_begin = True
        elif env.navigate_pos_by_obj_gs(obj="Fresh" + self.ingre, gs='Counter') is not None:
            can_begin = True
        elif env.navigate_pos_by_obj_gs(obj="Chopping" + self.ingre, gs='Cutboard') is not None:
            can_begin = True

        if env.navigate_pos_by_obj_gs(gs='Cutboard') is None:
            can_begin = False

        return can_begin, f"You can begin chopping the {self.ingre}.", [], score

    def __str__(self):
        return f"{HT_MAP['Chop']} {self.ingre}"


# ASSEMBLE_CHOPPED_FOOD
class HTAssemble(HighTask):
    #   (no empty counter) HTClean
    #   (decomposed task not empty)
    #       (no grad 1) chop grad 1
    #       (no grad 2) chop grad 2
    #       (not hold grad 1)
    #           (hold something) put on empty counter
    #           pickup grad 1
    #       assemble to grad 2
    #   success
    MAX_TRY_TIMES = 3
    RECIPY = {
        'ChoppedLettuce-ChoppedOnion': [('ChoppedLettuce', 'ChoppedOnion')],
        'ChoppedLettuce-ChoppedTomato': [('ChoppedLettuce', 'ChoppedTomato')],
        'ChoppedOnion-ChoppedTomato': [('ChoppedOnion', 'ChoppedTomato')],
        'ChoppedLettuce-ChoppedOnion-ChoppedTomato': [
            ('ChoppedOnion-ChoppedTomato', 'ChoppedLettuce'),
            ('ChoppedLettuce-ChoppedTomato', 'ChoppedOnion'),
            ('ChoppedLettuce-ChoppedOnion', 'ChoppedTomato'),
        ]
    }

    def __init__(self, obj: str):
        super().__init__()

        self.ingred = obj
        self.target = GOODS_TO_OBJ_GS[obj]
        assert self.target in self.RECIPY

    def _get_combine(self, rch_obj_fname):
        targ = (self.target, None, self.target)
        while True:
            recipy = self.RECIPY[targ[0]]
            for ingred, ingred2 in recipy:
                if ingred in rch_obj_fname or '-' not in ingred:
                    return ingred, ingred2, targ[0]
            targ = (recipy[0][0], recipy[0][1], targ[0])

    def _get_subtask(self, env: EnvState):
        if self._last_task:
            self._task = []
            return

        if env.navigate_pos_by_obj_gs(obj='Nothing', gs='Counter') is None:
            self._task = [HTClean_()]
            return

        rch_obj_fname = [fname(a) for a in env.rch_obj_h]
        targ = self._get_combine(rch_obj_fname)
        if targ[0] not in rch_obj_fname:
            self._task = [MTFail(f"You need to chop {OBJ_TO_GOODS_GS[targ[0]]} first.")]
            return
        if targ[1] is not None and targ[1] not in rch_obj_fname:
            self._task = [MTFail(f"You need to chop {OBJ_TO_GOODS_GS[targ[1]]} first.")]
            return
        if not match_any(fname(env.hold), targ[0]):
            if not match_any(fname(env.hold), "Nothing"):
                self._task = [MTPut(gs='Counter')]
                return
            self._task = [MTPick(obj=targ[0])]
            return
        self._task = [MTAssemble(obj=targ[1])]
        if targ[2] == self.target:
            self._last_task = True
        return

    def can_begin(self, env: EnvState):
        # calc score: 0 or 0.5
        target = self.target.replace("Chopped", "Cooked")
        cur_orders = [a for a in env.order.current_orders if a[0].full_name.replace("-Plate", "") == target]
        cur_orders.sort(key=lambda x: x[1] / x[2])
        finish_dish = [a for a in env.all_obj_a if
                       fname(a) in [target, target.replace("Cooked", "Cooking"), target.replace("Cooked", "Chopped")]]
        cur_orders = cur_orders[len(finish_dish):]
        score = (len(cur_orders) > 0) * 0.52

        # can begin
        rch_obj_fname = [fname(a) for a in env.rch_obj_h]
        miss = []
        while True:
            targ = self._get_combine(rch_obj_fname)
            if targ[0] not in rch_obj_fname: miss.append(targ[0])
            if targ[1] not in rch_obj_fname: miss.append(targ[1])
            if targ[2] == self.target: break
            rch_obj_fname.append(targ[2])

        if len(miss) > 0:
            prompt = ''
            for idx, m in enumerate(miss[::-1]):
                m = f"`{m.replace('Chopped', '')}`"
                if idx == 0:
                    prompt = m + prompt
                elif idx == 1:
                    prompt = m + " and " + prompt
                else:
                    prompt = m + ", " + prompt
            return False, f"You need to chop {prompt} first.", [str(HTChop(x.replace('Chopped', ''))) for x in miss], 0
        return True, f"You can assemble {self.ingred} now as all required chopped vegetables are on the map.", [], score

    def __str__(self):
        return f"{HT_MAP['Assemble']} {self.ingred}"


class HTPutout(HighTask):
    #   (no empty counter) HTClean
    #   (no fire) ERROR
    #   (hold is not extinguisher) put on empty counter
    #   (hold none) pickup extinguisher
    #   putout fire
    def __init__(self):
        super().__init__()

    def _get_subtask(self, env: EnvState):
        if self._last_task:
            self._task = []
            return

        if env.navigate_pos_by_obj_gs(obj='Nothing', gs='Counter') is None:
            self._task = [HTClean_()]
            return

        if not env.navigate_pos_by_obj_gs(obj='Fire', gs='Pot', inner=True):
            self._task = [MTFail("There is currently no fire on the map, so the situation is safe.")]
            return
        if not match_any(fname(env.hold), "FireExtinguisher"):
            if not match_any(fname(env.hold), "Nothing"):
                self._task = [MTPut(gs='Counter')]
                return
            self._task = [MTPick(obj='FireExtinguisher')]
            return
        self._task = [MTPutout()]
        self._last_task = True
        return

    def can_begin(self, env: EnvState):
        has_fire = env.navigate_pos_by_obj_gs(obj='Fire', gs='Pot', inner=True) is not None
        rch_obj_fname = [fname(a) for a in env.rch_obj_h]

        if not has_fire:
            return False, "There is currently no fire on the map, so the situation is safe.", [], 0
        elif "FireExtinguisher" not in rch_obj_fname:
            return False, "There is currently no fire extinguisher on the map, so you cannot put out the fire.", [], 0
        else:
            return True, "There is a fire on the map, and you can perform the putout action.", [], 0.6

    def __str__(self):
        return f"{HT_MAP['Putout']}"


# SOUP
class HTCook(HighTask):
    #   (no empty counter) HTClean
    #   (ingredient not on map) ERROR
    #   (all pot on fire) ERROR
    #   (all pot cooking)
    #       (hold something) put on empty counter
    #       wait for cooked
    #   (no empty pot)
    #       (hold is not plate) put on empty counter
    #       (hold none) pickup plate
    #       pickup soup, put on empty counter
    #   (hold other ingredient) put on empty counter
    #   (hold none) pickup ingredient
    #   put to empty pot, success
    def __init__(self, obj: str):
        super().__init__()
        self.soup = obj
        self.target = GOODS_TO_OBJ_GS[obj].replace("-Plate", "")
        assert self.target in ASSEMBLE_COOKED_FOOD

    def _get_subtask(self, env: EnvState):
        if self._last_task:
            self._task = []
            return

        if env.navigate_pos_by_obj_gs(obj='Nothing', gs='Counter') is None:
            self._task = [HTClean_()]
            return

        rch_obj_fname = [fname(a) for a in env.rch_obj_h]
        ingred = self.target.replace('Cooked', 'Chopped')
        ingred2 = add_something([ingred], "Plate")[0]

        if ingred not in rch_obj_fname and ingred2 not in rch_obj_fname:
            self._task = [MTFail(f"You should assemble {OBJ_TO_GOODS_GS[ingred]} first before cooking the soup.")]

        if all(["Fire" in fname(env.pos_obj[o.location]) for o in env.rch_grid if o.name == 'Pot']):
            self._task = [
                MTFail("You need to put out the fire in the pot first because all pots are currently on fire.")]
            return
        if len(env.get_pos_by_obj_gs(obj=COOKING_FOOD, gs='Pot')) == \
                len([e for e in env.world_all if e.name == 'Pot']):
            if not match_any(fname(env.hold), "Nothing"):
                self._task = [MTPut(gs='Counter')]
                return
            self._task = [MTWait(obj=COOKED_FOOD, gs='Pot')]
            return
        if env.navigate_pos_by_obj_gs(obj='Nothing', gs='Pot') is None:
            if not match_any(fname(env.hold), 'Plate'):
                if not match_any(fname(env.hold), "Nothing"):
                    self._task = [MTPut(gs='Counter')]
                    return
                self._task = [MTPick(gs='PlateTile')]
                return
            self._task = [MTWait(obj=ASSEMBLE_COOKED_FOOD + ASSEMBLE_CHARRED_FOOD, gs='Pot', timeout=10),
                          MTPick(obj=ASSEMBLE_COOKED_FOOD + ASSEMBLE_CHARRED_FOOD, gs='Pot'),
                          MTPut(gs='Counter')]
            return
        if not match_any(fname(env.hold), [ingred, ingred2]):
            if not match_any(fname(env.hold), "Nothing"):
                if match_any(fname(env.hold), "Plate"):
                    self._task = [MTPut(gs='PlateTile')]
                    return
                self._task = [MTPut(gs='Counter')]
                return
            if env.navigate_pos_by_obj_gs(obj=ingred) is not None:
                self._task = [MTPick(obj=ingred)]
                return
            else:
                self._task = [MTPick(obj=ingred2)]
                return
        self._task = [MTCook()]
        self._last_task = True
        return

    def can_begin(self, env: EnvState):
        # calc score: 0 or 0.5
        cur_orders = [a for a in env.order.current_orders if a[0].full_name.replace("-Plate", "") == self.target]
        cur_orders.sort(key=lambda x: x[1] / x[2])
        finish_dish = [a for a in env.all_obj_a if fname(a) in [GOODS_TO_OBJ_GS[self.soup],
                                                                GOODS_TO_OBJ_GS[self.soup].replace("Cooked",
                                                                                                   "Cooking")]]
        cur_orders = cur_orders[len(finish_dish):]
        score = (len(cur_orders) > 0) * 0.54

        rch_obj_fname = [fname(a) for a in env.rch_obj_h]

        ingred = self.target.replace('Cooked', 'Chopped')
        ingred2 = add_something([ingred], "Plate")[0]

        if ingred not in rch_obj_fname and ingred2 not in rch_obj_fname:
            return False, f"You should assemble {OBJ_TO_GOODS_GS[ingred]} first before cooking the soup.", [
                str(HTAssemble(OBJ_TO_GOODS_GS[ingred]))], 0
        if env.navigate_pos_by_obj_gs(gs="Pot") is None:
            return False, "No reachable pot", [], 0
        
        if all(["Fire" in fname(env.pos_obj[o.location]) for o in env.rch_grid if o.name == 'Pot']):
            return False, "You need to put out the fire in the pot first because all pots are currently on fire.", [HTPutout()], 0
        return True, f"The required ingredients for {self.soup} are ready, so you can start cooking it.", [], score

    def __str__(self):
        return f"{HT_MAP['Cook']} {self.soup}"


# SOUP
class HTPick(HighTask):
    #   (no empty counter) HTClean
    #   (hold not plate) put on empty counter
    #   (hold nothing) pickup plate
    #   (no target) wait
    #   pickup target, put on empty counter
    def __init__(self, obj: str):
        super().__init__()
        self.dish = obj
        self.target = GOODS_TO_OBJ_GS[obj].replace("-Plate", "")
        assert self.target in ASSEMBLE_COOKED_FOOD

    def _get_subtask(self, env: EnvState):
        if self._last_task:
            self._task = []
            return

        if env.navigate_pos_by_obj_gs(obj='Nothing', gs='Counter') is None:
            self._task = [HTClean_()]
            return

        rch_obj_fname = [fname(a) for a in env.rch_obj_h]
        if not match_any(fname(env.hold), "Plate"):
            if not match_any(fname(env.hold), "Nothing"):
                self._task = [MTPut(gs='Counter')]
                return
            if env.navigate_pos_by_obj_gs(obj='Plate') is not None:
                self._task = [MTPick(obj='Plate')]
                return
            else:
                self._task = [MTPick(gs='PlateTile')]
                return
        if self.target not in rch_obj_fname:
            if self.target.replace("Cooked", "Cooking") in rch_obj_fname:
                self._task = [MTFail(f"The {self.dish} is cooking, so you cannot pick it up.")]
                return
            else:
                self._task = [MTFail(f"The {self.dish} is not cooked yet, so you cannot pick it up.")]
                return
        self._task = [MTPick(obj=self.target)]
        self._last_task = True
        return

    def can_begin(self, env: EnvState):
        # calc score: 0-1. order satisfy 0.5-1, charred 0-1
        cur_orders = [a for a in env.order.current_orders if a[0].full_name.replace("-Plate", "") == self.target]
        cur_orders.sort(key=lambda x: x[1] / x[2])
        finish_dish = [a for a in env.all_obj_a if fname(a) == GOODS_TO_OBJ_GS[self.dish]]
        cur_orders = cur_orders[len(finish_dish):]
        if len(cur_orders) == 0:
            score1 = 0
        else:
            score1 = 1 - min(0.44, min([a[1] / a[2] for a in cur_orders]))

        rch_obj_fname = [fname(a) for a in env.rch_obj_h]

        grid_info = env.get_all_grid_info()
        match_soup = [e for e in grid_info if e['gs'].name == 'Pot' and fname(e['obj']) == self.target]
        if len(match_soup) == 0:
            score2 = 0
        else:
            score2 = 1 - min([e['obj'].rest_turn_time() / COOKED_BEFORE_FIRE_TIME_SECONDS for e in match_soup])

        score = max(score1, score2)

        if "Plate" not in rch_obj_fname and env.navigate_pos_by_obj_gs(gs='PlateTile') is None:
            return False, "No reachable plate.", [], 0

        if self.target not in rch_obj_fname:
            if self.target.replace("Cooked", "Cooking") in rch_obj_fname:
                return False, f"The {self.dish} is currently cooking. You will plate it when it is ready.", [], 0
            else:
                return False, f"The {self.dish} is not cooked yet, so you cannot plate it.", [], 0
        return True, f"The {self.dish} has finished cooking and is ready to be plated.", [], score

    def __str__(self):
        return f"{HT_MAP['Pick']} {self.dish}"


# SOUP
class HTServe(HighTask):
    # (no empty counter) HTClean
    # (soup not on map) ERROR
    # (hold not target) put on empty counter
    # (hold nothing) pickup target
    # deliver
    #
    # time(score): (0.0)1, (0.5)1, (1.0)2
    def __init__(self, obj: str):
        super().__init__()
        self.dish = obj
        self.target = GOODS_TO_OBJ_GS[obj].replace("-Plate", "")
        assert self.target in ASSEMBLE_COOKED_FOOD

    def _get_subtask(self, env: EnvState):
        if self._last_task:
            self._task = []
            return

        if env.navigate_pos_by_obj_gs(obj='Nothing', gs='Counter') is None:
            self._task = [HTClean_()]
            return

        rch_obj_fname = [fname(a) for a in env.rch_obj_h]
        targ = add_something([self.target], 'Plate')[0]
        if targ not in rch_obj_fname:
            if targ.replace("Cooked", "Cooking") in rch_obj_fname:
                self._task = [MTFail(
                    f"The {self.dish} is currently cooking, so you need to wait for it to be cooked and then pick it up before you can serve it. You are free to do other actions while waiting.")]
                return
            else:
                self._task = [MTFail(f"The {self.dish} is not yet cooked, so you cannot serve it.")]
                return
        if not match_any(fname(env.hold), targ):
            if not match_any(fname(env.hold), "Nothing"):
                self._task = [MTPut(gs='Counter')]
                return
            self._task = [MTPick(obj=targ)]
            return
        self._task = [MTDeliver()]
        self._last_task = True
        return

    def can_begin(self, env: EnvState):
        rch_obj_fname = [fname(a) for a in env.rch_obj_h]
        # calc score: 0-1
        cur_orders = [a for a in env.order.current_orders if a[0].full_name.replace("-Plate", "") == self.target]
        if len(cur_orders) == 0:
            score = 0
        else:
            score = 1 - min(0.42, min([a[1] / a[2] for a in cur_orders]))

        if env.navigate_pos_by_obj_gs(gs="Delivery") is None:
            return False, "No reachable delivery.", [], 0

        if add_something([self.target], 'Plate')[0] not in rch_obj_fname:
            if self.target.replace("Cooked", "Cooking") in rch_obj_fname:
                return False, f"The {self.dish} is currently cooking. You need to wait for it to be cooked and plate it first.", [
                    str(HTPick(self.dish))], 0
            elif self.target in rch_obj_fname:
                return False, f"The {self.dish} is still in the pot. You need to plate it first.", [
                    str(HTPick(self.dish))], 0
            else:
                return False, f"The {self.dish} is not yet cooked. You need to cook it and plate it first.", [
                    str(HTPick(self.dish))], 0
        return True, f"You can serve the {self.dish} now because the cooked soup has been picked up and is ready to be served.", [], score

    def __str__(self):
        return f"{HT_MAP['Serve']} {self.dish}"


class HTDrop(HighTask):
    #   (no empty counter) HTClean
    #   (no charred+p)
    #       (no charred) ERROR
    #       (hold not plate) put on empty counter
    #       (hold nothing) pickup plate
    #       pickup charred
    #   (hold not charred+p) put on empty counter
    #   (hold nothing) pickup charred+p
    #   drop target
    #
    # score: 1
    def __init__(self):
        super().__init__()

    def _get_subtask(self, env: EnvState):
        if self._last_task:
            self._task = []
            return

        if env.navigate_pos_by_obj_gs(obj='Nothing', gs='Counter') is None:
            self._task = [HTClean_()]
            return

        rch_obj_fname = [fname(a) for a in env.rch_obj_h]
        if not match_any(rch_obj_fname, ASSEMBLE_CHARRED_PLATE_FOOD):
            if not match_any(rch_obj_fname, ASSEMBLE_CHARRED_FOOD):
                self._task = [MTFail(f"There is no charred food on the map to drop.")]
                return
            if not match_any(fname(env.hold), "Plate"):
                if not match_any(fname(env.hold), "Nothing"):
                    self._task = [MTPut(gs='Counter')]
                    return
                self._task = [MTPick(gs='PlateTile')]
                return
            self._task = [MTPick(obj=ASSEMBLE_CHARRED_FOOD)]
            return
        if not match_any(fname(env.hold), ASSEMBLE_CHARRED_PLATE_FOOD):
            if not match_any(fname(env.hold), "Nothing"):
                self._task = [MTPut(gs='Counter')]
                return
            self._task = [MTPick(obj=ASSEMBLE_CHARRED_PLATE_FOOD)]
            return
        if env.navigate_pos_by_obj_gs(gs="Bin") is not None:
            self._task = [MTDrop()]
        else:
            self._task = [MTPut(gs="Counter")]
        self._last_task = True
        return

    def can_begin(self, env: EnvState):
        rch_obj_fname = [fname(a) for a in env.rch_obj_h]

        if "Plate" not in rch_obj_fname and env.navigate_pos_by_obj_gs(gs='PlateTile') is None:
            return False, "No reachable plate.", [], 0

        if not match_any(rch_obj_fname, ASSEMBLE_CHARRED_PLATE_FOOD + ASSEMBLE_CHARRED_FOOD):
            if match_any(rch_obj_fname, add_something(ASSEMBLE_CHARRED_FOOD, "Fire")):
                return False, f"The charred food is on fire; you need to put out the fire first.", [str(HTPutout())], 0
            return False, f"There is no charred food on the map to drop.", [], 0
        return True, " You can perform the drop action because there are charred food items on the map.", [], 0.6

    def __str__(self):
        return f"{HT_MAP['Drop']}"