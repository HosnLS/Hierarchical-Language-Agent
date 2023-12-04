# recipe planning
import gym_cooking.recipe_planner.utils as RECIPY_UTIL
from gym_cooking.utils.config import *

# helpers
import copy
from termcolor import colored as color
from collections import namedtuple


# -----------------------------------------------------------
# GRIDSQUARES
# -----------------------------------------------------------
GridSquareRepr = namedtuple("GridSquareRepr", "name location holding")

class Rep:
    FLOOR = ' '
    COUNTER = '-'
    CUTBOARD = '/'
    DELIVERY = '*'
    BIN = 'B'
    POT = 'U'
    TOMATO = 't'
    LETTUCE = 'l'
    ONION = 'o'
    PLATE = 'p'
    TOMATO_TILE = "T"
    LETTUCE_TILE = "L"
    ONION_TILE = "O"
    PLATE_TILE = "P"
    FIRE_EXTINGUISHER = "f"

class GridSquare:
    def __init__(self, name, location):
        self.name = name
        self.location = location   # (x, y) tuple
        self.holding = None
        self.color = 'white'
        self.collidable = True     # cannot go through
        self.dynamic = False       # cannot move around

    def __str__(self):
        return color(self.rep, self.color)

    def __eq__(self, o):
        return isinstance(o, GridSquare) and self.name == o.name

    def __copy__(self):
        gs = type(self)(self.name, self.location)
        gs.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            gs.holding = copy.copy(self.holding)
        return gs

    def acquire(self, obj):
        obj.location = self.location
        self.holding = obj

    def release(self):
        temp = self.holding
        self.holding = None
        return temp

class Floor(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self,"Floor", location)
        self.color = None
        self.rep = Rep.FLOOR
        self.collidable = False
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

class Counter(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self,"Counter", location)
        self.rep = Rep.COUNTER
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

class AgentCounter(Counter):
    def __init__(self, location):
        GridSquare.__init__(self,"Agent-Counter", location)
        self.rep = Rep.COUNTER
        self.collidable = True
    def __eq__(self, other):
        return Counter.__eq__(self, other)
    def __hash__(self):
        return Counter.__hash__(self)
    def get_repr(self):
        return GridSquareRepr(name=self.name, location=self.location, holding= None)

class Cutboard(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "Cutboard", location)
        self.rep = Rep.CUTBOARD
        self.collidable = True
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

class Delivery(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "Delivery", location)
        self.rep = Rep.DELIVERY
        self.holding = []
    def acquire(self, obj):
        obj.location = self.location
        self.holding.append(obj)
    def release(self):
        if self.holding:
            return self.holding.pop()
        else: return None
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

    def exists(self, obj_name):
        return any([o.full_name == obj_name for o in self.holding])

    def pop(self, obj_name):
        remaining = []
        success = False
        for o in self.holding:
            if not success and o.full_name == obj_name:
                success = True
            else:
                remaining.append(o)
        self.holding=remaining

class Bin(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "Bin", location)
        self.rep = Rep.BIN
        self.collidable = True
    def acquire(self, obj):
        pass
    def release(self):
        pass
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

class Pot(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "Pot", location)
        self.rep = Rep.POT
        self.collidable = True

    def acquire(self, obj):
        GridSquare.acquire(self, obj)

# -----------------------------------------------------------
# OBJECTS
# -----------------------------------------------------------
# Objects are wrappers around foods items, plates, and any combination of them

ObjectRepr = namedtuple("ObjectRepr", "name location is_held")
ValidFoodNames = ('ChoppedLettuce', 'ChoppedOnion', 'ChoppedTomato',
                  'ChoppedLettuce-ChoppedOnion', 'ChoppedLettuce-ChoppedTomato', 'ChoppedOnion-ChoppedTomato',
                  'ChoppedLettuce-ChoppedOnion-ChoppedTomato')

class Object:
    def __init__(self, location, contents):
        self.location = location
        self.contents = contents if isinstance(contents, list) else [contents]
        self.is_held = False
        self.collidable = False
        self.dynamic = False

    def __str__(self):
        res = "-".join(list(map(lambda x : str(x), sorted(self.contents, key=lambda i: i.name))))
        return res

    def __eq__(self, other):
        # check that content is the same and in the same state(s)
        return isinstance(other, Object) and \
                self.name == other.name and \
                len(self.contents) == len(other.contents) and \
                self.full_name == other.full_name
                # all([i == j for i, j in zip(sorted(self.contents, key=lambda x: x.name),
                #                             sorted(other.contents, key=lambda x: x.name))])

    def __copy__(self):
        new = Object(self.location, self.contents[0])
        new.__dict__ = self.__dict__.copy()
        new.contents = [copy.copy(c) for c in self.contents]
        return new

    @property
    def name(self):
        # concatenate names of alphabetically sorted items, e.g.
        sorted_contents = sorted(self.contents, key=lambda c: c.name)
        return "-".join([c.name for c in sorted_contents])

    @property
    def full_name(self):
        # concatenate names of alphabetically sorted items, e.g.
        sorted_contents = sorted(self.contents, key=lambda c: c.full_name)
        return "-".join([c.full_name for c in sorted_contents])

    def get_repr(self):
        return ObjectRepr(name=self.full_name, location=self.location, is_held=self.is_held)

    def get_name(self):
        return self.full_name

    def contains(self, c_name):
        return c_name in list(map(lambda c : c.name, self.contents))

    def needs_chopped(self):
        if len(self.contents) > 1: return False
        return self.contents[0].needs_chopped()

    def is_chopped(self):
        for c in self.contents:
            if isinstance(c, Plate) or c.get_state() != 'Chopped':
                return False
        return True

    def chop(self, current_time):
        assert len(self.contents) == 1
        assert self.needs_chopped()
        self.contents[0].update_state(current_time)
        # assert not (self.needs_chopped())

    def merge(self, obj):
        if isinstance(obj, Object):
            # move obj's contents into this instance
            for i in obj.contents: self.contents.append(i)
        elif not (isinstance(obj, Food) or isinstance(obj, Plate) or isinstance(obj, Fire)):
            raise ValueError("Incorrect merge object: {}".format(obj))
        else:
            self.contents.append(obj)

    def unmerge(self, full_name):
        # remove by full_name, assumming all unique contents
        matching = list(filter(lambda c: c.full_name == full_name, self.contents))
        self.contents.remove(matching[0])
        return matching[0]

    def update_state(self, current_time):
        for c in self.contents:
            c.update_state(current_time)

    def split_food_plate(self):
        food = []
        plate = None
        for c in self.contents:
            if isinstance(c, Plate):
                assert plate is None
                plate = c
            else:
                food.append(c)
        return food, plate

    def is_merged(self):
        return len(self.contents) > 1

    def is_deliverable(self):
        # must be merged, and all contents must be Plates or Foods in done state
        for c in self.contents:
            if not (isinstance(c, Plate) or (isinstance(c, Food) and c.is_deliverable())):
                return False
        return self.is_merged()

    # cook logic
    def is_cookable(self):
        for c in self.contents:
            if not (isinstance(c, Plate) or isinstance(c, Food) and c.is_cookable()):
                return False
        return any([isinstance(c, Food) for c in self.contents])

    def is_cooked(self):
        for c in self.contents:
            if not ((isinstance(c, Food) and c.is_cooked()) or isinstance(c, Fire)):
                return False
        return True

    def is_onfire(self):
        return any([isinstance(c, Fire) for c in self.contents])

    def is_cooking(self):
        for c in self.contents:
            if not (isinstance(c, Food) and c.is_cooking()):
                return False
        return True

    def rest_cooking_time(self):
        for c in self.contents:
            if isinstance(c, Food):
                if c.is_cooked():
                    return 0.
                else:
                    assert c.is_cooking()
                    return c.state._rest_steps
        return 0.

    def rest_turn_time(self):
        if 'Fire' in self.full_name:
            for c in self.contents:
                if isinstance(c, Fire):
                    return c.rest
        for c in self.contents:
            if isinstance(c, Food):
                return c.state._rest_steps
        return 0.


    def cook(self, current_time):
        for c in self.contents:
            assert isinstance(c, Food)
            c.cook(current_time)

def mergeable(obj1, obj2):
    # query whether two objects are mergeable

    # if one object is empty -> mergeable
    if obj1 is None or obj2 is None:
        return True
    if len(obj1.contents) == 0 or len(obj2.contents) == 0:
        return True

    # if one is cooked and the other one is empty plate -> mergeable
    if obj1.is_cooked() and len(obj2.contents) == 1 and Plate() in obj2.contents:
        return True
    if obj2.is_cooked() and len(obj1.contents) == 1 and Plate() in obj1.contents:
        return True

    contents = obj1.contents + obj2.contents
    # check that there is at most one plate
    try:
        contents.remove(Plate())
    except:
        pass  # do nothing, 1 plate is ok
    finally:
        try:
            contents.remove(Plate())
        except:
            pass
        else:
            return False  # more than 1 plate
    sorted_contents = sorted(contents, key=lambda c: c.name)
    merged_name = "-".join([c.full_name for c in sorted_contents])
    return merged_name in ValidFoodNames


# -----------------------------------------------------------

class FoodState:
    FRESH = getattr(RECIPY_UTIL, 'Fresh')
    CHOPPING = getattr(RECIPY_UTIL, 'Chopping')
    CHOPPED = getattr(RECIPY_UTIL, 'Chopped')
    COOKING = getattr(RECIPY_UTIL, 'Cooking')
    COOKED = getattr(RECIPY_UTIL, 'Cooked')
    CHARRED = getattr(RECIPY_UTIL, 'Charred')
    DELIVERABLE = [CHOPPED, COOKED]

class FoodSequence:
    FRESH = [FoodState.FRESH]
    FRESH_CHOPPED = [FoodState.FRESH, FoodState.CHOPPED]
    FRESH_CHOPPING_CHOPPED = [FoodState.FRESH, FoodState.CHOPPING, FoodState.CHOPPED]
    FRESH_CHOPPING_CHOPPED_COOKING_COOKED = FRESH_CHOPPING_CHOPPED + [FoodState.COOKING, FoodState.COOKED]
    FRESH_CHOPPING_CHOPPED_COOKING_COOKED_CHARRED = FRESH_CHOPPING_CHOPPED_COOKING_COOKED + [FoodState.CHARRED]

class Food:
    def __init__(self):
        self.state = self.state_seq[self.state_index](self.name)
        self.movable = False
        self.color = self._set_color()

    def __str__(self):
        return color(self.rep, self.color)

    # def __hash__(self):
    #     return hash((self.state, self.name))

    def __eq__(self, other):
        return isinstance(other, Food) and self.get_state() == other.get_state()

    def __len__(self):
        return 1   # one food unit

    @property
    def full_name(self):
        return '{}{}'.format(self.get_state(), self.name)

    def set_state(self, state):
        assert state in self.state_seq, "Desired state {} does not exist for the food with sequence {}".format(state, self.state_seq)
        self.state_index = self.state_seq.index(state)
        self.state = state(self.name)

    def get_state(self):
        return self.state.name

    def needs_chopped(self):
        return self.state_seq[(self.state_index+1)%len(self.state_seq)] in [FoodState.CHOPPED, FoodState.CHOPPING]

    def is_deliverable(self):
        return self.state_seq[self.state_index] in FoodState.DELIVERABLE

    def update_state(self, current_time):
        if isinstance(self.state, FoodState.COOKING) or isinstance(self.state, FoodState.COOKED):
            self.state.update_by_current_time(current_time=current_time)
        else:
            self.state.update_one_step(passed=1)
        if self.state.is_finished:
            self.state_index += 1
            assert 0 <= self.state_index and self.state_index < len(self.state_seq), "State index is out of bounds for its state sequence"
            self.state = self.state_seq[self.state_index](obj=self.name, start_time=current_time)

    def _set_color(self):
        pass

    # cook logic
    def is_cookable(self):
        return FoodState.COOKING in self.state_seq and 0 <= self.state_index and self.state_index < len(self.state_seq) and self.state_seq[(self.state_index+1)%len(self.state_seq)] == FoodState.COOKING

    def is_cooked(self):
        return isinstance(self.state, FoodState.COOKED) or isinstance(self.state, FoodState.CHARRED)

    def is_cooking(self):
        return isinstance(self.state, FoodState.COOKING)

    def cook(self, current_time):
        assert self.is_cookable()
        self.state_index += 1
        assert self.state_seq[self.state_index] == FoodState.COOKING
        self.state = FoodState.COOKING(obj=self.name, start_time=current_time)

class Tomato(Food):
    def __init__(self, state_index = 0):
        self.state_index = state_index   # index in food's state sequence
        self.state_seq = FoodSequence.FRESH_CHOPPING_CHOPPED_COOKING_COOKED_CHARRED
        self.rep = 't'
        self.name = 'Tomato'
        Food.__init__(self)
    def __hash__(self):
        return Food.__hash__(self)
    def __eq__(self, other):
        return Food.__eq__(self, other)
    def __str__(self):
        return Food.__str__(self)

class Lettuce(Food):
    def __init__(self, state_index = 0):
        self.state_index = state_index   # index in food's state sequence
        self.state_seq = FoodSequence.FRESH_CHOPPING_CHOPPED_COOKING_COOKED_CHARRED
        self.rep = 'l'
        self.name = 'Lettuce'
        Food.__init__(self)
    def __eq__(self, other):
        return Food.__eq__(self, other)
    def __hash__(self):
        return Food.__hash__(self)

class Onion(Food):
    def __init__(self, state_index = 0):
        self.state_index = state_index   # index in food's state sequence
        self.state_seq = FoodSequence.FRESH_CHOPPING_CHOPPED_COOKING_COOKED_CHARRED
        self.rep = 'o'
        self.name = 'Onion'
        Food.__init__(self)
    def __eq__(self, other):
        return Food.__eq__(self, other)
    def __hash__(self):
        return Food.__hash__(self)


# -----------------------------------------------------------

class Plate:
    def __init__(self):
        self.rep = "p"
        self.name = 'Plate'
        self.full_name = 'Plate'
        self.color = 'white'
    def __hash__(self):
        return hash((self.name))
    def __str__(self):
        return color(self.rep, self.color)
    def __eq__(self, other):
        return isinstance(other, Plate)
    def __copy__(self):
        return Plate()
    def needs_chopped(self):
        return False

# -----------------------------------------------------------

class Fire:
    def __init__(self):
        self.rep = "x"
        self.name = 'Fire'
        self.full_name = 'Fire'
        self.color = 'red'
        self.rest = FIRE_PUTOUT_TIME_SECONDS
        self.last_putout_time = {}
        self.latest_time = -1e9
    def __hash__(self):
        return hash((self.name))
    def __str__(self):
        return color(self.rep, self.color)
    def __eq__(self, other):
        return isinstance(other, Fire)
    def __copy__(self):
        return Fire()
    def putout(self, agent_name, current_time):
        self.update_state(current_time)
        self.last_putout_time[agent_name] = current_time
    def remove_agent_putout_state(self, agent_name):
        if agent_name in self.last_putout_time.keys():
            self.last_putout_time.pop(agent_name)
    def update_state(self, current_time):
        # latest_time -> current_time
        # last_putout_time -> last_putout_time + FIRE_RECOVER_GAP_TIME_SECONDS
        # last_putout_time + FIRE_RECOVER_GAP_TIME_SECONDS -> current_time
        last_putout_time = max([v for k, v in self.last_putout_time.items()] + [-1e9])
        decrease = max([0, min([current_time, last_putout_time + FIRE_RECOVER_GAP_TIME_SECONDS]) - max([self.latest_time, last_putout_time])])
        increase = max([0, current_time - max([self.latest_time, last_putout_time + FIRE_RECOVER_GAP_TIME_SECONDS])])
        self.rest = self.rest - decrease
        if not self.is_finished:
            self.rest = min([FIRE_PUTOUT_TIME_SECONDS, self.rest + increase])
        self.latest_time = current_time
    @property
    def is_finished(self):
        return self.rest <= 1e-6

# -----------------------------------------------------------

class FireExtinguisher:
    def __init__(self):
        self.rep = 'f'
        self.name = 'FireExtinguisher'
        self.full_name = 'FireExtinguisher'
        self.color = 'red'
    def __hash__(self):
        return hash((self.name))
    def __str__(self):
        return color(self.rep, self.color)
    def __eq__(self, other):
        return isinstance(other, FireExtinguisher)
    def __copy__(self):
        return FireExtinguisher()

# -----------------------------------------------------------
# TILES
# -----------------------------------------------------------

class Tile(GridSquare):
    def __init__(self, location, tile_name, obj_cls):
        GridSquare.__init__(self, tile_name, location)
        self.rep = None
        self.obj_cls = obj_cls
    def release(self):
        obj = Object(location=self.location, contents=self.obj_cls())
        return obj
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

class TomatoTile(Tile):
    def __init__(self, location):
        Tile.__init__(self, location, 'FreshTomatoTile', Tomato)
        self.rep = Rep.TOMATO_TILE

class LettuceTile(Tile):
    def __init__(self, location):
        Tile.__init__(self, location, 'FreshLettuceTile', Lettuce)
        self.rep = Rep.LETTUCE_TILE

class OnionTile(Tile):
    def __init__(self, location):
        Tile.__init__(self, location, 'FreshOnionTile', Onion)
        self.rep = Rep.ONION_TILE

class PlateTile(Tile):
    def __init__(self, location):
        Tile.__init__(self, location, 'PlateTile', Plate)
        self.rep = Rep.PLATE_TILE


# -----------------------------------------------------------
# NAMES
# -----------------------------------------------------------
GRIDSQUARES = ["Floor", "Counter", "Cutboard", "Delivery", "Bin", "Pot", "FreshTomatoTile", "FreshOnionTile", "FreshLettuceTile", "PlateTile"]
PUTTABLE_GRIDSQUARES = ['Counter', 'Cutboard']
FOOD_TILE = ['FreshTomatoTile', 'FreshLettuceTile', 'FreshOnionTile']
FRESH_FOOD = ['FreshTomato', 'FreshLettuce', 'FreshOnion']
CHOPPING_FOOD = ['ChoppingTomato', 'ChoppingOnion', 'ChoppingLettuce']
CHOPPED_FOOD = ['ChoppedTomato', 'ChoppedOnion', 'ChoppedLettuce']
COOKING_FOOD = ['CookingTomato', 'CookingOnion', 'CookingLettuce']
COOKED_FOOD = ['CookedTomato', 'CookedOnion', 'CookedLettuce']
CHARRED_FOOD = ['CharredTomato', 'CharredOnion', 'CharredLettuce']


def add_something(obj_list, x):
    ret = []
    for obj in obj_list:
        objs = obj.split('-')
        objs.append(x)
        objs = sorted(objs)
        ret.append('-'.join(objs))
    return ret
ASSEMBLE_CHOPPED_FOOD = ['ChoppedTomato', 'ChoppedOnion', 'ChoppedLettuce', 'ChoppedLettuce-ChoppedOnion', 'ChoppedLettuce-ChoppedTomato', 'ChoppedOnion-ChoppedTomato', 'ChoppedLettuce-ChoppedOnion-ChoppedTomato']
ASSEMBLE_CHOPPED_PLATE_FOOD = add_something(ASSEMBLE_CHOPPED_FOOD, 'Plate')
ASSEMBLE_COOKING_FOOD = [f.replace('Chopped', 'Cooking') for f in ASSEMBLE_CHOPPED_FOOD]
ASSEMBLE_COOKING_PLATE_FOOD = add_something(ASSEMBLE_COOKING_FOOD, 'Plate')
ASSEMBLE_COOKED_FOOD = [f.replace('Chopped', 'Cooked') for f in ASSEMBLE_CHOPPED_FOOD]
ASSEMBLE_COOKED_PLATE_FOOD = add_something(ASSEMBLE_COOKED_FOOD, 'Plate')
ASSEMBLE_CHARRED_FOOD = [f.replace('Chopped', 'Charred') for f in ASSEMBLE_CHOPPED_FOOD]
ASSEMBLE_CHARRED_PLATE_FOOD = add_something(ASSEMBLE_CHARRED_FOOD, 'Plate')

# -----------------------------------------------------------
# PARSING
# -----------------------------------------------------------
RepToClass = {
    Rep.FLOOR: globals()['Floor'],
    Rep.COUNTER: globals()['Counter'],
    Rep.CUTBOARD: globals()['Cutboard'],
    Rep.DELIVERY: globals()['Delivery'],
    Rep.BIN: globals()['Bin'],
    Rep.POT: globals()['Pot'],
    Rep.TOMATO: globals()['Tomato'],
    Rep.LETTUCE: globals()['Lettuce'],
    Rep.ONION: globals()['Onion'],
    Rep.PLATE: globals()['Plate'],
    Rep.TOMATO_TILE: globals()['TomatoTile'],
    Rep.LETTUCE_TILE: globals()['LettuceTile'],
    Rep.ONION_TILE: globals()['OnionTile'],
    Rep.PLATE_TILE: globals()['PlateTile'],
    Rep.FIRE_EXTINGUISHER: globals()['FireExtinguisher']
}



