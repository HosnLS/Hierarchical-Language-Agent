import itertools

from gym_cooking.utils.core import GRIDSQUARES, PUTTABLE_GRIDSQUARES, FRESH_FOOD, CHOPPED_FOOD, CHOPPING_FOOD, \
    COOKING_FOOD, COOKED_FOOD, ASSEMBLE_CHOPPED_FOOD, ASSEMBLE_CHOPPED_PLATE_FOOD, ASSEMBLE_COOKING_FOOD, \
    ASSEMBLE_COOKING_PLATE_FOOD, ASSEMBLE_COOKED_FOOD, ASSEMBLE_COOKED_PLATE_FOOD, FOOD_TILE, \
    ASSEMBLE_CHARRED_FOOD, ASSEMBLE_CHARRED_PLATE_FOOD


class Event:
    def __init__(self, playerA, event, location, time, playerB=None):
        self.playerA = playerA
        self.event = event
        self.location = location
        self.time = time
        self.playerB = playerB


def get_all_events(recipes):
    no_op = ['No-op']

    move = ['Move']

    chop = [f'Chop_{f}' for f in FRESH_FOOD]

    cook = [f'Cook_{f}' for f in ASSEMBLE_CHOPPED_FOOD]

    assemble = [f'Assemble_{f}' for f in ASSEMBLE_CHOPPED_FOOD + ASSEMBLE_CHOPPED_PLATE_FOOD]

    put = [f'Put_{f}_on_{gs}' for f, gs in
           itertools.product(
               FRESH_FOOD + ASSEMBLE_CHOPPED_FOOD + ASSEMBLE_CHOPPED_PLATE_FOOD + ASSEMBLE_COOKED_PLATE_FOOD
               + ASSEMBLE_CHARRED_PLATE_FOOD + ['FireExtinguisher', 'Plate'], PUTTABLE_GRIDSQUARES)] \
          + ['Put_Plate_on_PlateTile']

    pickup = [f'Pickup_{f}_from_{gs}' for f, gs in
              itertools.product(
                  FRESH_FOOD + ASSEMBLE_CHOPPED_FOOD + ASSEMBLE_CHOPPED_PLATE_FOOD + ASSEMBLE_COOKED_PLATE_FOOD
                  + ASSEMBLE_CHARRED_PLATE_FOOD + ['FireExtinguisher', 'Plate'], PUTTABLE_GRIDSQUARES)] \
             + [f'Pickup_{f}_from_{gs}' for f, gs in itertools.product(ASSEMBLE_COOKED_FOOD + ASSEMBLE_CHARRED_FOOD, ["Pot"])]
    pickup += [f'Pickup_{f}_from_{gs}' for f, gs in zip(FRESH_FOOD, FOOD_TILE)] \
              + ['Pickup_Plate_from_PlateTile']

    deliver = [f'Deliver_{f}' for f in ASSEMBLE_CHOPPED_PLATE_FOOD + ASSEMBLE_COOKED_PLATE_FOOD]

    drop = [f'Drop_{f}' for f in FRESH_FOOD + ASSEMBLE_CHOPPED_FOOD + ASSEMBLE_COOKED_FOOD + ASSEMBLE_CHARRED_FOOD]

    putout = ['Putout_Fire']

    return no_op + move + chop + cook + assemble + put + pickup + deliver + drop + putout

# chop
# putout
# cook
# deliver
# drop
# put on gs
# assemble x (on gs)
# pickup x (on gs)
