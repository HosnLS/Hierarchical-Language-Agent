from gym_cooking.utils.core import *
from gym_cooking.utils.event import Event
import numpy as np


def interact(agent, world, current_time) -> Event:
    """Carries out interaction for this agent taking this action in this world.

    The action that needs to be executed is stored in `agent.action`.
    """

    # agent does nothing (i.e. no arrow key)
    if agent.action == (0, 0):
        # return None
        return Event(playerA=agent.name, event='No-op', location=agent.location, time=current_time)

    action_x, action_y = world.inbounds(tuple(np.asarray(agent.location) + np.asarray(agent.action)))
    gs = world.get_gridsquare_at((action_x, action_y))

    # if floor in front --> move to that square
    if isinstance(gs, Floor):  # and gs.holding is None:
        agent.move_to(gs.location)
        return Event(playerA=agent.name, event='Move', location=agent.location, time=current_time)

    # if holding something
    elif agent.holding is not None:
        # if delivery in front --> deliver
        if isinstance(gs, Delivery):
            obj = agent.holding
            if obj.is_deliverable():
                gs.acquire(obj)
                agent.release()
                # print('\nDelivered {}!'.format(obj.full_name))
                return Event(playerA=agent.name, event=f'Deliver_{obj.full_name}', location=gs.location,
                             time=current_time)

        # if bin in front --> drop foods into bin
        elif isinstance(gs, Bin):
            world.remove(agent.holding)
            objs = []
            if 'Plate' in agent.holding.full_name:
                objs.append(agent.holding.unmerge('Plate'))
            if 'FireExtinguisher' in agent.holding.full_name:
                objs.append(agent.holding.unmerge('FireExtinguisher'))

            holding_name = agent.holding.full_name

            agent.release()
            # if has a plate in hand, keep plate in hand
            if len(objs) > 0:
                agent.acquire(Object(agent.location, objs))
                world.insert(agent.holding)

            if holding_name != "":
                return Event(playerA=agent.name, event=f'Drop_{holding_name}', location=gs.location, time=current_time)


        # if food/plate tile in front --> get food/plate
        elif isinstance(gs, Tile):
            # Get object from tile
            obj = gs.release()

            if mergeable(agent.holding, obj):
                world.remove(agent.holding)
                agent.acquire(obj)
                world.insert(agent.holding)
                return Event(playerA=agent.name, event=f'Assemble_{agent.holding.full_name}', location=gs.location,
                             time=current_time)
            # if hold a plate before plate tile --> put the plate into the tile
            elif agent.holding.name == "Plate" and isinstance(gs, PlateTile):
                world.remove(agent.holding)
                agent.release()
                return Event(playerA=agent.name, event=f'Put_Plate_on_PlateTile', location=gs.location,
                             time=current_time)

        # if gs is Pot --> try cooking
        elif isinstance(gs, Pot):
            if not world.is_occupied(gs.location):
                # if Pot is not occupies; if cookable -> cook; else do nothing
                if agent.holding.is_cookable():
                    food, plate = agent.holding.split_food_plate()
                    world.remove(agent.holding)
                    agent.release()
                    obj_food = Object(location=gs.location, contents=food)
                    # print("[Pot] cook", obj_food.full_name)
                    event = Event(playerA=agent.name, event=f'Cook_{obj_food.full_name}', location=gs.location,
                                  time=current_time)
                    gs.acquire(obj_food)
                    obj_food.cook(current_time)
                    world.insert(obj_food)
                    if plate is not None:
                        obj_plate = Object(location=agent.location, contents=plate)
                        agent.acquire(obj_plate)
                        world.insert(obj_plate)
                    return event
            else:
                if agent.holding.contents[0] == Plate() and len(agent.holding.contents) == 1:
                    # agent is holding an empty plate; try to get cooked food from pot
                    obj = world.get_object_at(gs.location, None, find_held_objects=False)
                    if obj.is_cooked() and not obj.is_onfire():
                        gs.release()
                        world.remove(obj)
                        world.remove(agent.holding)
                        agent.holding.merge(obj)
                        world.insert(agent.holding)
                        # print("[Pot] pickup", obj.full_name)
                        return Event(playerA=agent.name, event=f'Pickup_{obj.full_name}_from_{gs.name}',
                                     location=gs.location, time=current_time)
                if agent.holding.contents[0] == FireExtinguisher() and len(agent.holding.contents) == 1:
                    # agent is holding a fire extinguisher
                    obj = world.get_object_at(gs.location, None, find_held_objects=False)
                    if 'Fire' in obj.full_name:
                        fire = [_ for _ in obj.contents if isinstance(_, Fire)][0]
                        fire.putout(agent.name, current_time)
                        return Event(playerA=agent.name, event=f'Putout_Fire', location=gs.location, time=current_time)

        # if occupied gridsquare in front --> try merging
        elif world.is_occupied(gs.location):
            # Get object on gridsquare/counter
            obj = world.get_object_at(gs.location, None, find_held_objects=False)

            if mergeable(agent.holding, obj):
                agent_has_plate = (Plate() in agent.holding.contents)

                world.remove(obj)
                o = gs.release()  # agent is holding object
                world.remove(agent.holding)
                agent.acquire(obj)
                world.insert(agent.holding)

                merged_name = agent.holding.full_name

                # if agent has plate --> keep the holding; else the object should be on the counter
                if agent_has_plate:
                    return Event(playerA=agent.name, event=f'Assemble_{merged_name}', location=gs.location,
                                 time=current_time)
                else:
                    gs.acquire(agent.holding)
                    agent.release()
                    return Event(playerA=agent.name, event=f'Assemble_{merged_name}', location=gs.location,
                                 time=current_time)
            elif Plate() in obj.contents and Plate() in agent.holding.contents:
                # if unable to merge due to two plates in conflict
                world.remove(obj)
                plate0 = obj.unmerge('Plate')
                world.insert(obj)
                agent_has_others = any([_ != Plate() for _ in agent.holding.contents])

                if mergeable(agent.holding, obj):
                    world.remove(obj)
                    o = gs.release()  # agent is holding object
                    world.remove(agent.holding)
                    agent.acquire(obj)
                    world.insert(agent.holding)

                    merged_name = agent.holding.full_name

                    # if agent has others --> put the contents on plate; else get the contents from the plate
                    if not agent_has_others:
                        obj = Object(location=gs.location, contents=plate0)
                        world.insert(obj)
                        return Event(playerA=agent.name, event=f'Assemble_{merged_name}', location=gs.location,
                                     time=current_time)
                    else:
                        gs.acquire(agent.holding)
                        agent.release()
                        agent.acquire(Object(location=agent.location, contents=plate0))
                        world.insert(agent.holding)
                        return Event(playerA=agent.name, event=f'Assemble_{merged_name}', location=gs.location,
                                     time=current_time)
                else:
                    world.remove(obj)
                    obj.merge(plate0)
                    world.insert(obj)

        # if holding something, empty gridsquare in front --> chop or drop
        elif not world.is_occupied(gs.location):
            event = None
            obj = agent.holding
            if isinstance(gs, Cutboard):
                try:
                    needs_chopped = obj.needs_chopped()
                    if needs_chopped:
                        event = Event(playerA=agent.name, event=f'Chop_{obj.full_name}', location=gs.location,
                                      time=current_time)
                        obj.chop(current_time)
                except AttributeError:
                    pass
            gs.acquire(obj)  # obj is put onto gridsquare
            if event is None:
                event = Event(playerA=agent.name, event=f'Put_{obj.full_name}_on_{gs.name}', location=gs.location,
                              time=current_time)
            agent.release()
            assert (world.get_object_at(gs.location, obj, find_held_objects=False).is_held == False), \
                ("Verifying put down works", obj.location, obj.is_held, obj.full_name, gs.location, gs.full_name)
            return event

    # if not holding anything
    elif agent.holding is None:
        # not empty in front --> pick up
        if isinstance(gs, Pot) or isinstance(gs, Delivery) or isinstance(gs, Bin):
            pass

        elif world.is_occupied(gs.location) and not isinstance(gs, Delivery) and not isinstance(gs, Bin):
            obj = world.get_object_at(gs.location, None, find_held_objects=False)
            # if in playable game mode, then chop raw items on cutting board
            try:
                needs_chopped = obj.needs_chopped()
            except AttributeError:
                needs_chopped = False
            if isinstance(gs, Cutboard) and needs_chopped:
                obj.chop(current_time)
            else:
                gs.release()
                agent.acquire(obj)
                return Event(playerA=agent.name, event=f'Pickup_{obj.full_name}_from_{gs.name}', location=gs.location,
                             time=current_time)

        # if tile in front --> get object from tile
        elif isinstance(gs, Tile):
            # Get object from tile
            obj = gs.release()

            agent.acquire(obj)
            world.insert(agent.holding)
            return Event(playerA=agent.name, event=f'Pickup_{obj.full_name}_from_{gs.name}', location=gs.location,
                         time=current_time)

        # if empty in front --> interact
        elif not world.is_occupied(gs.location):
            pass
    return Event(playerA=agent.name, event='No-op', location=agent.location, time=current_time)
    # return None
