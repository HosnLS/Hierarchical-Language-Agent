# Recipe planning
import random
from pathlib import Path

import gym_cooking.recipe_planner.recipe as RECIPY

# Other core modules
import gym_cooking
from gym_cooking.utils.interact import interact
from gym_cooking.utils.world import World
from gym_cooking.utils.core import *
from gym_cooking.utils.agent import SimAgent
from gym_cooking.utils.agent import COLORS
from gym_cooking.utils.order_schedule import OrderScheduler
from gym_cooking.utils.event import get_all_events

import copy
import networkx as nx
import numpy as np
from itertools import combinations, product
from collections import namedtuple

import gym
import os
from copy import deepcopy
from dataclasses import dataclass

CollisionRepr = namedtuple("CollisionRepr", "time agent_names agent_locations")

@dataclass
class MapSetting:
    level: str
    user_recipy: bool = True    # whether user can see the recipy
    ai_recipy: bool = True      # whether ai can see the recipy
    max_num_timesteps: int = 100    # max number of timesteps
    max_num_orders: int = 3         # max number of orders

    num_agents: int = 2  # fixed

class OvercookedEnvironment(gym.Env):
    """Environment object for Overcooked."""

    def __init__(self, arglist):
        super().__init__()

        self.arglist = arglist
        self.debug = False
        if hasattr(arglist, 'debug'):
            self.debug = arglist.debug
        if self.debug:
            arglist.record = True
        self.t = 0
        self.set_filename()

        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False

        # For event logging
        self._event_history = []
        self._EVENT_HISTORY_MAX_LEN = 200

        # changeable
        self.chg_grid = None
        self.chg_pos = None
        self.chg_rand_list = []

    def get_repr(self):
        return self.world.get_repr() + tuple([agent.get_repr() for agent in self.sim_agents])

    def __str__(self):
        # Print the world and agents.
        _display = list(map(lambda x: ''.join(
            map(lambda y: y + ' ', x)), self.rep))
        return '\n'.join(_display)

    def __eq__(self, other):
        return self.get_repr() == other.get_repr()

    def __copy__(self):
        new_env = OvercookedEnvironment(self.arglist)
        new_env.__dict__ = self.__dict__.copy()
        new_env.world = copy.copy(self.world)
        new_env.order_scheduler = copy.copy(self.order_scheduler)
        new_env.sim_agents = [copy.copy(a) for a in self.sim_agents]
        new_env.distances = self.distances

        # Make sure new objects and new agents' holdings have the right pointers.
        for a in new_env.sim_agents:
            if a.holding is not None:
                a.holding = new_env.world.get_object_at(
                    location=a.location,
                    desired_obj=None,
                    find_held_objects=True)
        return new_env

    def set_filename(self):
        # self.filename = "{}_agents{}_seed{}".format(self.arglist.level, \
        #                                             self.arglist.num_agents, self.arglist.seed)
        # if self.debug:
        #     self.filename = "debug_" + self.filename
        pass

    def load_level(self, level, num_agents):
        x = 0
        y = 0
        with open(os.path.join(Path(gym_cooking.__file__).absolute().parent, 'utils', 'levels', '{}.txt'.format(level)), 'r') as file:
            # Mark the phases of reading.
            phase = 1
            for line in file:
                line = line.strip('\n')
                if line == '':
                    phase += 1

                # Phase 1: Read in kitchen map.
                elif phase == 1:
                    for x, rep in enumerate(line):
                        # Object, i.e. Tomato, Lettuce, Onion, or Plate.
                        if rep in 'tlopf':
                            counter = Counter(location=(x, y))
                            obj = Object(
                                location=(x, y),
                                contents=RepToClass[rep]())
                            counter.acquire(obj=obj)
                            self.world.insert(obj=counter)
                            self.world.insert(obj=obj)
                        # GridSquare, i.e. Floor, Counter, Cutboard, Delivery, Bin
                        elif rep in RepToClass:
                            newobj = RepToClass[rep]((x, y))
                            self.world.objects.setdefault(
                                newobj.name, []).append(newobj)
                        else:
                            # Empty. Set a Floor tile.
                            f = Floor(location=(x, y))
                            self.world.objects.setdefault(
                                'Floor', []).append(f)
                    y += 1
                # Phase 2: Read in recipe list.
                elif phase == 2:
                    self.recipes.append(getattr(RECIPY, line)())

                # Phase 3: Read in agent locations (up to num_agents).
                elif phase == 3:
                    if len(self.sim_agents) < num_agents:
                        loc = line.split(' ')
                        sim_agent = SimAgent(
                            name='agent-' + str(len(self.sim_agents) + 1),
                            id_color=COLORS[len(self.sim_agents)],
                            location=(int(loc[0]), int(loc[1])))
                        self.sim_agents.append(sim_agent)
                elif phase == 4:
                    self.chg_grid = line
                    self.chg_pos = []
                elif phase == 5:
                    pos = line.split(' ')
                    self.chg_pos.append((int(pos[0]), int(pos[1])))

        if self.chg_grid is not None:
            self.chg_rand_index = 0
            self.chg_rand_list = [random.randint(
                0, len(self.chg_pos)-1) for _ in range(101)]
            self.process_chg()

        self.distances = {}
        self.world.width = x + 1
        self.world.height = y
        self.world.perimeter = 2 * (self.world.width + self.world.height)

        self.all_events = get_all_events(self.recipes)

    def reset(self):
        self.world = World(arglist=self.arglist)
        self.recipes = []
        self.sim_agents = []
        self.agent_actions = {}
        self.interact_history = []
        self.t = 0
        self.current_time = 0.

        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False

        # Load world & distances.
        self.load_level(
            level=self.arglist.level,
            num_agents=self.arglist.num_agents)
        self.order_scheduler = OrderScheduler(self.arglist, self.recipes)
        self.world_size = (self.world.width, self.world.height)
        self.world.make_loc_to_gridsquare()
        self.world.make_reachability_graph()
        self.cache_distances()
        # self.obs_tm1 = copy.copy(self)

        self.state = state = self.get_current_state()
        return state
        # return copy.copy(self)

    def close(self):
        return

    def step(self, action_dict, passed_time=1.):
        # Track internal environment info.
        self.t += 1
        self.current_time += passed_time
        if self.debug:
            print("===============================")
            print("[environment.step] @ TIMESTEP {}".format(self.t))
            print("===============================")

        # Get actions.
        for sim_agent in self.sim_agents:
            sim_agent.action = action_dict[sim_agent.name]

        # Check collisions.
        self.check_collisions()
        # self.obs_tm1 = copy.copy(self)

        # Execute.
        events = self.execute_navigation()
        for event in events:
            self._event_history.append(event)
            if len(self._event_history) > self._EVENT_HISTORY_MAX_LEN:
                self._event_history.pop(0)
            if event.event not in self.all_events:
                print("Invalid event detected: {}".format(event.event))

        # Update Orders
        self.order_scheduler.update(self.world, passed_time=passed_time)

        # Clear delivery
        self.clear_delivery()

        # Update object states, including cooking soup, fire etc.
        self.update_object_states()

        # Visualize.
        # self.display()
        # self.print_agents()

        # Get a plan-representation observation.
        # new_obs = copy.copy(self)
        # Get an image observation
        # image_obs = self.game.get_image_obs()

        if self.debug:
            for o in self.world.get_object_list():
                if isinstance(o, Object):
                    print("    ", o.full_name, o.location)
                else:
                    print("    ", o.name, o.location)

        self.state = state = self.get_current_state()
        done = self.done()
        reward = self.reward()
        '''info = {"t": self.t, "obs": new_obs,
                "image_obs": image_obs,
                "done": done, "termination_info": self.termination_info}'''
        info = {"t": self.t, "done": done,
                "termination_info": self.termination_info, "events": events}
        return state, reward, done, info

    def done(self):
        # Done if the episode maxes out
        if self.current_time >= self.arglist.max_num_timesteps and self.arglist.max_num_timesteps:
            self.termination_info = "Terminating because passed {} timesteps".format(
                self.arglist.max_num_timesteps)
            self.successful = False
            return True

        return False

    def reward(self):
        # return 1 if self.successful else 0
        reward = self.order_scheduler.consume_reward()
        return reward

    def print_agents(self):
        for sim_agent in self.sim_agents:
            sim_agent.print_status()

    def display(self):
        self.update_display()
        print(str(self))

    def update_display(self):
        self.rep = self.world.update_display()
        for agent in self.sim_agents:
            x, y = agent.location
            self.rep[y][x] = str(agent)

    def get_agent_names(self):
        return [agent.name for agent in self.sim_agents]

    def clear_delivery(self):
        delivery_list = list(
            filter(lambda o: o.name == 'Delivery', self.world.get_object_list()))
        for delivery in delivery_list:
            o = delivery.release()
            while o:
                self.world.remove(o)
                o = delivery.release()

    def update_object_states(self):
        '''update the state for dynamic objects
        '''

        # update agents puting out fire
        for pot in self.world.get_all_gridsquares('Pot'):
            if pot.holding is not None and (
                    pot.holding.is_cooking() or pot.holding.is_cooked()) and 'Fire' in pot.holding.full_name:
                fire = [
                    x for x in pot.holding.contents if isinstance(x, Fire)][0]
                for agent in self.sim_agents:
                    if agent.action == (0, 0):
                        continue
                    action_x, action_y = self.world.inbounds(
                        tuple(np.asarray(agent.location) + np.asarray(agent.action)))
                    if action_x != pot.location[0] or action_y != pot.location[1]:
                        fire.remove_agent_putout_state(agent.name)

        # update cooking or cooked objects
        pots = self.world.get_all_gridsquares('Pot')
        for pot in pots:
            if pot.holding is not None and (pot.holding.is_cooking() or pot.holding.is_cooked()):
                self.world.remove(pot.holding)
                pre_update_name = pot.holding.full_name
                pot.holding.update_state(self.current_time)
                after_update_name = pot.holding.full_name
                self.world.insert(pot.holding)

                # the food turns into fire
                if 'Charred' not in pre_update_name and 'Charred' in after_update_name:
                    fire = Fire()
                    self.world.remove(pot.holding)
                    pot.holding.contents.append(fire)
                    self.world.insert(pot.holding)

        # remove fire
        for pot in self.world.get_all_gridsquares('Pot'):
            if pot.holding is not None and (
                    pot.holding.is_cooking() or pot.holding.is_cooked()) and 'Fire' in pot.holding.full_name:
                fire = [
                    x for x in pot.holding.contents if isinstance(x, Fire)][0]
                if fire.is_finished:
                    self.world.remove(pot.holding)
                    pot.holding.contents.remove(fire)
                    self.world.insert(pot.holding)

    def is_collision(self, agent1_loc, agent2_loc, agent1_action, agent2_action):
        """Returns whether agents are colliding.

        Collisions happens if agent collide amongst themselves or with world objects."""
        # Tracks whether agents can execute their action.
        execute = [True, True]

        # Collision between agents and world objects.
        agent1_next_loc = tuple(np.asarray(
            agent1_loc) + np.asarray(agent1_action))
        if self.world.get_gridsquare_at(location=agent1_next_loc).collidable:
            # Revert back because agent collided.
            agent1_next_loc = agent1_loc

        agent2_next_loc = tuple(np.asarray(
            agent2_loc) + np.asarray(agent2_action))
        if self.world.get_gridsquare_at(location=agent2_next_loc).collidable:
            # Revert back because agent collided.
            agent2_next_loc = agent2_loc

        if agent1_next_loc == agent2_loc:
            execute[0] = False

        if agent2_next_loc == agent1_loc:
            execute[1] = False

        # Inter-agent collision.
        if agent1_next_loc == agent2_next_loc:
            if agent1_next_loc == agent1_loc and agent1_action != (0, 0):
                execute[1] = False
            elif agent2_next_loc == agent2_loc and agent2_action != (0, 0):
                execute[0] = False
            else:
                execute[0] = False
                execute[1] = False

        # Prevent agents from swapping places.
        elif ((agent1_loc == agent2_next_loc) and
              (agent2_loc == agent1_next_loc)):
            execute[0] = False
            execute[1] = False
        return execute

    def process_chg(self):
        objs = self.world.get_all_gridsquares(self.chg_grid)
        if len(objs) != 0:
            obj = objs[-1]
            self.world.remove(obj)
            self.world.insert(Counter(obj.location))

        pos = self.chg_pos[self.chg_rand_list[self.chg_rand_index]]
        self.chg_rand_index = (self.chg_rand_index +
                               1) % len(self.chg_rand_list)
        objs = self.world.get_object_list()
        objs = [o for o in objs if o.location == pos]
        for o in objs:
            self.world.remove(o)
        mapping = {
            "FreshLettuceTile": LettuceTile,
            "FreshOnionTile": OnionTile,
            "FreshTomatoTile": TomatoTile
        }
        self.world.insert(mapping[self.chg_grid](pos))

    def assign_chg_rand_list(self, l):
        if self.chg_grid is None:
            return
        self.chg_rand_index = 0
        self.chg_rand_list = l
        self.process_chg()

    def check_collisions(self):
        """Checks for collisions and corrects agents' executable actions.

        Collisions can either happen amongst agents or between agents and world objects."""
        execute = [True for _ in self.sim_agents]

        # Check each pairwise collision between agents.
        for i, j in combinations(range(len(self.sim_agents)), 2):
            agent_i, agent_j = self.sim_agents[i], self.sim_agents[j]
            exec_ = self.is_collision(
                agent1_loc=agent_i.location,
                agent2_loc=agent_j.location,
                agent1_action=agent_i.action,
                agent2_action=agent_j.action)

            # Update exec array and set path to do nothing.
            if not exec_[0]:
                execute[i] = False
            if not exec_[1]:
                execute[j] = False

            # Track collisions.
            if not all(exec_):
                collision = CollisionRepr(
                    time=self.t,
                    agent_names=[agent_i.name, agent_j.name],
                    agent_locations=[agent_i.location, agent_j.location])
                self.collisions.append(collision)

        # print('\nexecute array is:', execute)

        # Update agents' actions if collision was detected.
        for i, agent in enumerate(self.sim_agents):
            if not execute[i]:
                agent.action = (0, 0)
            # print("{} has action {}".format(color(agent.name, agent.color), agent.action))

    def execute_navigation(self):
        events = []
        for agent in self.sim_agents:
            result = interact(agent=agent, world=self.world,
                              current_time=self.current_time)
            if result.event != 'No-op':
                self.interact_history.append(result)
                events.append(result)
            self.agent_actions[agent.name] = agent.action
            if self.chg_grid is not None and self.chg_grid in result.event:
                self.process_chg()
        self.last_step_events = copy.deepcopy(events)
        return events

    def cache_distances(self):
        """Saving distances between world objects."""
        counter_grid_names = [name for name in self.world.objects if
                              "Supply" in name or "Counter" in name or "Delivery" in name or "Cut" in name]
        # Getting all source objects.
        source_objs = copy.copy(self.world.objects["Floor"])
        for name in counter_grid_names:
            source_objs += copy.copy(self.world.objects[name])
        # Getting all destination objects.
        dest_objs = source_objs

        # From every source (Counter and Floor objects),
        # calculate distance to other nodes.
        for source in source_objs:
            self.distances[source.location] = {}
            # Source to source distance is 0.
            self.distances[source.location][source.location] = 0
            for destination in dest_objs:
                # Possible edges to approach source and destination.
                source_edges = [
                    (0, 0)] if not source.collidable else World.NAV_ACTIONS
                destination_edges = [
                    (0, 0)] if not destination.collidable else World.NAV_ACTIONS
                # Maintain shortest distance.
                shortest_dist = np.inf
                for source_edge, dest_edge in product(source_edges, destination_edges):
                    try:
                        dist = nx.shortest_path_length(self.world.reachability_graph, (source.location, source_edge),
                                                       (destination.location, dest_edge))
                        # Update shortest distance.
                        if dist < shortest_dist:
                            shortest_dist = dist
                    except:
                        continue
                # Cache distance floor -> counter.
                self.distances[source.location][destination.location] = shortest_dist

        # Save all distances under world as well.
        self.world.distances = self.distances

    def get_current_state(self):
        # gridsquare
        gridsquare_map = {
            gs: np.zeros(self.world_size, dtype=np.uint8) for gs in GRIDSQUARES
        }
        objs = []

        # gridsquares
        for o_list in self.world.objects.values():
            for o in o_list:
                if isinstance(o, GridSquare):
                    gridsquare_map[o.name][o.location[0], o.location[1]] = 1
                else:
                    objs.append(o)

        # objects
        ALL_ENTITIES = FRESH_FOOD + CHOPPING_FOOD + CHOPPED_FOOD + COOKING_FOOD + COOKED_FOOD + CHARRED_FOOD + ['Plate',
                                                                                                                'Fire',
                                                                                                                'FireExtinguisher']
        ALL_FOOD_ENTITIES = FRESH_FOOD + CHOPPED_FOOD + \
            COOKED_FOOD + CHARRED_FOOD + ['Plate']
        obj_map = {
            e: np.zeros(self.world_size, dtype=np.uint8) for e in ALL_ENTITIES
        }
        for o in objs:
            for c in o.contents:
                obj_map[c.full_name][o.location[0], o.location[1]] = 1
        obj_map['cooking_rest_time'] = np.zeros(
            self.world_size, dtype=np.float32)
        for pot in self.world.get_all_gridsquares('Pot'):
            if pot.holding is not None and pot.holding.is_cooking():
                obj_map['cooking_rest_time'][
                    pot.location[0], pot.location[1]] = pot.holding.rest_cooking_time() / COOKING_TIME_SECONDS
        obj_map['chopping_num_steps'] = np.zeros(
            self.world_size, dtype=np.float32)
        for cutboard in self.world.get_all_gridsquares('Cutboard'):
            if cutboard.holding is not None and cutboard.holding.full_name.startswith('Chopping'):
                obj_map['chopping_num_steps'][cutboard.location[0], cutboard.location[1]] = cutboard.holding.contents[
                    0].state._rest_steps / CHOPPING_NUM_STEPS

        # agents
        agent_map = {
            agent.name: np.zeros(self.world_size, dtype=np.uint8) for agent in self.sim_agents
        }
        for agent in self.sim_agents:
            agent_map[agent.name][agent.location[0], agent.location[1]] = 1

        # agent location and holdings
        agent_data = {
            agent.name: {
                "location": agent.location
            } for agent in self.sim_agents
        }
        current_holdings = {}
        for agent in self.sim_agents:
            if agent.holding is None:
                agent_data[agent.name]['holding'] = []
                agent_data[agent.name]['holding_onehot'] = current_holdings[agent.name] = np.zeros(
                    (len(ALL_FOOD_ENTITIES),), dtype=np.uint8)
            else:
                agent_data[agent.name]['holding'] = agent.holding.full_name.split(
                    '-')
                agent_data[agent.name]['holding_onehot'] = current_holdings[agent.name] = np.array(
                    [int(e in agent.holding.full_name) for e in ALL_FOOD_ENTITIES])

        # current orders
        current_orders = []
        for order, restTime, timeLimit, bonus in self.order_scheduler.current_orders:
            order_onehot = np.array([int(e in order.full_name)
                                    for e in ALL_FOOD_ENTITIES])
            current_orders.append((order_onehot, restTime / timeLimit))
        current_orders_np = np.concatenate(
            [np.concatenate([order_onehot, np.array([t])]) for order_onehot, t in current_orders])

        '''result = {
            'gridsquare': gridsquare_map,
            'objects': obj_map,
            'agent_locations': agent_map,
            'agents': agent_data,
            'current_orders': current_orders
        }'''
        result = [v for k, v in gridsquare_map.items()] + [v for k, v in obj_map.items()] + [v for k, v in
                                                                                             agent_map.items()]
        result = {
            'map': np.asarray(result),
            'current_orders': current_orders_np,
            'current_holdings': current_holdings,
        }
        return result

    def get_all_events(self):
        return copy.deepcopy(self.all_events)

    def get_ai_info(self):
        order_scheduler = deepcopy(self.order_scheduler)
        if not self.arglist.ai_recipy:
            order_scheduler.current_orders = []
        return {"world": self.world,
                "sim_agents": self.sim_agents,
                "order_scheduler": order_scheduler,
                "event_history": self._event_history,
                "current_time": self.current_time,
                "chg_grid": self.chg_grid}
