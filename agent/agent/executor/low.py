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


def bfs_search(grid: list[list[int]], start: tuple[int, int], end: list[tuple[int, int]]):
    # used for calculating path for agent
    # input: grid: True for available, False for unavailable
    #        start: start position
    #        end: end position(s)
    # output: distance, -1 if not found
    #         move: move sequence, from next_pos to end_pos

    point = namedtuple('point', ['x', 'y', 'parent'])
    queue = [point(start[0], start[1], None)]
    visited = set()
    visited.add(start)
    curr = point(start[0], start[1], None)
    while queue:
        curr = queue.pop(0)
        if (curr.x, curr.y) in end:  # wrong when heading multiple end point
            break
        for x_inc, y_inc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # wrong when heading multiple end point
            x = curr.x + x_inc
            y = curr.y + y_inc
            if ((x, y) in end or grid[x][y]) and (x, y) not in visited:
                queue.append(point(x, y, curr))
                visited.add((x, y))
    # prepare result
    if (curr.x, curr.y) not in end:
        return -1, None
    move = []
    while curr.parent:
        move.append((curr.x, curr.y))
        curr = curr.parent
    move.reverse()
    return len(move), move

def bfs_search_all(grid: list[list[int]], start: tuple[int, int]):
    # used for calculating path for agent
    # input: grid: True for available, False for unavailable
    #        start: start position
    #        end: end position(s)
    # output: distance, -1 if not found
    #         move: move sequence, from next_pos to end_pos
    result = [[[None, -1] for _ in range(len(grid[0]))] for _ in range(len(grid))]
    visited = [[False for _ in range(len(grid[0]))] for _ in range(len(grid))]

    point = namedtuple('point', ['x', 'y', 'move', 'dist'])
    queue = []

    result[start[0]][start[1]] = [None, -1]
    visited[start[0]][start[1]] = True

    for x_inc, y_inc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # wrong when heading multiple end point
        x = start[0] + x_inc
        y = start[1] + y_inc
        result[x][y] = [(x, y), 1]
        visited[x][y] = True

        if grid[x][y]:
            queue.append(point(x, y, (x_inc, y_inc), 1))

    while queue:
        curr = queue.pop(0)
        for x_inc, y_inc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # wrong when heading multiple end point
            x = curr.x + x_inc
            y = curr.y + y_inc
            result[x][y] = [(x, y), curr.dist]
            if grid[x][y] and not visited[x][y]:
                visited[x][y] = True
                queue.append(point(x, y, curr.move, curr.dist + 1))

    return result


def bfs_reachable(grid: list[list[int]], start: tuple[int, int]) -> list[list[bool]]:
    # used for calculating path for agent
    # input: grid: True for available, False for unavailable
    #        start: start position
    #        end: end position(s)
    # output: distance, -1 if not found
    #         move: move sequence, from next_pos to end_pos
    reach_map = [[False for _ in range(len(grid[0]))] for _ in range(len(grid))]
    point = namedtuple('point', ['x', 'y'])
    queue = [point(start[0], start[1])]
    visited = set()
    visited.add(start)
    reach_map[start[0]][start[1]] = True
    while queue:
        curr = queue.pop(0)
        for x_inc, y_inc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # wrong when heading multiple end point
            x = curr.x + x_inc
            y = curr.y + y_inc
            reach_map[x][y] = True
            if grid[x][y] and (x, y) not in visited:
                queue.append(point(x, y))
                visited.add((x, y))
    # prepare result
    return reach_map


def match_any(inner: list[str] | str | None, outer: list[str] | str | None) -> bool:
    # check whether the specified object is in the specified position
    if outer is None:
        return True
    if inner is None:
        return outer is None
    if isinstance(inner, str):
        inner = [inner]
    if isinstance(outer, str):
        outer = [outer]
    return any([i in outer for i in inner])


def fname(obj: Object | None) -> str:
    # get the full name of the object
    if obj is None:
        return "Nothing"
    else:
        return obj.full_name


def fname_content(obj: Object | None) -> list[str]:
    # get the full name of the object and its contents
    if obj is None:
        return ["Nothing"]
    else:
        ret = [obj.full_name]
        if hasattr(obj, 'contents'):
            for o in obj.contents:
                ret += fname_content(o)
        return ret


class EnvState:
    # a class to store the environment state
    def __init__(self, world: World,
                 agents: list[SimAgent],
                 agent_idx: int,
                 order: OrderScheduler,
                 event_history: list[Event],
                 chg_grid,
                 time: float):
        self.world_all = world.get_object_list()
        self.world_width, self.world_height = world.width, world.height
        self.agents = agents
        self.agent_idx = agent_idx
        self.order = order
        self.event_history = event_history
        self.chg_grid = chg_grid
        self.time = time

        # pos to obj/gs
        self.pos_obj = defaultdict(lambda: None)
        self.pos_gs = defaultdict(lambda: None)
        for o in self.world_all:
            if isinstance(o, GridSquare):
                self.pos_gs[o.location] = o
            else:
                self.pos_obj[o.location] = o

        # all objs
        self.all_obj = [a for a in self.world_all if isinstance(a, Object) and not a.is_held]
        self.all_obj_h = self.all_obj + ([self.hold] if self.hold is not None else [])
        self.all_obj_a = self.all_obj + [a.holding for a in self.agents if a.holding is not None]

        # to grid
        self.to_grid = [[1 for _ in range(self.world_height)] for _ in range(self.world_width)]
        for obj in self.world_all:
            if obj.collidable: self.to_grid[obj.location[0]][obj.location[1]] = 0
        self.to_grid_a = deepcopy(self.to_grid)
        for agent in self.agents[:self.agent_idx] + self.agents[self.agent_idx + 1:]:
            self.to_grid_a[agent.location[0]][agent.location[1]] = 0

        # reachable map
        self.rch_map = bfs_reachable(self.to_grid, self.self_pos)
        self.rch_obj = [o for o in self.all_obj if self.rch_map[o.location[0]][o.location[1]]]
        self.rch_obj_h = self.rch_obj + ([self.hold] if self.hold is not None else [])
        self.rch_grid = [a for a in self.world_all if isinstance(a, GridSquare)
                         and self.rch_map[a.location[0]][a.location[1]]]

        # bfs search
        self.bfs_search_a = bfs_search_all(self.to_grid_a, self.self_pos)
        self.bfs_search = bfs_search_all(self.to_grid, self.self_pos)

    @property
    def self_pos(self):
        return self.agents[self.agent_idx].location

    @property
    def hold(self) -> Object | None:
        # output: object held by agent
        return self.agents[self.agent_idx].holding

    def get_pos_by_obj_gs(
            self,
            obj: list[str] | str | None = None,
            gs: list[str] | str | None = None,
            inner: bool = False
    ) -> list:
        # input:  obj: object name, None for any, str for exact match, 'Nothing' & ['Nothing'] for clear obj, None for any
        #         gs: goal name, None for any
        #         inner: whether to consider contents
        # output: list of positions, no repeat

        # 1 formalize
        if isinstance(obj, str): obj = [obj]
        if isinstance(gs, str): gs = [gs]
        # 2 match
        results = set()
        for pos in set(self.pos_obj.keys()) | set(self.pos_gs.keys()):
            if inner:
                obj_fname = fname_content(self.pos_obj[pos])
            else:
                obj_fname = fname(self.pos_obj[pos])
            if not match_any(obj_fname, obj): continue
            gs_name = "Nothing" if self.pos_gs[pos] is None else self.pos_gs[pos].name
            if not match_any(gs_name, gs): continue
            results.add(pos)
        return list(results)

    def check_pos_by_obj_gs(
            self,
            pos: tuple[int, int],
            obj: list[str] | str | None = None,
            gs: list[str] | str | None = None,
            inner: bool = False
    ) -> bool:
        # input:  pos: position
        #         obj: object name, None for any, str for exact match, 'Nothing' & ['Nothing'] for clear obj
        #         gs: goal name, None for any
        #         inner: whether to consider contents
        # output: list of positions, no repeat

        pos_list = self.get_pos_by_obj_gs(obj, gs, inner)
        return pos in pos_list

    def navigate_pos_by_obj_gs(
            self,
            obj: list[str] | str | None = None,
            gs: list[str] | str | None = None,
            inner: bool = False,
            agent: bool = False
    ) -> tuple[int, int] | None:
        # 1 get possible positions
        pos_list = self.get_pos_by_obj_gs(obj, gs, inner)
        if not pos_list: return None
        # 2 get closeest object regarding other agents
        moves = [self.bfs_search_a[pos[0]][pos[1]] for pos in pos_list]
        moves = [m for m in moves if m[1] > 0]
        moves.sort(key=lambda x:x[1])
        if len(moves) > 0:
            return moves[0][0]
        if agent:
            return None
        # 3 get closest object regardless other objects
        moves = [self.bfs_search[pos[0]][pos[1]] for pos in pos_list]
        moves = [m for m in moves if m[1] > 0]
        moves.sort(key=lambda x: x[1])
        if len(moves) > 0:
            return moves[0][0]
        else:
            return None

    def get_all_grid_info(self) -> list:
        res: list = []
        for pos in self.pos_gs.keys():
            res.append({
                "pos": pos,
                "gs": self.pos_gs[pos],
                "obj": self.pos_obj[pos],
                "rch": self.rch_map[pos[0]][pos[1]]
            })

        return res

# Low Task: Approach (LTApproach) and Interact (LTInteract)

class LowTask:
    Success = 1
    Working = 0

    def __call__(self, *args, **kwargs) -> tuple[int, tuple[int, int]]:
        raise NotImplementedError


class LTApproach(LowTask):
    DestBlock = -1  # blocked by other agent
    DestUnreachable = -2  # not reachable

    def __init__(self, pos: tuple[int, int]):
        self.pos = pos

    def __call__(self, env: EnvState):
        # 1 consider other agents
        grid = env.to_grid_a
        distance, move = bfs_search(grid, env.self_pos, [self.pos])
        if distance > 1:
            m = (move[0][0] - env.self_pos[0], move[0][1] - env.self_pos[1])
            return LTApproach.Working, m
        elif distance == 1:
            return LTApproach.Success, (0, 0)
        # 2 blocked by agent
        grid = env.to_grid
        distance, move = bfs_search(grid, env.self_pos, [self.pos])
        other_agent_pos = [agent.location for agent in env.agents[:env.agent_idx] + env.agents[env.agent_idx + 1:]]
        if distance < 0:
            return LTApproach.DestUnreachable, (0, 0)
        elif move[0] in other_agent_pos:
            return LTApproach.DestBlock, random.choice([(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)])  # prevent, stack
        else:
            m = (move[0][0] - env.self_pos[0], move[0][1] - env.self_pos[1])
            return LTApproach.Working, m


class LTInteract(LowTask):
    DestTooFar = -1  # not close to target

    def __init__(self, pos: tuple[int, int]):
        self.pos = pos

    def __call__(self, env: EnvState):
        # 1 check distance less than 1
        if abs(env.self_pos[0] - self.pos[0]) + abs(env.self_pos[1] - self.pos[1]) > 1:
            return LTInteract.DestTooFar, (0, 0)
        # 2 interact
        return LTInteract.Success, (self.pos[0] - env.self_pos[0], self.pos[1] - env.self_pos[1])
