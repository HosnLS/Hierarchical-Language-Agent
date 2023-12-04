from gym_cooking.utils.config import *
from agent.executor.low import EnvState, bfs_reachable, fname
from agent.executor.high import \
    HighTask, HTChop, HTAssemble, HTPutout, HTCook, HTPick, HTServe, HTDrop, \
    OBJ_TO_GOODS_GS, OBJ_TO_GOODS_POT, ALL_FRESH_FOOD, ALL_ASSEMBLE, ALL_SOUP, HT_MAP

MOVE_TO_HT = \
    {f"{HT_MAP['Chop']} {x}": HTChop(x) for x in ALL_FRESH_FOOD} | \
    {f"{HT_MAP['Assemble']} {x}": HTAssemble(x) for x in ALL_ASSEMBLE} | \
    {f"{HT_MAP['Putout']}": HTPutout()} | \
    {f"{HT_MAP['Cook']} {x}": HTCook(x) for x in ALL_SOUP} | \
    {f"{HT_MAP['Pick']} {x}": HTPick(x) for x in ALL_SOUP} | \
    {f"{HT_MAP['Serve']} {x}": HTServe(x) for x in ALL_SOUP} | \
    {f"{HT_MAP['Drop']}": HTDrop()}

ALL_MOVES = list(MOVE_TO_HT.keys())


def prep_chk_moves(env: EnvState) -> list:
    # moves
    all_moves = []
    for obj in ALL_FRESH_FOOD:
        can_begin = HTChop(obj).can_begin(env)
        all_moves.append([f"{HT_MAP['Chop']} {obj}", *can_begin])
    for obj in ALL_ASSEMBLE:
        can_begin = HTAssemble(obj).can_begin(env)
        all_moves.append([f"{HT_MAP['Assemble']} {obj}", *can_begin])
    can_begin = HTPutout().can_begin(env)
    all_moves.append([f"{HT_MAP['Putout']}", *can_begin])
    for obj in ALL_SOUP:
        can_begin = HTCook(obj).can_begin(env)
        all_moves.append([f"{HT_MAP['Cook']} {obj}", *can_begin])
    for obj in ALL_SOUP:
        can_begin = HTPick(obj).can_begin(env)
        all_moves.append([f"{HT_MAP['Pick']} {obj}", *can_begin])
    for obj in ALL_SOUP:
        can_begin = HTServe(obj).can_begin(env)
        all_moves.append([f"{HT_MAP['Serve']} {obj}", *can_begin])
    can_begin = HTDrop().can_begin(env)
    all_moves.append([f"{HT_MAP['Drop']}", *can_begin])

    for x in all_moves:
        assert x[0] in ALL_MOVES, f'Invalid move {x[0]}'

    return all_moves


def prep_prompt_order(env: EnvState) -> list:
    ret = []
    for rec in env.order.current_orders:
        soup_name = OBJ_TO_GOODS_GS[rec[0].full_name]
        rate = rec[1] / rec[2]
        ret.append({'name': soup_name, 'rate': rate})

    return ret


def prep_prompt_map(env: EnvState) -> str:
    # current map

    prompt = '''
Items on the map:

'''
    # obj gs reachable info
    res = env.get_all_grid_info()

    obj_count = {}
    for a in res:
        if (a["gs"].name == "Counter" or a["gs"].name == "Cutboard") and a["rch"] and a["obj"] is not None \
                or a["obj"] is not None and a["rch"] and a["obj"].is_held:
            obj = OBJ_TO_GOODS_GS[a["obj"].full_name]
            obj_count[obj] = obj_count.get(obj, 0) + 1
    for key, value in obj_count.items():
        prompt += f'- {value} {key}\n'

    # Pot
    num_empty_pot = len([a for a in res if a["gs"].name ==
                        "Pot" and a["obj"] is None and a["rch"]])
    prompt += f'- {num_empty_pot} empty {"pots" if num_empty_pot > 1 else "pot"}\n'
    for a in res:
        if a["gs"].name == "Pot" and a["rch"] and a["obj"] is not None:
            obj = OBJ_TO_GOODS_POT[a["obj"].full_name]
            if 'Cooking' in a["obj"].full_name:
                remain_time = a["obj"].rest_turn_time()
                rate = remain_time / COOKING_TIME_SECONDS
                prompt += f'- 1 pot cooking {obj}: '
                # progress
                if rate > 0.75:
                    prompt += 'just started and far from finished'
                elif rate > 0.5:
                    prompt += 'has been cooking for a while and needs some time to finish'
                elif rate > 0.25:
                    prompt += 'has been cooking for a long time and will finish soon'
                else:
                    prompt += 'will finish in no time'
            elif 'Cooked' in a["obj"].full_name:
                remain_time = a["obj"].rest_turn_time()
                rate = remain_time / COOKED_BEFORE_FIRE_TIME_SECONDS
                prompt += f'- 1 pot with cooked {obj}: '
                if rate > 0.75:
                    prompt += 'just cooked and far from charred'
                elif rate > 0.5:
                    prompt += 'has been cooked for a while and needs some time to get charred'
                elif rate > 0.25:
                    prompt += 'has been cooked for a long time and will get charred soon'
                else:
                    prompt += 'will get charred in no time'
            elif 'Fire' in a["obj"].full_name:
                remain_time = a["obj"].rest_turn_time()
                rate = remain_time / FIRE_PUTOUT_TIME_SECONDS
                prompt += f' - 1 pot on fire with charred {obj}'
                if rate > 0.5:
                    prompt += 'The fire is big and will take some time to put out'
                else:
                    prompt += 'The fire is small and will get put out soon'
            else:
                prompt += f' - 1 pot with charred {obj}'

            prompt += '\n'

    return prompt


def prep_prompt_map_s(env: EnvState) -> str:
    # current map

    prompt = '''
Items on the map:

'''

    res = env.get_all_grid_info()

    for a in res:
        if not a['rch']:
            continue
        if a['gs'].name == "Floor":
            continue
        prompt += f"{a['gs'].name} at {a['gs'].location}"
        if a['gs'].name in ["Counter", "Cutboard"] and a['obj'] is not None:
            obj_name = OBJ_TO_GOODS_GS[a["obj"].full_name]
            prompt += f" with {obj_name} on it"
        elif a['gs'].name == "Pot" and a['obj'] is not None:
            obj = OBJ_TO_GOODS_POT[a["obj"].full_name]
            if 'Cooking' in a["obj"].full_name:
                remain_time = a["obj"].rest_turn_time()
                rate = remain_time / COOKING_TIME_SECONDS
                prompt += f' cooking {obj}: '
                # progress
                if rate > 0.75:
                    prompt += 'just started and far from finished'
                elif rate > 0.5:
                    prompt += 'has been cooking for a while and needs some time to finish'
                elif rate > 0.25:
                    prompt += 'has been cooking for a long time and will finish soon'
                else:
                    prompt += 'will finish in no time'
            elif 'Cooked' in a["obj"].full_name:
                remain_time = a["obj"].rest_turn_time()
                rate = remain_time / COOKED_BEFORE_FIRE_TIME_SECONDS
                prompt += f' with cooked {obj}: '
                if rate > 0.75:
                    prompt += 'just cooked and far from charred'
                elif rate > 0.5:
                    prompt += 'has been cooked for a while and needs some time to get charred'
                elif rate > 0.25:
                    prompt += 'has been cooked for a long time and will get charred soon'
                else:
                    prompt += 'will get charred in no time'
            elif 'Fire' in a["obj"].full_name:
                remain_time = a["obj"].rest_turn_time()
                rate = remain_time / FIRE_PUTOUT_TIME_SECONDS
                prompt += f' on fire with charred {obj}'
                if rate > 0.5:
                    prompt += 'The fire is big and will take some time to put out'
                else:
                    prompt += 'The fire is small and will get put out soon'
            else:
                prompt += f' - 1 pot with charred {obj}'

        prompt += ".\n"

    # self pos and hold
    prompt += f"Currently you are at {env.self_pos}"
    if env.agents[env.agent_idx].holding is not None:
        hold_name = OBJ_TO_GOODS_GS[env.agents[env.agent_idx].holding.full_name]
        prompt += f" holding {hold_name}"
    prompt += '.\n'

    # other pos and hold
    prompt += f"The human player is at {env.agents[1 - env.agent_idx].location}"
    if env.agents[1 - env.agent_idx].holding is not None:
        hold_name = OBJ_TO_GOODS_GS[env.agents[1 -
                                               env.agent_idx].holding.full_name]
        prompt += f" holding {hold_name}"
    prompt += ".\n"

    return prompt


def prep_prompt(env: EnvState, int_hist: list, llm_his: list, mov_his: list, chat: str) -> dict:
    ret = {}
    ret['chk_moves'] = prep_chk_moves(env)
    ret['order'] = prep_prompt_order(env)
    ret['map'] = prep_prompt_map(env)
    ret['int_hist'] = int_hist[-3:]
    ret['llm_hist'] = llm_his[-3:]
    ret['mov_hist'] = mov_his[-100:]
    ret['chatin'] = chat

    return ret


def prep_prompt_s(env: EnvState, int_hist: list, llm_his: list, mov_his: list, chat: str) -> dict:
    ret = {}
    ret['chk_moves'] = prep_chk_moves(env)
    ret['order'] = prep_prompt_order(env)
    ret['map'] = prep_prompt_map_s(env)
    ret['int_hist'] = int_hist[-3:]
    ret['llm_hist'] = llm_his[-3:]
    ret['mov_hist'] = mov_his[-100:]
    ret['chatin'] = chat

    return ret
