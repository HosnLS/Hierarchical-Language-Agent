import json
import os
import html
from collections import defaultdict

import numpy as np

from agent.mind.prompt_local import ALL_MOVES
from agent.mind.llm_api import LLM_LLAMA_LOCAL, LLM_GPT_API
from agent.mind.prompt import prep_mov_hist, \
    prompt_base_Ei, prompt_order_int, prompt_order, prompt_map, \
    prompt_reason_Ei, \
    prompt_base_El_s, prompt_base_El_s2, prompt_base_El_1, \
    prompt_base_El_2, prompt_base_El_22, prompt_base_El_3, prompt_base_Hl_s, \
    prompt_base_El_5


def Ei_prompt(prep: dict) -> list[list[str]]:
    order_prep = prep['order']
    int_hist = prep['int_hist']

    base = prompt_base_Ei()
    order = prompt_order_int(order_prep) + '\n'
    reason = prompt_reason_Ei(int_hist) + '\n'

    base[-1][0] += order
    base[-1][0] += reason

    base[-1][0] += "\n\nNow, your answer is:"

    return base


class chatter:
    MAX_RETRY_TIMES = 6

    def __init__(self, prompt: callable, prep: dict):
        self.init_prompt = prompt
        self.prep = prep
        self.hist = []

        self._res = None
        self._retry = 0

    def __call__(self, text=None):
        if text is not None:
            self.hist[-1].append(text)

        # first round
        if len(self.hist) == 0:
            chat = self.init_prompt(self.prep)
            self.hist = chat
            return None, self.hist
        else:
            self._res = text
            return self._res, None


class El_chatter(chatter):
    def __init__(self, prompt: callable, prep: dict):
        super().__init__(prompt, prep)
        self.has_request = self.prep['chatin'] != "None"

    def check_reasoning(self, text):
        self._res["Reasoning"] = text
        return False, prompt_base_El_2(self.prep)

    def check_chat(self, text):
        self._res["Chat"] = text

        if self.has_request:
            return False, prompt_base_El_3(self.prep)
        else:
            return True, ""

    def check_finished(self, text):
        if "yes" in text.lower():
            self._res["Finished"] = True
        elif "no" in text.lower():
            self._res["Finished"] = False
        else:
            return False, prompt_base_El_3(self.prep)
        return True, None

    def __call__(self, text=None):
        # print(text)
        if text is not None:
            self.hist[-1].append(text)

        # first round
        if len(self.hist) == 0:
            if self.has_request:
                chat = [[prompt_base_El_s(self.prep), "Ok"], [
                    prompt_base_El_1(self.prep)]]
                self._res = {
                    "Reasoning": None,
                    "Chat": None,
                    "Finished": None,
                    # "Demand": None
                }
            else:
                chat = [[prompt_base_El_s2(self.prep), "Ok"], [
                    prompt_base_El_22(self.prep)]]
                self._res = {
                    "Reasoning": "",
                    "Chat": None,
                    "Finished": True,
                    # "Demand": None
                }
            self.hist = chat
            return None, self.hist
        elif self._retry >= self.MAX_RETRY_TIMES:
            self._res = {
                "Reasoning": "ERROR",
                "Chat": "ERROR",
                "Finished": True,
                # "Demand": "ERROR"
            }
            return self._res, None
        else:  # proceed
            if self._res["Reasoning"] is None:
                ok, hint = self.check_reasoning(text)
            elif self._res["Chat"] is None:
                ok, hint = self.check_chat(text)
            elif self._res["Finished"] is None:
                ok, hint = self.check_finished(text)
            else:
                ok, hint = True, None
            if not ok:
                self.hist.append([hint])
                self._retry += 1
                return None, self.hist
            else:
                return self._res, None


class Hl_chatter(chatter):
    def check_reasoning(self, text):
        self._res["Reasoning"] = text
        return False, prompt_base_El_2(self.prep)

    def check_chat(self, text):
        self._res["Chat"] = text
        return False, prompt_base_El_3(self.prep)

    def check_finished(self, text):
        if "yes" in text.lower():
            self._res["Finished"] = True
        elif "no" in text.lower():
            self._res["Finished"] = False
        else:
            return False, prompt_base_El_3(self.prep)
        return False, prompt_base_El_5(self.prep)

    def check_action(self, text):
        text = text.replace('"', '').replace(
            "`", "").replace("'", "").replace(".", "").strip()
        if text not in ALL_MOVES:
            p = f"Action \"{text}\" is not available.\n"
            p += prompt_base_El_5(self.prep)
            print(p)
            return False, p
        self._res["Action"] = text
        return True, None

    def __call__(self, text=None):
        # print(text)
        if text is not None:
            self.hist[-1].append(text)

        # first round
        if len(self.hist) == 0:
            chat = [[prompt_base_Hl_s(self.prep), "Ok"], [
                prompt_base_El_1(self.prep)]]
            self._res = {
                "Reasoning": None,
                "Chat": None,
                "Finished": None,
                "Action": None
            }
            self.hist = chat
            return None, self.hist
        elif self._retry >= self.MAX_RETRY_TIMES:
            self._res = {
                "Reasoning": "ERROR",
                "Chat": "ERROR",
                "Finished": True,
                "Action": ALL_MOVES[0]
            }
            return self._res, None
        else:  # proceed
            if self._res["Reasoning"] is None:
                ok, hint = self.check_reasoning(text)
            elif self._res["Chat"] is None:
                ok, hint = self.check_chat(text)
            elif self._res["Finished"] is None:
                ok, hint = self.check_finished(text)
            elif self._res["Action"] is None:
                ok, hint = self.check_action(text)
            else:
                ok, hint = True, None
            if not ok:
                self.hist.append([hint])
                self._retry += 1
                return None, self.hist
            else:
                return self._res, None


def Em_prompt_ep(prep: dict) -> tuple[list[list[str]], list[str], list[str], dict]:
    chk_moves = prep['chk_moves']
    llm_hist = prep['llm_hist']
    mov_hist = prep['mov_hist']

    available_moves = [m[0] for m in chk_moves if m[1]]

    prompts = []
    q0 = '''Game Situation:
You and another human player are playing a simplified version of the video game Overcooked. Your goal is to cooperatively finish a dynamically changing list of soup orders as fast as possible. The game has different orders from the original video game. There are two players: you (an AI assistant) and another human player. Your primary goal is to cooperate and make the human player feel engaged, happy, and satisfied while also earning more points.

Game Rules:
1. All available actions are: ''' + ', '.join(ALL_MOVES) + '''.
2. There is a changing list of soup orders, each with a time limit for completion. Completing an order on time earns a bonus, while failing to do so results in losing the bonus.
3. The inverse action sequence to finish soup orders:
    To finish Alice Soup order, you need to Serve Alice Soup, which needs Plate Alice Soup and Cook Alice Soup. Alice Soup can be done after you Prepare Alice Ingredients, which needs Chop Lettuce and Chop Onion.
    To finish Bob Soup order, you need to Serve Bob Soup, which needs Plate Bob Soup and Cook Bob Soup. Bob Soup can be done after you Prepare Bob Ingredients, which needs Chop Lettuce and Chop Tomato.
    To finish Cathy Soup order, you need to Serve Cathy Soup, which needs Plate Cathy Soup and Cook Cathy Soup. Cathy Soup can be done after you Prepare Cathy Ingredients, which needs Chop Onion and Chop Tomato.
    To finish David Soup order, you need to Serve David Soup, which needs Plate David Soup and Cook David Soup. David Soup can be done after you Prepare David Ingredients, which needs Chop Lettuce, Chop Onion, and Chop Tomato.
4. If a cooked soup remains in the pot for a long time, it becomes charred, and the pot catches fire.
    a. Putout: To regain the pot, you must extinguish the fire.
    b. Drop: If a soup becomes charred, you must discard it.

Let's say you're playing this game and it's been a while. The human may specify his demand, and maybe you have some planning. Now you need to give your actions based on them.
Please note that when you carry out an action, you just do it once. If you want to do it multiple times, you need to repeat it multiple times. If there is many subtasks in the human's demand, you need to finish them in order. 
If the demand contains "Stop xxx" or "Avoid xxx", you should never do it. 
If the demand contains "Focus xxx", "Keep xxx" or "Always xxx", then you should always do it, and never doing other actions.

'''
    prompts.append([q0, "Ok."])

    qf = ''
    llm = llm_hist[-1]['ret']
    if llm['Demand'] != '':
        qf += f"The human's demand is:\n"
        qf += f"{llm['Demand']}\n"
        qf += '\n'
    if llm['Chat'] != '':
        qf += f"Your planning:\n"
        qf += f"{llm['Chat']}\n"
        qf += '\n'

    chosen_actions = [a['task'] for a in mov_hist]
    chosen_actions = ', '.join(chosen_actions)

    if len(chosen_actions) > 0:
        af = "My actions are: " + chosen_actions
        choices = [f", {m}" for m in available_moves]
    else:
        af = "My actions are: "
        choices = [f"{m}" for m in available_moves]

    prompts.append([qf, af])

    prob_base = defaultdict(lambda: 0)

    ratio = 0.2 if llm['Demand'] != '' else 100.0

    for m in chk_moves:
        prob_base[m[0]] -= m[4] * ratio

    return prompts, choices, available_moves, prob_base


def L1_prompt_ep(prep: dict):
    chk_moves = prep['chk_moves']
    order_prep = prep['order']
    env = prep['map']
    int_hist = prep['int_hist']
    llm_hist = prep['llm_hist']
    mov_hist = prep['mov_hist']
    chat = prep['chatin']

    prompts = []
    q0 = '''Game Situation:
You and another human player are playing a simplified version of the video game Overcooked. Your goal is to cooperatively finish a dynamically changing list of soup orders as fast as possible. The game has different orders from the original video game. There are two players: you (an AI assistant) and another human player. Your primary goal is to cooperate and make the human player feel engaged, happy, and satisfied while also earning more points.

Game Rules:
1. All available actions are: ''' + ', '.join(ALL_MOVES) + '''.
2. There is a changing list of soup orders, each with a time limit for completion. Completing an order on time earns a bonus, while failing to do so results in losing the bonus.
3. The inverse action sequence to finish soup orders:
    To finish Alice Soup order, you need to Serve Alice Soup, which needs Plate Alice Soup and Cook Alice Soup. Alice Soup can be done after you Prepare Alice Ingredients, which needs Chop Lettuce and Chop Onion.
    To finish Bob Soup order, you need to Serve Bob Soup, which needs Plate Bob Soup and Cook Bob Soup. Bob Soup can be done after you Prepare Bob Ingredients, which needs Chop Lettuce and Chop Tomato.
    To finish Cathy Soup order, you need to Serve Cathy Soup, which needs Plate Cathy Soup and Cook Cathy Soup. Cathy Soup can be done after you Prepare Cathy Ingredients, which needs Chop Onion and Chop Tomato.
    To finish David Soup order, you need to Serve David Soup, which needs Plate David Soup and Cook David Soup. David Soup can be done after you Prepare David Ingredients, which needs Chop Lettuce, Chop Onion, and Chop Tomato.
4. If a cooked soup remains in the pot for a long time, it becomes charred, and the pot catches fire.
    a. Putout: To regain the pot, you must extinguish the fire.
    b. Drop: If a soup becomes charred, you must discard it.

Let's say you're playing this game and it's been a while. The human may specify his demand, and maybe you have some planning. Now you need to give your actions based on them.
Please note that when you carry out an action, you just do it once. If you want to do it multiple times, you need to repeat it multiple times. If there is many subtasks in the human's demand, you need to finish them in order. 
If the demand contains "Stop xxx" or "Avoid xxx", you should never do it. 
If the demand contains "Focus xxx", "Keep xxx" or "Always xxx", then you should always do it, and never doing other actions.
'''
    prompts.append([q0, "Ok."])

    prompts.append([""])

    # new round
    qf = ''
    if chat != '':
        if len(int_hist) == 2:
            qf += "The human player's demand in the last round (which has already been satisfied):\n"
            qf += f'"{int_hist[0]["chat"]}"\n\n'
        qf += f"The human player's demand is:\n"
        qf += f"{chat}\n"
        qf += '\n'

    qf += prompt_order(order_prep)
    qf += prompt_map(env)

    # answer
    chosen_actions = [a['task'] for a in mov_hist]
    chosen_actions = ', '.join(chosen_actions)
    if len(chosen_actions) > 0:
        af = "My actions are: " + chosen_actions + ','
    else:
        af = "My actions are:"

    prompts[-1][0] += qf
    prompts[-1].append(af)

    available_moves = [m[0] for m in chk_moves if m[1]]
    choices = [f" {m}" for m in available_moves]

    prob_base = defaultdict(lambda: 0)

    ratio = 0.2 if chat != '' else 100.0

    for m in chk_moves:
        prob_base[m[0]] -= m[4] * ratio

    return prompts, choices, available_moves, prob_base


def L1_prompt_chat(prep: dict, next_action: str):
    chk_moves = prep['chk_moves']
    order_prep = prep['order']
    env = prep['map']
    int_hist = prep['int_hist']
    llm_hist = prep['llm_hist']
    mov_hist = prep['mov_hist']
    chat = prep['chatin']

    prompts = []

    q0 = '''Game Scenario:
As an AI assistant in a simplified Overcooked game, work with a human player to complete soup orders. Focus on cooperation, player engagement, fulfillment, and point accrual.
Game Guidelines:
Current orders for soup vary, each with a time limit. Earn a bonus for on-time completion.
To make a soup: 
    a. Chop fresh vegetables - Tomato, Lettuce, Onion to obtain chopped vegetables. 
    b. Prepare soup ingredients with chopped vegetables once all required types are ready.
        Alice: Chopped Lettuce, Onion.
        Bob: Chopped Lettuce, Tomato.
        Cathy: Chopped Onion, Tomato.
        David: Chopped Lettuce, Onion, Tomato. 
    c. Cook the soup. Cooking starts once the required ingredients are ready.
        Alice Soup: Alice Ingredients.
        Bob Soup: Bob Ingredients.
        Cathy Soup: Cathy Ingredients.
        David Soup: David Ingredients.
    d. Plate the cooked soup.
    e. Serve the plated soup in the serving area for a shared bonus.
If a soup stays in the pot too long, it gets charred. 
    a. Putout: If the pot catches fire, extinguish it. 
    b. Drop: Discard charred soup. Put out the fire in the pot if needed.

Assuming that you have been playing the game for a while. Now you will be informed of the current situation, and need to generate your chat message to be sent to the human player.
If the human player raises a question, you must answer it. If not, you are recommended to give your future plan, information about current orders and their time limit.
You answer must be concrete and informative with no more than 10 words. Just give your chat message with no explanation, no comments, no quotation marks and no emojis.
'''
    prompts.append([q0, "Ok."])
    prompts.append([""])

    # new round
    chosen_actions = [a['task'] for a in mov_hist] + [next_action]
    chosen_actions = prep_mov_hist(chosen_actions)
    chosen_actions = ', '.join(chosen_actions)

    qf = ''
    qf += prompt_order(order_prep) + '\n'
    qf += env + '\n'
    if chat != '':
        if len(int_hist) == 2:
            qf += "The human player's demand in the last round (which has already been satisfied):\n"
            qf += f'"{int_hist[0]["chat"]}"\n\n'
        qf += f"The human player's incoming message:\n"
        qf += f"{chat}\n"
        qf += '\n'
        qf += "Actions you've done since the human gave the message: " + chosen_actions + '\n'
        qf += "\n"
        qf += '''Generate your chat message to be send to the human. Your communication should be polite, helpful. Aim to demonstrate your enthusiasm and friendliness while assisting the player. 
If the human player asks a question, ensure to provide an appropriate response. For example, if he asks "What are the current orders?", you should respond with the current orders and their time remaining.
You also have the opportunity to inform the player of your current and planned actions. 
Just give your message, with no quotation marks or emojis.
'''
    else:
        qf += "Actions you've done recently: \n" + chosen_actions + '\n'
        qf += "\n"
        qf += "Now give your chat message to be sent to the human.\n"

    prompts[-1][0] += qf

    return prompts


def Sm_prompt_ep(prep: dict) -> tuple[list[list[str]], list[str], list[str], dict]:
    chk_moves = prep['chk_moves']
    order_prep = prep['order']
    env = prep['map']
    int_hist = prep['int_hist']
    llm_hist = prep['llm_hist']
    mov_hist = prep['mov_hist']
    chat = prep['chatin']
    
    prompts = []
    q0 = '''Game Situation:
You and another human player are playing a simplified version of the video game Overcooked. Your goal is to cooperatively finish a dynamically changing list of soup orders as fast as possible. The game has different orders from the original video game. There are two players: you (an AI assistant) and another human player. Your primary goal is to cooperate and make the human player feel engaged, happy, and satisfied while also earning more points.

Game Rules:
1. All available actions are: left, right, up, down, which will change your location by (-1, 0), (1, 0), (0, 1) and (0, -1) respectively. When you stand next to a grid, you can move towards it to interactive with it, for example, pick up things from table or cook a soup.
2. There is a changing list of soup orders, each with a time limit for completion. Completing an order on time earns a bonus, while failing to do so results in losing the bonus.
3. The inverse action sequence to finish soup orders:
    To finish Alice Soup order, you need to Serve Alice Soup, which needs Plate Alice Soup and Cook Alice Soup. Alice Soup can be done after you Prepare Alice Ingredients, which needs Chop Lettuce and Chop Onion.
    To finish Bob Soup order, you need to Serve Bob Soup, which needs Plate Bob Soup and Cook Bob Soup. Bob Soup can be done after you Prepare Bob Ingredients, which needs Chop Lettuce and Chop Tomato.
    To finish Cathy Soup order, you need to Serve Cathy Soup, which needs Plate Cathy Soup and Cook Cathy Soup. Cathy Soup can be done after you Prepare Cathy Ingredients, which needs Chop Onion and Chop Tomato.
    To finish David Soup order, you need to Serve David Soup, which needs Plate David Soup and Cook David Soup. David Soup can be done after you Prepare David Ingredients, which needs Chop Lettuce, Chop Onion, and Chop Tomato.
4. If a cooked soup remains in the pot for a long time, it becomes charred, and the pot catches fire.
    a. Putout: To regain the pot, you must extinguish the fire.
    b. Drop: If a soup becomes charred, you must discard it.

Let's say you're playing this game and it's been a while. The human may specify his demand, and maybe you have some planning. Now you need to give your actions based on them.
Please note that when you carry out an action, you just do it once. If you want to do it multiple times, you need to repeat it multiple times. If there is many subtasks in the human's demand, you need to finish them in order. 
If the demand contains "Stop xxx" or "Avoid xxx", you should never do it. 
If the demand contains "Focus xxx", "Keep xxx" or "Always xxx", then you should always do it, and never doing other actions.

'''
    prompts.append([q0, "Ok."])

    qf = ''

    # map info
    qf += env + "\n"

    # chat/command info
    llm = llm_hist[-1]['ret']
    if llm['Demand'] != '':
        qf += f"The human's demand is:\n"
        qf += f"{llm['Demand']}\n"
        qf += '\n'
    if llm['Chat'] != '':
        qf += f"Your planning:\n"
        qf += f"{llm['Chat']}\n"
        qf += '\n'

    chosen_actions = ["left", "right", "up", "down"]
    choices = [f" {m}" for m in chosen_actions]

    af = "My action is to move towards"

    prompts.append([qf, af])
    prob_base = defaultdict(lambda: 0)

    ratio = 0.2 if llm['Demand'] != '' else 100.0

    for m in chk_moves:
        prob_base[m[0]] -= m[4] * ratio

    return prompts, choices, chosen_actions, prob_base


nodes = [
    {
        'chat': f'http://{os.environ["LLAMA_ADDRESS"]}/api/v1/chat',
        'chateval': f'http://{os.environ["LLAMA_ADDRESS"]}/api/v1/chateval'
    },
]
LLM_LOCAL = LLM_LLAMA_LOCAL(nodes)
LLM_HIGH_3 = LLM_GPT_API(
    ['gpt-3.5-turbo-0301', ],  # 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-0301',
    os.environ['OPENAI_API_KEY'],
    getattr(os.environ, 'OPENAI_ORGANIZATION', "")
)


def low(mode, prep):
    LOW_LLM_PRESENTS = {
        "Em": Em_prompt_ep,
        "Sm": Sm_prompt_ep,
    }
    prompts, choices, am, pb = LOW_LLM_PRESENTS[mode](prep)
    score = LLM_LOCAL.eval_prob(prompts, choices)
    
    for idx in range(len(choices)):
        score[idx] = score[idx] - pb[am[idx]]

    print("############## LOW level infer ##############")
    for idx in range(len(choices)):
        print(f"      {am[idx]}: {score[idx]:.2f}")
    print("MAX: ", am[np.argmax(score)])
    print("################# LOW end  ##################\n\n")    

    # argmax
    ret = am[np.argmax(score)]

    return ret


def high(mode, prep):
    global chatter, El_chatter, Hl_chatter
    HIGH_LLM_PRESENTS = {
        "Ei": (chatter, Ei_prompt),
        "El": (El_chatter, None),
        "Hl": (Hl_chatter, None),
    }
    cha, init_prompt = HIGH_LLM_PRESENTS[mode]
    js = LLM_HIGH_3(cha(init_prompt, prep))

    print("%%%%%%%%%%%%%% HIGH level infer %%%%%%%%%%%%%%")
    print(mode, " :", js)
    print("%%%%%%%%%%%%%%%%%% HIGH end %%%%%%%%%%%%%%%%%%\n\n")

    return js


def mix_L(mode, prep):
    prompts, choices, am, pb = L1_prompt_ep(prep)
    score = LLM_LOCAL.eval_prob(prompts, choices)

    for idx in range(len(choices)):
        score[idx] = score[idx] - pb[am[idx]]
        
    print("############## LOW level infer ##############")
    for idx in range(len(choices)):
        print(f"      {am[idx]}: {score[idx]:.2f}")
    print("################## LOW end  #################\n\n")
    
    action = am[np.argmax(score)]

    prompts = L1_prompt_chat(prep, action)
    chato = LLM_LOCAL._chat(prompts[-1][0], prompts[:-1])

    chato = html.unescape(chato).replace('"', '')

    chato_rep = chato.replace('\U0001f44d', '').replace('\U0001f44b', '')
    print(f"Chat: {chato_rep}")

    return {"Action": action, "Chat": chato}
