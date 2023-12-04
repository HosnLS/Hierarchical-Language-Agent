def prep_mov_hist(mov_hist):
    moves = {}
    for m in mov_hist:
        if m not in moves:
            moves[m] = 1
        else:
            moves[m] += 1

    def times(i):
        if i == 1:
            return " once"
        elif i == 2:
            return " twice"
        else:
            return f" {i} times"

    moves = [f"{k}{times(v)}" for k, v in moves.items()]

    return moves


# PROMPT

def prompt_base_Ei() -> list[list[str]]:
    # Intention Inference
    p = '''Game Scenario:
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

In-game Decision:
You need to interpret the human player's message into a simpler form, which will be sent to a downstream AI without access to human message history. Your answer must be clear and succinct.

The human's message can be:
1. Useless message: Message that has no specific demand such as "Enough", "Never mind", "You are free to do anything else" or "Try your best to earn more points." translates to "None."
2. Short-term request: "Chop 4 more" means "Chop xxx 4 times.", where "xxx" should be the vegetable in past intention. Keep you answer concise and make sure the numbers are corrent. "Plate the soup now" should be "Plate Soup once."
3. Intention needs to be inferred: Sometimes you need to infer about the hidden meaning of messages. For instance, "I will cook the first order. Can you take charge of the rest?" implies "Cook xxx once and Cook xxx once." where "xxx" are the subsequent soup orders. Similarly, "xxx is handled by me." implies "Cook xxx." where the two "xxx" are different soup in the orders. Emotional and cryptic message like "The David Soup is about to timeout!" suggest "Serve David Soup once."
4. Long-term request: Messages such as "Keep chopping tomatoes" become "Always keep chopping tomatoes, and don't stop." 
5. Questions: Like "What are the orders", "What is xxx Soup" or any question-like queries. You must repeat the original question completely in your output. You must leave the question to the downstream AI intactly who will answer it.
6. Special case: Messages related to asking for orders, like "Tell me the orders", "Keep telling me the orders" or "I want to know the orders" should be translated to "What are the orders now?".

If the human's intention conflicts with soup orders, you should follow the human's intention even if it is not on the orders. Always prioritize the human's message.

Any explanations, comments, tips or modal particles must not be included.
'''
    return [[p, "Ok."], [""]]


def prompt_order_int(order_prep: list) -> str:
    if len(order_prep) == 0:
        prompt = 'Soup orders are not visible to you.\n\n'
        return prompt

    orders = [rec['name'].replace("Plated ", "") for rec in order_prep]
    prompt = f"Current soup orders: {', '.join(orders)}\n\n"

    return prompt


def prompt_order(order_prep: list) -> str:
    # current orders
    prompt = '''
Current soup orders:
'''
    if len(order_prep) == 0:
        prompt += 'Soup orders are currently not visible for you.\n'
        return prompt

    for rec in order_prep:
        soup_name = rec['name'].replace("Plated ", "")
        rate = rec['rate']
        prompt += f'- {soup_name}: '
        if rate > 0.75:
            prompt += 'plenty of time\n'
        elif rate > 0.5:
            prompt += 'still some time\n'
        elif rate > 0.25:
            prompt += 'not much time\n'
        else:
            prompt += 'will expire in no time\n'

    return prompt


def prompt_map(inp: str) -> str:
    return inp


def prompt_reason_Ei(int_hist: list) -> str:
    prompt = ""
    # task history
    if len(int_hist) == 2 and int_hist[0]['ret'] is not None:
        prompt += "The human player's intention in the last round (which has already been satisfied):\n"
        prompt += f'"{int_hist[0]["ret"]}"\n\n'

    prompt += "The human player's message now:\n"
    prompt += f'"{int_hist[-1]["chat"]}"\n\n'

    prompt += "\n"
    prompt += '''Be very careful that if the message is a question, you must repeat it completely in your answer. DO NOT ANSWER IT. You need to interpret the human player's message only if it is not a question.'''

    return prompt


def prompt_base_El_s(prep):
    # Chat & Completion Assessment: init (with ongoing intention)
    p = '''Game Scenario:
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

Gameplay Rounds:
Round One - Action Summary: In this stage, your task is to summarize the actions you've made that are directly beneficial to the human player's request. 
Round Two - Communication: Here, you generate your chat message to be sent to the human player.
Round Three - Satisfaction Evaluation: In this round, it's your responsibility to judge whether the player's request has been fully met based on your actions.

Note that there are multiple types of human's incoming message:
1. Short term request: Like "Chop 4 times", "Chop once", "Cook 2 Soup" or "Plate once". If you have done ALL actions he requests, then it is satisfied. It is OK if you've done more than he asks. If there are still actions to be done, then it is not satisfied.
2. Long term request: Like "Always prepare", "Keep chopping", "Plating continuously", "Cook don't stop" or "Avoid serving". In these cases, the requests will never be satisfied because they need to be done continuously, even if your actions conflict with them,
3. Question: Like "What are the current orders?" or "What is xxx Soup?" You need to answer to the question in the chat message. And you must give "Yes" in the Satisfaction Evaluation round.
4. Useless message: Like "None", "Free to do anything", "No specific intention", or statement of fact like "The orders are xxx". You must "Yes" in the Satisfaction Evaluation round.
'''
    return p


def prompt_base_El_s2(prep):
    # Chat & Completion Assessment: init init (without ongoing intention)
    p = '''Game Scenario:
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
You are recommended to give your future plan. Giving information about current orders and their time limit is also a good idea. You shouldn't focus on the Fire Extinguisher.
You answer must be concrete and informative with no more than 10 words. Just give your chat message with no explanation, no comments, no quotation marks and no emojis.
'''

    return p


def prompt_base_El_1(prep):
    # Chat & Completion Assessment: reasoning (with ongoing intention)
    def prompt_reason_El(mov_hist: list, chat: str) -> str:
        prompt = ""
        prompt += "The human player's incoming message:\n"
        prompt += f'"{chat}"\n\n'

        mh = [m['task'] for m in mov_hist]
        moves = prep_mov_hist(mh)

        prompt += "Actions you've done since the human gave the message:\n"
        prompt += f'{", ".join(moves)}\n\n' if len(moves) > 0 else "None\n\n"

        return prompt

    # chk_moves = prep['chk_moves']
    order_prep = prep['order']
    env = prep['map']
    # int_hist = prep['int_hist']
    # llm_hist = prep['llm_hist']
    mov_hist = prep['mov_hist']
    chat = prep['chatin']

    order = prompt_order(order_prep) + '\n'
    reason = prompt_reason_El(mov_hist, chat) + '\n'

    p = order
    p += env
    p += reason
    p += "\n"
    p += '''You need to examine the current state of the game environment, the human player's message, and actions you've taken so far. Now summarize the actions you've done that are directly beneficial to the human player's request. Any action not related to their request can be ignored.\n'''
    p += '''If the human player's request is "None" or a question, just briefly summarize your current actions.\n'''
    p += '''You must be honest and give actions that is surely done by yourself. Do not make up!\n'''
    p += '''Keep your answer short and concise. No more than 20 words.\n'''

    # print(reason)

    return p


def prompt_base_El_2(prep):
    # Chat & Completion Assessment: chat message (without ongoing intention)
    p = '''Generate your chat message to be send to the human. Your communication should be polite, helpful, and limited to 20 words max. Aim to demonstrate your enthusiasm and friendliness while assisting the player. 
If the human player asks a question, ensure to provide an appropriate response. For example, if he asks "What are the current orders?", you should respond with the current orders and their time remaining.
You also have the opportunity to inform the player of your current and planned actions. 
Just give your message, with no quotation marks or emojis.'''

    return p


def prompt_base_El_22(prep):
    # Chat & Completion Assessment: chat message (with ongoing intention)
    def prompt_reason_El2(mov_hist: list, chat: str) -> str:
        mh = [m['task'] for m in mov_hist]
        moves = prep_mov_hist(mh)

        prompt = "Actions you've done recently:\n"
        prompt += f'{", ".join(moves)}\n\n' if len(moves) > 0 else "None\n\n"

        return prompt

    order_prep = prep['order']
    env = prep['map']
    # int_hist = prep['int_hist']
    # llm_hist = prep['llm_hist']
    mov_hist = prep['mov_hist']
    chat = prep['chatin']

    order = prompt_order(order_prep) + '\n'
    reason = prompt_reason_El2(mov_hist, chat) + '\n'

    p = order
    p += env + '\n\n'
    p += reason

    p += '''
Now give your chat message to be sent to the human. 
'''

    return p


def prompt_base_El_3(prep):
    # Chat & Completion Assessment: completion assessment (with ongoing intention)
    p = '''Judge whether the player's request has been fulfilled by your actions. The possible responses are "Yes" or "No". 
If the human's incoming message is a question or a useless message, give "Yes". '''
    return p


def prompt_base_Hl_s(prep):
    # SMOA: init
    p = '''Game Scenario:
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

Gameplay Rounds:
Round One - Action Summary: In this stage, your task is to summarize the actions you've made that are directly beneficial to the human player's request. 
Round Two - Communication: Here, you generate your chat message to be sent to the human player.
Round Three - Satisfaction Evaluation: In this round, it's your responsibility to judge whether the player's request has been fully met based on your actions.
Round Four - Action Execution: You are to give your action to be carried out next.

Note that there are multiple types of human's incoming message:
1. Short term request: Like "Chop 4 times", "Chop once", "Cook 2 Soup" or "Plate once". If you have done ALL actions he requests, then it is satisfied. It is OK if you've done more than he asks. If there are still actions to be done, then it is not satisfied.
2. Long term request: Like "Always prepare", "Keep chopping", "Plating continuously", "Cook don't stop" or "Avoid serving". In these cases, the requests will never be satisfied because they need to be done continuously, even if your actions conflict with them,
3. Question: Like "What are the current orders?" or "What is xxx Soup?" You need to answer to the question in the chat message. And you must give "Yes" in the Satisfaction Evaluation round.
4. Useless message: Like "None", "Free to do anything", "No specific intention", or statement of fact like "The orders are xxx". You must "Yes" in the Satisfaction Evaluation round.
'''
    return p


def prompt_base_El_5(prep):
    # SMOA: act
    chk_moves = prep['chk_moves']
    all_moves = [m[0] for m in chk_moves if m[1]]
    p = "Give your action to be carried out next. You should try to serve, plate and cook soup when possible. Select it from " + ", ".join(
        all_moves) + ". You can only choose one from it and not allowed to make up new action. Explanation or comment is not needed."
    return p
