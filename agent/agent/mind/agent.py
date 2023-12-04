from pathlib import Path
import threading
import requests
import time
from copy import copy, deepcopy
import random
from dataclasses import dataclass

from agent.executor.low import EnvState
from agent.executor.high import HighTask
from agent.mind.prompt_local import MOVE_TO_HT, prep_prompt, prep_prompt_s
from agent.mind.call import low, high, mix_L
from gym_cooking.utils.replay import Replay


def request_client(mode, llm, data):
    if mode in ['L1l']:
        return mix_L(mode, data)
    elif mode in ['Ei', 'El', 'Hl']:
        return high(mode, data)
    elif mode in ['Em', 'Sm']:
        return low(mode, data)
    else:
        raise NotImplementedError


@dataclass
class AgentSetting:
    mode: str
    high_llm: str = 'gpt-3.5'
    low_llm: str = 'llama'
    speed: float = 2.5


class HLAagent:
    INT_HIST_MAX_LEN = 5 * 10000
    LLM_HIST_MAX_LEN = 5 * 10000
    MAX_LLM_UNG_MOV = 15
    MAX_LLM_FIN_MOV = 6
    MAX_MOV_UNG_MOV = 15
    MAX_MOV_FIX_MOV = 3

    def __init__(self, setting: AgentSetting, replay: Replay):
        self.setting = setting
        self.replay = replay

        self._lock = threading.Lock()
        self._last_env = None
        # high level
        self._int_hist = []  # submit_time, finish_time, chat, ret (intention)
        # int_time(finish_time), submit_time, finish_time, chat, ret {Chat, Demand, Finished}
        self._llm_hist = []
        self._num_high_threads = 0
        # low level
        self._mov_hist: list = []  # task, status, submit_time, finish_time

        # thread safe
        self._it_time = 0
        self._lt_time = 0  # last time
        self._mt_time = 0
        self._task = None

    @property
    def _is_finished(self):
        if not self._int_hist:
            return True
        intent = self._int_hist[-1]

        if intent['finish_time'] is None:
            return False
        intent_time = intent['finish_time']

        llms = self._llm_hist[-100:]
        llms = [l for l in llms if l['submit_time'] > intent_time]

        return any([l['ret']['Finished'] for l in llms]) or len(llms) > self.MAX_LLM_UNG_MOV

    def _get_high_mov_hist(self):
        if self._is_finished:
            mov_his = self._mov_hist[-100:]
            mov_his = [m for m in mov_his if m['status'].startswith(
                'Success') or m['status'].startswith('Ongoing')]
            mov_his = mov_his[-self.MAX_LLM_FIN_MOV:]
        else:
            intent_time = self._int_hist[-1]['submit_time']
            mov_his = self._mov_hist[-100:]
            mov_his = [m for m in mov_his if m['finish_time'] > intent_time]
            mov_his = [m for m in mov_his if m['status'].startswith(
                'Success') or m['status'].startswith('Ongoing')]
            mov_his = mov_his[-self.MAX_LLM_UNG_MOV:]

        return mov_his

    def _get_low_mov_hist(self):
        if self._is_finished:
            mov_his = self._mov_hist[-100:]
            mov_his = [m for m in mov_his if m['status'].startswith('Success')]
            mov_his = mov_his[-self.MAX_MOV_FIX_MOV:]
        else:
            intent_time = self._int_hist[-1]['submit_time']
            mov_his = self._mov_hist[-100:]
            mov_his = [m for m in mov_his if m['finish_time'] > intent_time]
            mov_his = [m for m in mov_his if m['status'].startswith('Success')]
            mov_his = mov_his[-self.MAX_MOV_UNG_MOV:]

        return mov_his

    def _get_mov_infer_prep(self):
        if self._is_finished:
            ret = {'ret': {'Demand': '', 'Chat': ''}}
            if self._llm_hist:
                ret = {'ret': {'Demand': '',
                               'Chat': self._llm_hist[-1]['ret']['Chat']}}
        else:
            intent = self._int_hist[-1]
            if intent['finish_time'] is None:
                ret = {'ret': {'Demand': intent['chat'], 'Chat': ''}}
            else:
                ret = {'ret': {'Demand': intent['ret'], 'Chat': ''}}
        return [ret]

    def _int_infer(self, chat: str = ''):
        submit_time = time.time()
        with self._lock:
            self._int_hist.append({
                'submit_time': submit_time,
                'finish_time': None,
                'chat': chat,
                'ret': None
            })

            int_his = self._int_hist[-2:]

            prep = prep_prompt(self._last_env, int_his, [], [], '')
            prep = deepcopy(prep)

            self._num_high_threads += 1

        try:
            js = request_client("Ei", self.setting.high_llm, prep)
        except:
            with self._lock:
                self._num_high_threads -= 1
            return
        # print(js)
        finish_time = time.time()

        # log
        self.replay.log("ai.int_infer",
                        {"prep": prep, "ret": js, "time_start": submit_time, "time_end": finish_time})

        with self._lock:
            self._num_high_threads -= 1
            if js is None:
                js = "None"
            # no cross-time period
            intent = [
                i for i in self._int_hist if i['submit_time'] == submit_time]
            if not intent:
                return
            intent = intent[0]
            intent['finish_time'] = finish_time
            intent['ret'] = js
            self._int_hist = self._int_hist[-self.INT_HIST_MAX_LEN:]

        self._llm_infer()

    def _llm_infer(self):
        submit_time = time.time()
        with self._lock:
            # if int is inferring, return
            if self._int_hist and self._int_hist[-1]['finish_time'] is None:
                return
            # no new chat checker
            if self._is_finished:
                intention = "None"
            else:
                i = self._int_hist[-1]
                # (or in other words: {i['ret']})
                intention = f"{i['ret']}" if self._int_hist else "None"
            int_time = self._int_hist[-1]['submit_time'] if self._int_hist else 0

            # prepare moves history
            mov_his = self._get_high_mov_hist()

            prep = prep_prompt(self._last_env, [], [], mov_his, intention)
            prep = deepcopy(prep)

            self._num_high_threads += 1

        try:
            js = request_client("El", self.setting.high_llm, prep)
        except:
            with self._lock:
                self._num_high_threads -= 1
            return
        finish_time = time.time()

        # log
        self.replay.log("ai.llm_infer",
                        {"prep": prep, "ret": js, "time_start": submit_time, "time_end": finish_time})

        with self._lock:
            self._num_high_threads -= 1

            if js is None:
                return
            if self._int_hist and self._int_hist[-1]['finish_time'] is None:
                return

            # check intention out date
            int_time2 = self._int_hist[-1]['submit_time'] if self._int_hist else 0
            if int_time != int_time2:
                return
            # check llm out date
            if self._llm_hist:
                llm2 = self._llm_hist[-1]
                if llm2['submit_time'] > submit_time:
                    return

            self._llm_hist.append({
                'submit_time': submit_time,
                'finish_time': finish_time,
                'chat': '',
                'ret': js
            })
            self._llm_hist = self._llm_hist[-self.LLM_HIST_MAX_LEN:]

    def high_level_infer(self, env: EnvState = None, chat: str = ''):
        if env is not None:
            with self._lock:
                self._last_env = env

        thread = threading.Thread(target=self._int_infer, args=(chat,))
        thread.daemon = True
        thread.start()

    def low_level_infer(self):
        submit_time = time.time()

        llm_his = self._get_mov_infer_prep()
        mov_his = self._get_low_mov_hist()

        prep = prep_prompt(self._last_env, [], llm_his, mov_his, '')
        try:
            ht = request_client("Em", self.setting.low_llm, prep)
        except:
            return
        finish_time = time.time()

        self.replay.log("ai.mov_infer",
                        {"prep": prep, "ret": ht, "time_start": submit_time, "time_end": finish_time})
        self._task = deepcopy(MOVE_TO_HT[ht])
        self._mov_hist.append({'task': str(self._task), 'status': 'Ongoing. Initiated.',
                               'submit_time': submit_time, 'finish_time': finish_time})

    def _check_interrupt(self):
        # check llm incoming
        if self._llm_hist and self._lt_time < self._llm_hist[-1]['finish_time']:
            # obtain chat
            chat = self._llm_hist[-1]['ret']['Chat']
            self._lt_time = self._llm_hist[-1]['finish_time']
        else:
            chat = ''

        # check int incoming
        if self._int_hist and self._it_time < self._int_hist[-1]['submit_time']:
            self._it_time = self._int_hist[-1]['submit_time']
            if self._it_time > self._lt_time:
                if self._mov_hist and self._mov_hist[-1]['status'].startswith('Ongoing'):
                    self._mov_hist[-1]['status'] = 'Interrupted. '
                self._task = None
        if self._int_hist and self._int_hist[-1]['finish_time'] is not None \
                and self._it_time < self._int_hist[-1]['finish_time']:
            self._it_time = self._int_hist[-1]['finish_time']
            if self._it_time > self._lt_time:
                if self._mov_hist and self._mov_hist[-1]['status'].startswith('Ongoing'):
                    self._mov_hist[-1]['status'] = 'Interrupted. '
                self._task = None

        return chat

    def __call__(self, env: EnvState):
        self._lock.acquire()
        # update env
        self._last_env = env

        # check high level incoming
        chat = self._check_interrupt()

        # submit count check
        start_llm_infer = False
        if self._num_high_threads <= 0:
            if not self._llm_hist:
                start_llm_infer = True
            elif self._llm_hist[-1]['finish_time'] < time.time() - 5:
                start_llm_infer = True
            if not self._is_finished and self._mov_hist and self._mov_hist[-1]['submit_time'] != self._mt_time:
                start_llm_infer = True
                self._mt_time = self._mov_hist[-1]['submit_time']

        if start_llm_infer:
            self._lock.release()
            thread = threading.Thread(target=self._llm_infer)
            thread.daemon = True
            thread.start()
            self._lock.acquire()

        while True:
            if self._task is None:
                self.low_level_infer()
            if self._task is None:
                continue

            state, move, msg = self._task(env)
            if state == HighTask.Working:  # working
                self._mov_hist[-1]['status'] = 'Ongoing. ' + msg
                self._lock.release()
                return move, chat
            elif state == HighTask.Failed:  # reassign task
                self._mov_hist[-1]['status'] = 'Failed. ' + msg
                print(f"Move Failed: {self._mov_hist[-1]['task']}")
                self._task = None
                self._lock.release()
                return (0, 0), chat
            else:
                self._mov_hist[-1]['status'] = f'Success.'
                self._task = None


class SMOAagent:
    INT_HIST_MAX_LEN = 5 * 10000
    LLM_HIST_MAX_LEN = 5 * 10000
    MAX_LLM_UNG_MOV = 9
    MAX_LLM_FIN_MOV = 6

    def __init__(self, setting: AgentSetting, replay: Replay):
        self.setting = setting
        self.replay = replay

        self._lock = threading.Lock()
        self._last_env = None
        # high level
        self._int_hist = []  # submit_time, finish_time, chat, ret (intention)
        # int_time(finish_time), submit_time, finish_time, chat, ret {Chat, Demand, Finished}
        self._llm_hist = []
        self._num_high_threads = 0
        # low level
        self._mov_hist: list = []  # task, status, submit_time, finish_time

        # thread safe
        self._it_time = 0
        self._lt_time = 0  # last time
        self._mt_time = 0
        self._tasks = []
        self._task = None

    @property
    def _is_finished(self):
        if not self._int_hist:
            return True
        intent = self._int_hist[-1]

        if intent['finish_time'] is None:
            return False
        intent_time = intent['finish_time']

        llms = self._llm_hist[-100:]
        llms = [l for l in llms if l['submit_time'] > intent_time]

        return any([l['ret']['Finished'] for l in llms])

    def _get_high_mov_hist(self):
        if self._is_finished:
            mov_his = self._mov_hist[-100:]
            mov_his = [m for m in mov_his if m['status'].startswith(
                'Success') or m['status'].startswith('Ongoing')]
            mov_his = mov_his[-self.MAX_LLM_FIN_MOV:]
        else:
            intent_time = self._int_hist[-1]['submit_time']
            mov_his = self._mov_hist[-100:]
            mov_his = [m for m in mov_his if m['finish_time'] > intent_time]
            mov_his = [m for m in mov_his if m['status'].startswith(
                'Success') or m['status'].startswith('Ongoing')]
            mov_his = mov_his[-self.MAX_LLM_UNG_MOV:]

        return mov_his

    def _int_infer(self, chat: str = ''):
        submit_time = time.time()
        with self._lock:
            self._int_hist.append({
                'submit_time': submit_time,
                'finish_time': None,
                'chat': chat,
                'ret': None
            })

            int_his = self._int_hist[-2:]

            prep = prep_prompt(self._last_env, int_his, [], [], chat)
            prep = deepcopy(prep)

            self._num_high_threads += 1

        try:
            js = request_client("Ei", self.setting.high_llm, prep)
        except:
            with self._lock:
                self._num_high_threads -= 1
            return
        finish_time = time.time()

        # log
        self.replay.log("ai.int_infer",
                        {"prep": prep, "ret": js, "time_start": submit_time, "time_end": finish_time})

        with self._lock:
            self._num_high_threads -= 1
            if js is None:
                js = "None"
            # no cross-time period
            intent = [
                i for i in self._int_hist if i['submit_time'] == submit_time]
            if not intent:
                return
            intent = intent[0]
            intent['finish_time'] = finish_time
            intent['ret'] = js
            self._int_hist = self._int_hist[-self.INT_HIST_MAX_LEN:]

        self._llm_infer()

    def _llm_infer(self):
        submit_time = time.time()
        with self._lock:
            # if int is inferring, return
            if self._int_hist and self._int_hist[-1]['finish_time'] is None:
                return
            # no new chat checker
            if self._is_finished:
                intention = "None"
            else:
                intention = self._int_hist[-1]['ret'] if self._int_hist else "None"
            int_time = self._int_hist[-1]['submit_time'] if self._int_hist else 0

            # prepare moves history
            mov_his = self._get_high_mov_hist()

            prep = prep_prompt(self._last_env, [], [], mov_his, intention)
            prep = deepcopy(prep)

            self._num_high_threads += 1

        try:
            js = request_client("Hl", self.setting.high_llm, prep)
        except:
            with self._lock:
                self._num_high_threads -= 1
            return
        finish_time = time.time()

        # log
        self.replay.log("ai.llm_infer",
                        {"prep": prep, "ret": js, "time_start": submit_time, "time_end": finish_time})

        with self._lock:
            self._num_high_threads -= 1

            if js is None:
                return
            if self._int_hist and self._int_hist[-1]['finish_time'] is None:
                return

            # check intention out date
            int_time2 = self._int_hist[-1]['submit_time'] if self._int_hist else 0
            if int_time != int_time2:
                return
            # check llm out date
            if self._llm_hist:
                llm2 = self._llm_hist[-1]
                if llm2['submit_time'] > submit_time:
                    return

            self._llm_hist.append({
                'submit_time': submit_time,
                'finish_time': finish_time,
                'chat': '',
                'ret': js
            })
            self._llm_hist = self._llm_hist[-self.LLM_HIST_MAX_LEN:]

    def high_level_infer(self, env: EnvState = None, chat: str = ''):
        if env is not None:
            with self._lock:
                self._last_env = env

        thread = threading.Thread(target=self._int_infer, args=(chat,))
        thread.daemon = True
        thread.start()

    def low_level_infer(self):
        submit_time = time.time()

        if not self._tasks:
            return

        ht = self._tasks[0]
        if ht is None:
            self._tasks.pop(0)
            return
        task = deepcopy(MOVE_TO_HT[ht])
        can_begin = task.can_begin(self._last_env)

        if can_begin[0]:
            self._tasks.pop(0)
            self._task = task
            finish_time = time.time()
            self._mov_hist.append({'task': str(self._task), 'status': 'Ongoing. Initiated.',
                                   'submit_time': submit_time, 'finish_time': finish_time})
            self.replay.log("ai.mov_infer", {'prep': None, 'ret': str(self._task),
                                             'time_start': submit_time, 'time_end': submit_time})
            return

        while not can_begin[0] and len(can_begin[2]) > 0:
            task = deepcopy(MOVE_TO_HT[can_begin[2][0]])
            can_begin = task.can_begin(self._last_env)
        if can_begin[0]:
            self._task = task
            finish_time = time.time()
            self._mov_hist.append({'task': str(self._task), 'status': 'Ongoing. Initiated.',
                                   'submit_time': submit_time, 'finish_time': finish_time})
            self.replay.log("ai.mov_infer", {'prep': None, 'ret': str(self._task),
                                             'time_start': submit_time, 'time_end': submit_time})
            return
        else:
            self._tasks.pop(0)
            return

    def _check_interrupt(self):
        if self._llm_hist and self._lt_time < self._llm_hist[-1]['finish_time']:
            # obtain chat
            chat = self._llm_hist[-1]['ret']['Chat']
            self._task = None
            if self._mov_hist and self._mov_hist[-1]['status'].startswith('Ongoing'):
                self._mov_hist[-1]['status'] = 'Interrupted. '
            self._tasks = [self._llm_hist[-1]['ret']['Action'], ]
            self._lt_time = self._llm_hist[-1]['finish_time']
        else:
            chat = ''

        # check immediate action
        if self._int_hist and self._it_time < self._int_hist[-1]['submit_time']:
            self._it_time = self._int_hist[-1]['submit_time']
            if self._it_time > self._lt_time:
                if self._mov_hist and self._mov_hist[-1]['status'].startswith('Ongoing'):
                    self._mov_hist[-1]['status'] = 'Interrupted. '
                self._task = None
                self._tasks = []

        return chat

    def __call__(self, env: EnvState):
        self._lock.acquire()
        # update env
        self._last_env = env

        # check high level incoming
        chat = self._check_interrupt()

        # submit count check
        start_llm_infer = False
        if self._num_high_threads <= 0:
            if not self._llm_hist:
                start_llm_infer = True
            if self._task is None and not self._tasks:  # additional
                start_llm_infer = True

        if start_llm_infer:
            self._lock.release()
            thread = threading.Thread(target=self._llm_infer)
            thread.daemon = True
            thread.start()
            self._lock.acquire()

        while True:
            for _ in range(10):
                if self._task is not None:
                    break
                self.low_level_infer()

            if self._task is None:
                self._lock.release()
                return (0, 0), chat

            state, move, msg = self._task(env)
            if state == HighTask.Working:  # working
                self._mov_hist[-1]['status'] = 'Ongoing. ' + msg
                self._lock.release()
                return move, chat
            elif state == HighTask.Failed:  # reassign task
                self._mov_hist[-1]['status'] = 'Failed. ' + msg
                print(f"Move Failed: {self._mov_hist[-1]['task']}")
                self._task = None
                self._lock.release()
                return (0, 0), chat
            else:
                self._mov_hist[-1]['status'] = f'Success.'
                self._task = None


class FMOAagent:
    LLM_HIST_MAX_LEN = 5 * 10000
    MAX_MOV_UNG_MOV = 9
    MAX_MOV_FIX_MOV = 3

    def __init__(self, setting: AgentSetting, replay: Replay):
        self.setting = setting
        self.replay = replay

        self._last_env = None

        self._int_hist: list = []  # submit_time, finish_time, chat
        # submit_time, finish_time, chat, ret (Action, Chat)
        self._llm_hist: list = []
        self._mov_hist: list = []  # submit_time, finish_time, task, status

        # atom level
        self._task = None
        self._chat = ''

    @property
    def _is_finished(self):
        if not self._int_hist:
            return True
        intent = self._int_hist[-1]
        intent_time = intent['finish_time']

        llms = self._llm_hist[-100:]
        llms = [l for l in llms if l['submit_time'] > intent_time]

        return len(llms) > self.MAX_MOV_UNG_MOV

    def _get_low_mov_hist(self):
        if self._is_finished:
            mov_his = self._mov_hist[-100:]
            mov_his = [m for m in mov_his if m['status'].startswith('Success')]
            mov_his = mov_his[-self.MAX_MOV_FIX_MOV:]
        else:
            intent_time = self._int_hist[-1]['submit_time']
            mov_his = self._mov_hist[-100:]
            mov_his = [m for m in mov_his if m['finish_time'] > intent_time]
            mov_his = [m for m in mov_his if m['status'].startswith('Success')]
            mov_his = mov_his[-self.MAX_MOV_UNG_MOV:]

        return mov_his

    def _infer(self):
        submit_time = time.time()

        chat = self._int_hist[-1]['chat'] if not self._is_finished and self._int_hist else ""
        mov_his = self._get_low_mov_hist()

        prep = prep_prompt(
            self._last_env, self._int_hist[-2:], [], mov_his, chat)

        try:
            js = request_client("L1l", self.setting.low_llm, prep)
        except:
            return
        finish_time = time.time()

        self.replay.log("ai.llm_infer", {
                        "prep": prep, "ret": js, "time_start": submit_time, "time_end": finish_time})

        # drop long ago history
        self._llm_hist.append({
            'submit_time': submit_time,
            'finish_time': finish_time,
            'chat': chat,
            'ret': js
        })
        self._llm_hist = self._llm_hist[-self.LLM_HIST_MAX_LEN:]

        # update mov hist
        if self._mov_hist and self._mov_hist[-1]['status'].startswith('Ongoing'):
            self._mov_hist[-1]['status'] = 'Interrupted. '
        self._task = deepcopy(MOVE_TO_HT[js['Action']])
        self._chat = js['Chat']
        self._mov_hist.append({'task': str(self._task), 'status': 'Ongoing. Initiated.',
                               'submit_time': finish_time, 'finish_time': finish_time})
        self.replay.log("ai.mov_infer", {"prep": None, "ret": str(self._task),
                                         "time_start": finish_time, "time_end": finish_time})

    def high_level_infer(self, env: EnvState = None, chat: str = ''):
        if env is not None:
            self._last_env = env

        # update int
        submit_time = time.time()
        self._int_hist.append(
            {'submit_time': submit_time, 'finish_time': submit_time, 'chat': chat})
        self.replay.log("ai.int_infer", {"prep": {"chatin": chat}, "ret": None,
                                         "time_start": submit_time, "time_end": submit_time})

        self._infer()

    def __call__(self, env: EnvState):
        # update env
        self._last_env = env

        while True:
            if self._task is None:
                self._infer()
            if self._task is None:
                continue

            state, move, msg = self._task(env)
            if state == HighTask.Working:  # working
                self._mov_hist[-1]['status'] = 'Ongoing. ' + msg
                chat = self._chat
                self._chat = ''
                return move, chat
            elif state == HighTask.Failed:  # reassign task
                self._mov_hist[-1]['status'] = 'Failed. ' + msg
                print(f"Move Failed: {self._mov_hist[-1]['task']}")
                self._task = None
                chat = self._chat
                self._chat = ''
                return (0, 0), chat
            else:
                self._mov_hist[-1]['status'] = f'Success.'
                self._task = None


class NEAagent:
    INT_HIST_MAX_LEN = 5 * 10000
    LLM_HIST_MAX_LEN = 5 * 10000
    MAX_LLM_UNG_MOV = 9
    MAX_LLM_FIN_MOV = 6
    MAX_MOV_UNG_MOV = 9
    MAX_MOV_FIX_MOV = 3

    def __init__(self, setting: AgentSetting, replay: Replay):
        self.setting = setting
        self.replay = replay

        self._lock = threading.Lock()
        self._last_env = None
        # high level
        self._int_hist = []  # submit_time, finish_time, chat, ret (intention)
        # int_time(finish_time), submit_time, finish_time, chat, ret {Chat, Demand, Finished}
        self._llm_hist = []
        self._num_high_threads = 0
        # low level
        self._mov_hist: list = []  # task, status, submit_time, finish_time

        # thread safe
        self._it_time = 0
        self._lt_time = 0  # last time
        self._mt_time = 0
        self._task = None

    @property
    def _is_finished(self):
        if not self._int_hist:
            return True
        intent = self._int_hist[-1]

        if intent['finish_time'] is None:
            return False
        intent_time = intent['finish_time']

        llms = self._llm_hist[-100:]
        llms = [l for l in llms if l['submit_time'] > intent_time]

        return any([l['ret']['Finished'] for l in llms]) or len(llms) > self.MAX_LLM_UNG_MOV

    def _get_high_mov_hist(self):
        if self._is_finished:
            mov_his = self._mov_hist[-100:]
            mov_his = [m for m in mov_his if m['status'].startswith(
                'Success') or m['status'].startswith('Ongoing')]
            mov_his = mov_his[-self.MAX_LLM_FIN_MOV:]
        else:
            intent_time = self._int_hist[-1]['submit_time']
            mov_his = self._mov_hist[-100:]
            mov_his = [m for m in mov_his if m['finish_time'] > intent_time]
            mov_his = [m for m in mov_his if m['status'].startswith(
                'Success') or m['status'].startswith('Ongoing')]
            mov_his = mov_his[-self.MAX_LLM_UNG_MOV:]

        return mov_his

    def _get_low_mov_hist(self):
        if self._is_finished:
            mov_his = self._mov_hist[-100:]
            mov_his = [m for m in mov_his if m['status'].startswith('Success')]
            mov_his = mov_his[-self.MAX_MOV_FIX_MOV:]
        else:
            intent_time = self._int_hist[-1]['submit_time']
            mov_his = self._mov_hist[-100:]
            mov_his = [m for m in mov_his if m['finish_time'] > intent_time]
            mov_his = [m for m in mov_his if m['status'].startswith('Success')]
            mov_his = mov_his[-self.MAX_MOV_UNG_MOV:]

        return mov_his

    def _get_mov_infer_prep(self):
        if self._is_finished:
            ret = {'ret': {'Demand': '', 'Chat': ''}}
            if self._llm_hist:
                ret = {'ret': {'Demand': '',
                               'Chat': self._llm_hist[-1]['ret']['Chat']}}
        else:
            intent = self._int_hist[-1]
            if intent['finish_time'] is None:
                ret = {'ret': {'Demand': intent['chat'], 'Chat': ''}}
            else:
                ret = {'ret': {'Demand': intent['ret'], 'Chat': ''}}
        return [ret]

    def _int_infer(self, chat: str = ''):
        submit_time = time.time()
        with self._lock:
            self._int_hist.append({
                'submit_time': submit_time,
                'finish_time': None,
                'chat': chat,
                'ret': None
            })

            int_his = self._int_hist[-2:]

            prep = prep_prompt(self._last_env, int_his, [], [], '')
            prep = deepcopy(prep)

            self._num_high_threads += 1

        try:
            js = request_client("Ei", self.setting.high_llm, prep)
        except:
            with self._lock:
                self._num_high_threads -= 1
            return
        # print(js)
        finish_time = time.time()

        # log
        self.replay.log("ai.int_infer",
                        {"prep": prep, "ret": js, "time_start": submit_time, "time_end": finish_time})

        with self._lock:
            self._num_high_threads -= 1
            if js is None:
                js = "None"
            # no cross-time period
            intent = [
                i for i in self._int_hist if i['submit_time'] == submit_time]
            if not intent:
                return
            intent = intent[0]
            intent['finish_time'] = finish_time
            intent['ret'] = js
            self._int_hist = self._int_hist[-self.INT_HIST_MAX_LEN:]

        self._llm_infer()

    def _llm_infer(self):
        submit_time = time.time()
        with self._lock:
            # if int is inferring, return
            if self._int_hist and self._int_hist[-1]['finish_time'] is None:
                return
            # no new chat checker
            if self._is_finished:
                intention = "None"
            else:
                i = self._int_hist[-1]
                # (or in other words: {i['ret']})
                intention = f"{i['ret']}" if self._int_hist else "None"
            int_time = self._int_hist[-1]['submit_time'] if self._int_hist else 0

            # prepare moves history
            mov_his = self._get_high_mov_hist()

            prep = prep_prompt(self._last_env, [], [], mov_his, intention)
            prep = deepcopy(prep)

            self._num_high_threads += 1

        try:
            js = request_client("El", self.setting.high_llm, prep)
        except:
            with self._lock:
                self._num_high_threads -= 1
            return
        finish_time = time.time()

        # log
        self.replay.log("ai.llm_infer",
                        {"prep": prep, "ret": js, "time_start": submit_time, "time_end": finish_time})

        with self._lock:
            self._num_high_threads -= 1

            if js is None:
                return
            if self._int_hist and self._int_hist[-1]['finish_time'] is None:
                return

            # check intention out date
            int_time2 = self._int_hist[-1]['submit_time'] if self._int_hist else 0
            if int_time != int_time2:
                return
            # check llm out date
            if self._llm_hist:
                llm2 = self._llm_hist[-1]
                if llm2['submit_time'] > submit_time:
                    return

            self._llm_hist.append({
                'submit_time': submit_time,
                'finish_time': finish_time,
                'chat': '',
                'ret': js
            })
            self._llm_hist = self._llm_hist[-self.LLM_HIST_MAX_LEN:]

    def high_level_infer(self, env: EnvState = None, chat: str = ''):
        if env is not None:
            with self._lock:
                self._last_env = env

        thread = threading.Thread(target=self._int_infer, args=(chat,))
        thread.daemon = True
        thread.start()

    def low_level_infer(self):
        submit_time = time.time()

        llm_his = self._get_mov_infer_prep()
        mov_his = self._get_low_mov_hist()

        prep = prep_prompt_s(self._last_env, [], llm_his, mov_his, '')
        # try:
        ht = request_client("Sm", self.setting.low_llm, prep)
        # except:
        #     return
        finish_time = time.time()

        self.replay.log("ai.mov_infer",
                        {"prep": prep, "ret": ht, "time_start": submit_time, "time_end": finish_time})
        self._task = ht
        self._mov_hist.append({'task': str(self._task), 'status': 'Ongoing. Initiated.',
                               'submit_time': submit_time, 'finish_time': finish_time})

    def _check_interrupt(self):
        # check llm incoming
        if self._llm_hist and self._lt_time < self._llm_hist[-1]['finish_time']:
            # obtain chat
            chat = self._llm_hist[-1]['ret']['Chat']
            self._lt_time = self._llm_hist[-1]['finish_time']
        else:
            chat = ''

        # check int incoming
        if self._int_hist and self._it_time < self._int_hist[-1]['submit_time']:
            self._it_time = self._int_hist[-1]['submit_time']
            if self._it_time > self._lt_time:
                if self._mov_hist and self._mov_hist[-1]['status'].startswith('Ongoing'):
                    self._mov_hist[-1]['status'] = 'Interrupted. '
                self._task = None
        if self._int_hist and self._int_hist[-1]['finish_time'] is not None \
                and self._it_time < self._int_hist[-1]['finish_time']:
            self._it_time = self._int_hist[-1]['finish_time']
            if self._it_time > self._lt_time:
                if self._mov_hist and self._mov_hist[-1]['status'].startswith('Ongoing'):
                    self._mov_hist[-1]['status'] = 'Interrupted. '
                self._task = None

        return chat

    def __call__(self, env: EnvState):
        self._lock.acquire()
        # update env
        self._last_env = env

        # check high level incoming
        chat = self._check_interrupt()

        # submit count check
        start_llm_infer = False
        if self._num_high_threads <= 0:
            if not self._llm_hist:
                start_llm_infer = True
            elif self._llm_hist[-1]['finish_time'] < time.time() - 5:
                start_llm_infer = True
            if not self._is_finished and self._mov_hist and self._mov_hist[-1]['submit_time'] != self._mt_time:
                start_llm_infer = True
                self._mt_time = self._mov_hist[-1]['submit_time']

        if start_llm_infer:
            self._lock.release()
            thread = threading.Thread(target=self._llm_infer)
            thread.daemon = True
            thread.start()
            self._lock.acquire()

        while True:
            if self._task is None:
                self.low_level_infer()
            if self._task is None:
                continue

            move_map = dict(left=(-1, 0), right=(1, 0),
                            up=(0, 1), down=(0, -1))
            move = move_map[self._task]

            self._mov_hist[-1]['status'] = f'Success.'
            self._task = None
            self._lock.release()
            return move, chat


def get_agent(sett: AgentSetting, replay: Replay):
    mapping = {
        "HLA": HLAagent,
        "SMOA": SMOAagent,
        "FMOA": FMOAagent,
        "NEA": NEAagent,
    }
    return mapping[sett.mode](sett, replay)
