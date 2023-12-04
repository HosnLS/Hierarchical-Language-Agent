import pickle
import time
import threading


class Replay:
    def __init__(self):
        self._d = {}
        self._d['dict'] = {}
        self._d['his'] = []
        # his
        #  - 1
        #  - 2
        self._lock = threading.Lock()

    @classmethod
    def from_file(cls, filename):
        self = cls()
        self._d = pickle.load(open(filename, 'rb'))
        return self

    def save(self, filename):
        pickle.dump(self._d, open(filename, 'wb'))

    def __getitem__(self, item):
        return self._d['dict'][item]

    def __setitem__(self, key, value):
        with self._lock:
            self._d['dict'][key] = value

    def log(self, name: str, args: dict):
        with self._lock:
            self._d['his'].append(
                {'time': time.time(), 'name': name, 'args': args})

    def __iter__(self):
        return iter(self._d['his'])
    