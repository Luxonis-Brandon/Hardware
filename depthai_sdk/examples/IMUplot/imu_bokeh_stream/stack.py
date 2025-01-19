from collections import deque
from statistics import mean
from threading import Lock


class RollingStack(deque):
    def __init__(self, stack_size=3, init_val={}) -> None:
        deque.__init__(self, maxlen=stack_size)
        self.append(init_val)
        self.lock = Lock()

    def latest(self):
        # not all deque functions are threadsafe
        with self.lock:
            return self[-1]

    def all(self):
        return list(self)
