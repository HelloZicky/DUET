import time


class Timer:
    def __init__(self):
        self._start = time.time()
        self._last = self._start

    def tick(self, round=True):
        ret_value = time.time() - self._last
        self._last += ret_value
        if round:
            ret_value = int(ret_value)
        return ret_value

    def total_time(self):
        return time.time() - self._start
