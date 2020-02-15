from collections import namedtuple
from itertools import product


class RunBuilder:
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        return [Run(*v) for v in product(*params.values())]
