from concern.config import Configurable, State
# from concern.log import Logger
from structure.builder import Builder
from structure.representers import *
# from structure.measurers import *
# from structure.visualizers import *
# from data.data_loader import *
# from data import *
# from training.model_saver import ModelSaver
# from training.checkpoint import Checkpoint
# from training.optimizer_scheduler import OptimizerScheduler


class Structure(Configurable):
    builder = State()
    representer = State()
    measurer = State()
    visualizer = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)

    @property
    def model_name(self):
        return self.builder.model_name


class ShowSettings(Configurable):
    data_loader = State()
    representer = State()
    visualizer = State()

    def __init__(self, **kwargs):
        self.load_all(**kwargs)


class Experiment(Configurable):
    structure = State(autoload=False)
    logger = State(autoload=True)

    def __init__(self, **kwargs):
        self.load('structure', **kwargs)

        cmd = kwargs.get('cmd', {})
        if 'name' not in cmd:
            cmd['name'] = self.structure.model_name

        self.load_all(**kwargs)
        self.distributed = cmd.get('distributed', False)
        self.local_rank = cmd.get('local_rank', 0)

        if cmd.get('validate', False):
            self.load('validation', **kwargs)
        else:
            self.validation = None
