# Copyright 2019 Graphcore Ltd.
from importlib import reload
import logging


class Log(object):

    def __init__(self, observer, config=None):
        self.observer = observer
        self.config = config or {'level': 'INFO'}
        self.logger = self._get_logger()

    def _get_logger(self):
        # make sure logging is fresh and not affected by other imports
        reload(logging)
        task_id = self.observer.config.get('group', '')
        logging.basicConfig(
            level=self.config['level'],
            format=f"{task_id}; %(asctime)s; %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        logger = logging.getLogger('machinable')

        return logger

    # forward function calls to logger

    def __getattr__(self, item):
        def forward(*args, **kwargs):
            method = getattr(self.logger, item)
            return method(*args, **kwargs)

        return forward
