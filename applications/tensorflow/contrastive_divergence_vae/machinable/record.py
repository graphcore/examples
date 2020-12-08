# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

import pendulum
from collections import OrderedDict
import csv
import pickle
import datetime

from machinable.utils import prettydict, msg


class Record(object):
    """Tabular record writer

    # Arguments
    observer: machinable.observer.Observer, parent observer instance
    config: dict, configuration options
    scope: String, optional destination name
    """

    def __init__(self, observer, config=None, scope='default'):
        self.observer = observer
        self.config = config if config is not None else {}
        self.scope = scope
        self.history = []
        self._record = OrderedDict()
        # create records directory
        self.observer.get_path('records', create=True)

    def write(self, key, value, fmt=None):
        """Writes a cell value

        Note that you can use array notation to write value as well, for example

        ```python
        self.record.write('loss', 0.1)
        # is equivalent to
        self.record['loss'] = 0.1
        ```

        # Arguments
        key: String, the column name
        value: Value to write
        fmt: String, optional formatter.
        """
        if fmt == 'seconds':
            value = str(datetime.timedelta(seconds=value))

        self._record[key] = value

    def update(self, dict_like=None, **kwargs):
        """Update record values using a dictionary. Equivalent to dict's update method.

        # Arguments
        dict_like: dict, update values
        """
        self._record.update(dict_like, **kwargs)

    def empty(self):
        """Whether the record writer is empty (len(self.record) == 0)
        """
        return len(self._record) == 0

    def timing(self, mode='iteration', timestamp=None, return_type='seconds'):
        """Get timing statistics about the records

        # Arguments
        mode: String, 'iteration', 'avg' or 'total' that determines whether the last iteration, the average iteration
              or the total time is collected
        timestamp: Mixed, end of the timespan; if None, datetime.now() is used
        return_type: String, specifying the return format 'seconds', 'words', or 'period' (pendulum.Period object)

        Returns: See return_type
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()

        if mode == 'iteration':
            if len(self.history) > 0:
                start = self.history[-1]['_timestamp']
            else:
                start = self.observer.statistics['on_run_start']
        elif mode == 'total':
            start = self.observer.statistics['on_run_start']
        elif mode == 'avg':
            # todo
            raise NotImplementedError('Mode avg is currently not implemented.')
        else:
            raise ValueError(f"Invalid mode: '{mode}'; must be 'iteration', 'avg' or 'total'.")

        elapsed = pendulum.instance(start).diff(timestamp)

        if return_type == 'seconds':
            seconds = elapsed.in_seconds()
            if seconds == 0:
                return 1e-15  # avoids division by zero errors
            return seconds
        elif return_type == 'words':
            return elapsed.in_words()
        elif return_type == 'period':
            return elapsed
        else:
            raise ValueError(f"Invalid return_type: '{return_type}'; must be 'seconds', 'words' or 'period'.")

    def save(self, echo=False, force=False):
        """Save the current row

        # Arguments
        echo: Boolean, if True the written row will be printed to the terminal
        force: Boolean, if True the row will be written even if it contains no columns

        # Returns
        The row data
        """
        data = self._record.copy()
        self._record = OrderedDict()

        # don't save if there are no records
        if len(data) == 0 and not force:
            return data

        # meta-data
        data['_timestamp'] = datetime.datetime.now()

        total_time = self.timing('total', timestamp=data['_timestamp'], return_type='period')
        data['_total_seconds_taken'] = total_time.in_seconds()
        data['_total_time_taken'] = total_time.in_words()

        self.history.append(data)

        if self.scope == 'default':
            self.observer.events.trigger('observer.record.on_save', data)

        # write human-readable csv
        with self.observer.get_stream(f'records/{self.scope}.csv', 'a') as f:
            writer = csv.writer(f)

            if len(self.history) == 1:
                writer.writerow(data.keys())
            writer.writerow(data.values())

        # write pickle version
        with self.observer.get_stream(f'records/{self.scope}.p', 'wb') as f:
            pickle.dump(self.history, f)

        if self.scope == 'default':
            self.observer.events.trigger('observer.on_change', 'record.save')

        if echo:
            msg(f"{self.observer.config['group']}:{self.observer.config.get('uid', '')}")
            msg(prettydict(data, sort_keys=True))

        return data

    def __len__(self):
        return len(self._record)

    def __delitem__(self, key):
        del self._record[key]

    def __getitem__(self, key):
        return self._record[key]

    def __setitem__(self, key, value):
        self._record[key] = value
