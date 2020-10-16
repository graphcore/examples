# Copyright 2020 Graphcore Ltd.
import torch
import time
import torch.multiprocessing as multiprocessing
import poptorch
import torchvision
from torchvision import transforms
import poptorch


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class ProcessPoolDataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self,
                 options,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 num_workers=0, *args, **kwargs):
        self._combined_batch_size = batch_size * \
            options.device_iterations * \
            options.replication_factor * \
            options.Training.gradient_accumulation

        super().__init__(dataset,
                         batch_size=self._combined_batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    @property
    def combinedBatchSize(self):
        return self._combined_batch_size

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class AsyncDataLoader:
    @staticmethod
    def fetch_data(queue, dataloader, buffer_size, transform=None):
        # We put the shared buffer into the queue to send it to the main process
        dataloader = dataloader()
        # send the length of the dataset
        queue.put(len(dataloader))
        index = 0
        is_initialized = False
        is_finished = torch.tensor([False], dtype=torch.bool).contiguous().share_memory_()
        queue.put(is_finished)
        while not is_finished[0]:
            for _, (data, label) in enumerate(dataloader):
                if is_finished[0]:
                    break
                if not is_initialized:
                    data_buffer = torch.zeros((buffer_size,) + (data.size()))
                    if transform is not None:
                        data_buffer = transform(data_buffer)
                    data_buffer = data_buffer.contiguous().share_memory_()
                    queue.put(data_buffer)
                    labels_buffer = torch.zeros(buffer_size, int(data.size()[0]), dtype=torch.long).contiguous().share_memory_()
                    queue.put(labels_buffer)
                    is_dirty = torch.tensor([True]*buffer_size, dtype=torch.bool).contiguous().share_memory_()
                    queue.put(is_dirty)
                    is_initialized = True
                else:
                    element_ready = False
                    while not element_ready and not is_finished[0]:
                        for i in range(0, buffer_size):
                            if is_dirty[i]:
                                element_ready = True
                                index = i
                                if transform is None:
                                    transformed_data = transform(data)
                                else:
                                    transformed_data = data
                                data_buffer[index].copy_(transformed_data)
                                labels_buffer[index].copy_(label)
                                is_dirty[index] = False
                                break
                        if not element_ready:
                            time.sleep(0.01)


    def __init__(self, dataloader, transform=None, buffer_size=3):
        queue = multiprocessing.Queue()
        self.buffer_size = buffer_size
        self.data_fetcher = multiprocessing.Process(name='Async DataLoader', target=AsyncDataLoader.fetch_data, args=(queue, dataloader, buffer_size, transform, ))
        self.data_fetcher.start()
        self._length = queue.get(block=True)
        self._is_finished = queue.get(block=True)
        self.data_buffer = queue.get(block=True)
        self.labels_buffer = queue.get(block=True)
        self.buffer_is_dirty = queue.get(block=True)

    def __len__(self):
        return self._length

    def __iter__(self):
        index = 0
        batch_id = 0
        while True:
            element_ready = False
            while not element_ready:
                # Grab the first piece of data avaliabe.
                for i in range(0, self.buffer_size):
                    if not self.buffer_is_dirty[i]:
                        element_ready = True
                        index = i
                        yield self.data_buffer[index], self.labels_buffer[index]
                        batch_id += 1
                        if batch_id == self._length:
                            return
                        self.buffer_is_dirty[index] = True

    def stop_data_fetch(self):
        self._is_finished[0] = True
