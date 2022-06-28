import numpy as np
import h5py
from dynsys_framework.dynamic_system.model_executor import ModelExecutor
from dynsys_framework.dynamic_system.process import Process
from typing import Dict, List, Union, Optional, DefaultDict, Callable
from collections import defaultdict
import threading
import os
import queue
import time

RawRecord = Dict[str, list]
State = Dict[str, Union[float, int, np.ndarray]]
NumpyRecord = Union[DefaultDict[str, np.ndarray], Dict[str, np.ndarray]]
"""
Saving and loading records.
A record is a dictionary string: list(numpy.array).
List of records is refered to as 'records'.
It is used to save/load states of the model during the run.
"""


class Recorder(object):
    def __init__(self, results_path: str, target_experiment_name: str, capacity: int, record_initter: Callable[[], RawRecord], dtype=np.float32):
        self.capacity = capacity
        self.record_queue = queue.Queue(1000)

        self.dtype = dtype
        self.record_initter = record_initter
        self.experiment_name = target_experiment_name
        self.results_path = results_path
        self._i = 0
        self.stored_iters = []
        self.buffer = record_initter()
        self.running_flag = True

    def has_stored_iters(self):
        return len(self.stored_iters) > 0
    
    def append(self, state: State):
        append_state(self.buffer, state)
        self._i += 1
        if self._i % self.capacity == self.capacity - 1:
            try:
                self.record_queue.put((self.buffer, self._i), timeout=0.1)
                self.stored_iters.append(self._i)
            except Exception as e:
                print("ERROR: The Records queue is full. Current batch {} will be not stored : {}".format(self._i, e))
            self.buffer = self.record_initter()

    def flush_and_stop(self):
        try:
            self.record_queue.put((self.buffer, self._i), timeout=0.1)
            self.stored_iters.append(self._i)
        except Exception as e:
            print("ERROR: The Records queue is full. Current batch {} will be not stored : {}".format(self._i, e))
        self.buffer = self.record_initter()
        self.running_flag = False

    def merge(self, delete_iteration_record_fragments=False):
        merge_iterative_records(self.results_path, self.experiment_name, self.stored_iters)
        if delete_iteration_record_fragments:
            for iter_num in self.stored_iters:
                name = self.experiment_name + "_i{}".format(iter_num)
                os.remove(os.path.join(self.results_path, name +".hdf5"))
                os.remove(os.path.join(self.results_path, name +"_lite.hdf5"))


def recorder_consumer(recorder: Recorder, queue_timeout: float, waiting_cycle_secs: float, ignore_gradients=False):
    while recorder.running_flag or not recorder.record_queue.empty():
        try:
            raw_record, iter_num = recorder.record_queue.get(timeout=queue_timeout)
            name = recorder.experiment_name + "_i{}".format(iter_num)
            store_task(raw_record, recorder.results_path, name, save_lite=True, dtype=recorder.dtype, ignore_gradients=ignore_gradients)
        except Exception as e:
            pass
        time.sleep(waiting_cycle_secs)


def store_task(record: RawRecord, results_path, target_experiment_name, save_lite=True, dtype=np.float32, ignore_gradients=False):
    record_np = numpyfy_record(record, dtype=dtype, ignore_gradients=ignore_gradients)
    save_records(os.path.join(results_path, target_experiment_name + ".hdf5"), record_np)
    if save_lite:
        size = len(record_np[list(record_np.keys())[0]])
        save_records(
            os.path.join(results_path, target_experiment_name + "_lite.hdf5"),
            crop_record(record_np, max(0, size - 100), size - 1))


def merge_iterative_records(result_directory: str, name_prefix: str, iterations: List[int], extra_postfix="", debug=False):
    super_record = None
    for iter_num in iterations:
        pth = os.path.join(result_directory, name_prefix + "_i{}.hdf5".format(iter_num))
        if debug:
            print("Merging {}_i{}.hdf5".format(name_prefix, iter_num))
        if super_record is None:
            super_record = load_records(pth)[0]
        else:
            tmp = load_records(pth)[0]
            merge_records(super_record, tmp)
    pth = os.path.join(result_directory, name_prefix + extra_postfix + ".hdf5")
    pth_lite = os.path.join(result_directory, name_prefix + extra_postfix + "_lite.hdf5")
    save_records(pth, super_record)
    save_records(
        pth_lite,
        crop_record(super_record, max(0, iterations[-1] - 100), iterations[-1] - 1))


def merge_records(record: NumpyRecord, append_record: NumpyRecord):
    for k in record:
        record[k] = np.concatenate((record[k], append_record[k]))


def init_record(model_exec: ModelExecutor) -> RawRecord:
    return dict((k, []) for k in model_exec.get_last_state())


def append_state(record: RawRecord, state: State):
    for k in record:
        record[k].append(state[k])


def numpyfy_record(record: RawRecord, dtype=np.float32, ignore_gradients=False) -> NumpyRecord:
    record_np = {}
    for k in record:
        if ignore_gradients and k[0:2] == Process.DERIVATIVE_PREFIX:
            continue
        try:
            record_np[k] = np.asarray(record[k],dtype=dtype)
        except Exception as e:
            print("Failed to numpyfy key {}, skipping. Reason: {}".format(k, e))
    return record_np


def crop_record(record_np: NumpyRecord, start: int, end: int) -> NumpyRecord:
    for k in record_np:
        record_np[k] = record_np[k][start:end]
    return record_np


def print_record_shapes(record: NumpyRecord):
    for k in record:
        print("{}:          {}".format(k, record[k].shape))


def save_records(path: str, records: NumpyRecord):
    if isinstance(records, dict):
        records = [records]
    assert isinstance(records, list)
    assert isinstance(records[0], dict)
    with h5py.File(path, "w") as f:
        cou = 0
        for record in records:
            record_group = f.create_group(str(cou))
            cou += 1
            for k in record:
                data = record[k]
                if not isinstance(data, np.ndarray):
                    data = np.asarray(data)
                data.astype(dtype=np.float32)
                record_group.create_dataset(k, shape=data.shape, dtype=np.float32, data=data)


def _get_max_shape(group):
    shape = [0, 0, 0, 0]
    for var_key in group.keys():
        if group[var_key].ndim > len(shape):
            shape += [0]*(group[var_key].ndim - len(shape))
        for i in range(group[var_key].ndim):
            if shape[i] < group[var_key].shape[i]:
                shape[i] = group[var_key].shape[i]
    return tuple(shape)


def _get_default_array_factory(shape):
    def default_array():
        print("Using default array")
        return np.zeros(shape)
    return default_array


def load_records(path: str, to_working_memory=True, default_dict=True) -> List[NumpyRecord]:
    ret = []
    f = h5py.File(path, 'r')
    for record_key in f.keys():
        group = f[record_key]
        if default_dict:
            record = defaultdict(_get_default_array_factory(_get_max_shape(group)))
        else:
            record = {}
        for var_key in group.keys():
            if to_working_memory:
                record[var_key] = np.asarray(group[var_key])
            else:
                record[var_key] = group[var_key]
        ret.append(record)
    if to_working_memory:
        f.close()
    return ret


def load_init_state_from_record(param_record: NumpyRecord, init_state:State,
                                variables_to_set: Optional[List[str]] = None,
                                from_iteration: int = -1):
    if variables_to_set is None:
        variables_to_set = init_state.keys()
    for k in variables_to_set:
        if k not in param_record.keys():
            print("WARNING: no param {} in loaded init record. Its default init will be used.".format(k))
        else:
            init_state[k] = param_record[k][from_iteration]


def load_record_and_extract(path: str, variables: List[str]):
    record = load_records(path, to_working_memory=True, default_dict=False)[0]
    ret = {}
    for varb in variables:
        ret[varb] = np.zeros(record[varb].shape)
        ret[varb] += record[varb]
    del record
    return ret


def extract_from_record_files(paths: List[str], variables: List[str]):
    return tuple(map(lambda x: load_record_and_extract(x, variables), paths))