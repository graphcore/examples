# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import math
import logging

import popart
import popdist


__all__ = ["acquire_device"]

logger = logging.getLogger(__name__)


def _request_ipus(num_ipus):
    return pow(2, math.ceil(math.log2(num_ipus)))


def get_ipu_model(args, request_ipus):
    opts = {"numIPUs": request_ipus}
    if args.device_version is not None:
        opts["ipuVersion"] = args.device_version

    return get_device_manager(args).createIpuModelDevice(opts)


def get_offline_device(args, request_ipus):
    opts = {
        "numIPUs": request_ipus,
        "syncPattern": get_sync_pattern(args)
    }
    if args.device_tiles:
        opts["tilesPerIPU"] = args.device_tiles
    if args.device_version:
        opts["ipuVersion"] = args.device_version

    return get_device_manager(args).createOfflineIPUDevice(opts)


def get_device_manager(args):
    manager = popart.DeviceManager()
    manager.setOnDemandAttachTimeout(args.device_ondemand_timeout)
    return manager


def get_connection_type(args):
    if args.device_connection_type == "ondemand":
        return popart.DeviceConnectionType.OnDemand

    return popart.DeviceConnectionType.Always


def get_sync_pattern(args):

    return popart.SyncPattern.Full


def get_device_by_id(args, request_ipus):
    device = get_device_manager(args).acquireDeviceById(
        args.device_id,
        pattern=get_sync_pattern(args),
        connectionType=get_connection_type(args))

    if device is not None and device.numIpus != request_ipus:
        raise RuntimeError(f"Number of IPUs in device selected by id ({device.numIpus}) does not match"
                           f" the required IPUs from the model configuration ({request_ipus})")

    return device


def get_available_device(args, request_ipus):
    return get_device_manager(args).acquireAvailableDevice(
        request_ipus,
        pattern=get_sync_pattern(args),
        connectionType=get_connection_type(args))


def get_popdist_device(args, request_ipus, runtime_args):
    ipus_per_replica = request_ipus // runtime_args.local_replication_factor
    if not popdist.checkNumIpusPerReplica(ipus_per_replica):
        raise RuntimeError(f"The number IPUs per replica ({ipus_per_replica}) required "
                           f"for the model configuration does not match the specified "
                           f"popdist IPUs per replica ({popdist.getNumIpusPerReplica()}) "
                           f"ipus_per_replica = {ipus_per_replica}, "
                           f"replication_factor = {runtime_args.local_replication_factor}")
    args.device_id = popdist.getDeviceId(ipus_per_replica)
    return get_device_by_id(args, request_ipus)


def _acquire_device(args, num_ipus, runtime_args):
    request_ipus = _request_ipus(num_ipus)
    if runtime_args.use_popdist:
        logger.info(f"Need {num_ipus} IPUs per instance. Requesting {request_ipus} IPUs per instance.")
    else:
        logger.info(f"Need {num_ipus} IPUs. Requesting {request_ipus} IPUs.")

    if args.use_ipu_model:
        return get_ipu_model(args, request_ipus)

    if args.device_connection_type == "offline":
        return get_offline_device(args, request_ipus)

    if runtime_args.use_popdist:
        return get_popdist_device(args, request_ipus, runtime_args)

    if args.device_id is not None:
        return get_device_by_id(args, request_ipus)

    return get_available_device(args, request_ipus)


def acquire_device(args, num_ipus, runtime_args):
    device = _acquire_device(args, num_ipus, runtime_args)
    if device is None:
        raise OSError("Failed to acquire {} IPUs.".format(num_ipus))
    logger.info(f"Acquired device: {device}")
    return device
