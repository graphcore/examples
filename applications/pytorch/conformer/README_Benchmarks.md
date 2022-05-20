# Benchmarking on IPUs

This README describes how to run the Conformer model for throughput benchmarking on the Mk2 IPU.


## Preparation

Follow the installation instructions in applications/pytorch/conformer/README.md.

## Training

### Conformer small fp16 1 x POD16 

Command:
```console
        python3 \
            main.py \
            train \
            --trainer.log_every_n_step 1 \
            --train_dataset.use_generated_data true \
            --trainer.num_epochs=5 \
            --ipu_options.num_replicas=8 \
            --ipu_options.gradient_accumulation=3 \
            --train_iterator.batch_size=2 
```

### Conformer large fp16 1 x POD16 

Command:
```console
       python3 \
            main.py \
            train \
            --config_file configs/train_large.yaml  \
            --train_dataset.use_generated_data true \
```

### Conformer large fp16 1 x POD64


In order to run the command below, the following environment variables should be set:

PARTITION, IPUOF_VIPU_API_PARTITION_ID=ID of the Pod64 reconfigurable partition
TCP_IF_INCLUDE=sets the default route for traffic between Poplar hosts. It should be configured for a network to which all Poplar hosts have access, and for which the interfaces only have a single IP address.
VIPU_CLI_API_HOST=IP address of the host server where the VIPU controller is running
HOSTS=IP address of Poplar hosts

Command:
```console
        poprun \
            -vv \
            --host $HOSTS \
            --mpi-global-args="--allow-run-as-root --tag-output --mca oob_tcp_if_include $TCP_IF_INCLUDE --mca btl_tcp_if_include $TCP_IF_INCLUDE" \
            --mpi-local-args="-x OPAL_PREFIX -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -x IPUOF_VIPU_API_TIMEOUT=600 -x POPLAR_LOG_LEVEL=WARN -x TF_POPLAR_FLAGS -x DATASETS_DIR -x POPLAR_ENGINE_OPTIONS -x POPLAR_RUNTIME_OPTIONS" \
            --update-partition=yes \
            --remove-partition=no \
            --reset-partition=no \
            --vipu-server-timeout 300 \
            --vipu-server-host "$VIPU_CLI_API_HOST" \
            --vipu-cluster=$CLUSTER \
            --vipu-partition=$PARTITION \
            --print-topology=yes \
            --num-replicas=16 \
            --ipus-per-replica=4 \
            --num-ilds=4 \
            --num-instances=4 \
            --numa-aware=yes \
            python3 main.py train \
            --config_file configs/train_large.yaml \
            --train_dataset.use_generated_data True \
            --ipu_options.gradient_accumulation 336
```