# POD configuration

The directory provides utilities to help configure and run distributed training on Graphcore IPU-PODs.
These are 3 scripts required to configure a pod and launch a run with poprun :
In order of launch :

- The first script (copy_ssh.sh) copies the ssh id to all the other hosts (should be done from all the hosts).
- The second script (config_pod.sh) configures the pod environment variables assuming a partition exists and allows to sync the code/SDK/venvs across all hosts.
- The third script (setup_poprun.sh) sets up the poprun final options and stores them in poprun_prefix to add in front of the training script.

## Example

Install first the sdk on your host under local home ($HOME).
Choose `sdks` and `venvs` respectively for the directory names of the sdk and the virtual environment.

Make sure that you link your local home into /localdata/ :
```
sudo mv $HOME /localdata/
sudo ln -s /localdata/$USER $HOME
```

Start copy of ssh id on the remote hosts (to be done from all the connected hosts) :
```
pod-xx:~/public_examples/utils/distributed_training$ ./copy_ssh <remote_hosts>
```

Test the MPI communication between hosts :

```
pod-xx:~/public_examples/utils/distributed_training$ mpirun --tag-output --prefix $OPAL_PREFIX --mca plm_rsh_args "-o BatchMode=yes" --allow-run-as-root --mca oob_tcp_if_include xx.xx.0.0/16 --mca btl_tcp_if_include xx.xx.0.0/16 --host <HOST1>,<HOST2>,...<HOSTN> hostname
```


Get the partition name :
```
pod-xx:~/public_examples/utils/distributed_training$ vipu-admin list partitions
```

**Source the script** (instead of executing it) :
```
pod-xx:~/public_examples/utils/distributed_training$ . config_pod.sh <partion_name> <remote_hosts>
pod-xx:~/public_examples/utils/distributed_training$ . setup_poprun.sh <cluster_name> 16 16 4
```
Run your command using poprun_prefix :
```
pod-xx:~/public_examples/vision/cnns/tensorflow2$ poprun_prefix python3 train.py --config resnet50_16ipus_8k_bn_pipeline --loss-scaling 0 --num-epochs 5 --clean-dir False --validation False
```


## Potential issues (to be removed for a public release)

The net mask in config_pod.sh might change in the future and then the script should also be adapted.
The directory where the user installed the sdk and virtual environment might be different that the one imposed in the config_pod.sh script (sdks and venvs).
IPUOF_VIPU_API_HOST exported variable relies on command `vipu-admin --server-version` that could change output in the future.
