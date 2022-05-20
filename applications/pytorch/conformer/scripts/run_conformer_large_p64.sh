poprun -vv  \
	--vipu-partition=pod64  \
	--host lr17-1-poplar-1,lr17-1-poplar-2,lr17-1-poplar-3,lr17-1-poplar-4 \
	--update-partition=yes \
	--print-topology=yes \
	--mpi-global-args="--tag-output --allow-run-as-root  --mca oob_tcp_if_include 10.5.17.0/24 --mca btl_tcp_if_include 10.5.17.0/24 " \
	--num-replicas=16 \
	--numa-aware=yes \
	--num-ilds=4 \
	--num-instances=4 \
	--ipus-per-replica=4 \
	python3 main.py train \
	--config_file configs/train_large.yaml \
	--train_dataset.use_generated_data True \
	--ipu_options.gradient_accumulation 336  \	
