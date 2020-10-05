This folder contains code to run graph streaming bert (currently experimental).

To run gs-bert SQUAD on multiple replicas with replicated weight-sharding:
python bert.py --config=configs/squad_large_128_pingpong_rws.json --synthetic-data --epochs 10 --no-model-save --no-validation --steps-per-log 1 --momentum=0.0 --replication-factor=8`

