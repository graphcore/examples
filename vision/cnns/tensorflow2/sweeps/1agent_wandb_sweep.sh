#!/usr/bin/env bash

date;hostname;pwd;

N_RUNS=20

while [[ $# -gt 0 ]]; do
  case $1 in
    --sweep-id)
      SWEEP_ID="$2"
      shift; shift
      ;;
    --n-runs)
      N_RUNS="$2"
      shift; shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# check if sweep was set
if [ ${#POSITIONAL_ARGS} -gt 0 ];
then echo "Error. Unknown arguments: $POSITIONAL_ARGS[@]"; exit 1;
fi

# check if popun related args were set, otherwise the agent will fail
if [ -z "${POPRUN_HOSTS}" ];
then echo "Environment variable POPRUN_HOSTS must be set. Exiting..."; exit;
else echo "Running wandb on the following CPU host machines: ${POPRUN_HOSTS}";
fi

if [ -z "${IPUOF_VIPU_API_HOST}" ];
then echo "Environment variable IPUOF_VIPU_API_HOST must be set. Exiting..."; exit;
else echo "IPUOF_VIPU_API_HOST=${IPUOF_VIPU_API_HOST}";
fi

if [ -z "${IPUOF_VIPU_API_PARTITION_ID}" ]
then echo "Environment variable IPUOF_VIPU_API_PARTITION_ID must be set. Exiting..."; exit;
else echo "IPUOF_VIPU_API_PARTITION_ID=${IPUOF_VIPU_API_PARTITION_ID}";
fi

LOGGING_DIR="$(pwd)/logs"
mkdir -p $LOGGING_DIR

FILENAME=$(echo "${SWEEP_ID}" | tr '/' '_')

wandb login --host=https://wandb.sourcevertex.net/

echo $TMPDIR
echo $LOGGING_DIR
echo $SWEEP_ID
echo $FILENAME
echo $N_RUNS

wandb agent --count=${N_RUNS} ${SWEEP_ID} > $LOGGING_DIR/${FILENAME}_agent1.txt 2>&1 &

wait