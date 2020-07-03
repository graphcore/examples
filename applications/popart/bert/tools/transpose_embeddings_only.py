"""When running multi-checkpoint SQuAD, the embedding is created transposed, compared to in pretraining
to ensure optimal layout. When running a single instance, the transpose is automatically handled by
the intialisation code, but for multi-checkpoint SQuAD, we directly load the weights from the ONNX
checkpoint. This means we'll get a Popart runtime error due to a dimension mis-match on the embedding.
This script pre-processes the pretrained checkpoints to tranpose the embedding, so they can be directly
loaded into the Popart session."""

import argparse
import multiprocessing
import os
import glob
import sys
import popart
import onnx
from pathlib import Path

# Add the bert root to the PYTHONPATH
bert_root_path = str(Path(__file__).parent.parent)
sys.path.append(bert_root_path)

import bert
import utils


def parse_args():
    pparser = argparse.ArgumentParser()
    pparser.add_argument("--checkpoint-dir", type=str, required=True)
    pparser.add_argument("--output-dir", type=str, required=True)
    pparser.add_argument("--model-search-string", type=str, default="*.onnx")
    pparser.add_argument("--num-processes", type=int, default=1)
    args, remaining = pparser.parse_known_args()
    assert args.checkpoint_dir != args.output_dir, "Output directory cannot be the same as input directory"
    return args, remaining


def transpose_checkpoint_embedding(checkpoint):
    print(f"Loading checkpoint: {checkpoint}")
    initializers = utils.load_initializers_from_onnx(checkpoint)
    model = bert.Bert(config,
                      builder=popart.Builder(
                          opsets={"ai.onnx": 9, "ai.onnx.ml": 1, "ai.graphcore": 1}),
                      initializers=initializers,
                      execution_mode=args.execution_mode)

    indices, positions, segments, masks, _ = bert.bert_add_inputs(args, model)
    bert.bert_logits_graph(model, indices, positions, segments, masks)

    proto = model.builder.getModelProto()
    onnx_proto = onnx.load_from_string(proto)

    output_path = os.path.join(pargs.output_dir, os.path.basename(checkpoint))
    print(f"Saving to: {output_path}")
    onnx.save(onnx_proto, output_path)


if __name__ == "__main__":
    pargs, rem = parse_args()

    if "--config" not in rem:
        print("Please specify the target config using the '--config' parameter (e.g. --config configs/squad_base_384.json")
        sys.exit(1)

    args = utils.parse_bert_args(rem)
    config = bert.bert_config_from_args(args)

    os.makedirs(pargs.output_dir, exist_ok=True)

    checkpoint_files = glob.glob(os.path.join(pargs.checkpoint_dir, pargs.model_search_string))

    pool = multiprocessing.Pool(pargs.num_processes)
    pool.map(transpose_checkpoint_embedding, checkpoint_files)
