# Graphcore

## Tensorflow Checkpoint/Frozen Graph loader unit test

To run this test, you need to provide a checkpoint and a BERT config file.
You can also optionally provide a frozen graph to test the frozen-graph loading
functionalty.

The test is run through `pytest` and will require `tensorflow` or `gc-tensorflow`
to installed in your environment. 

### Required Data

The checkpoint and config file can be downloaded from [Google Research's BERT
github](https://github.com/google-research/bert): 

https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

To run the frozen-graph loading test, you'll need to produce the frozen graph from the check-point - see the "Creating the Frozen Graph" section for more information.

### Running the Checkpoint Load Test

	$ PYTHONPATH=$PYTHONPATH:. pytest tests/unit/tensorflow/
		--config-path <PATH_TO_BERT_DATA>/bert_config.json \
		--chkpt-path <PATH_TO_BERT_DATA>/bert_model.ckpt

### Running the Frozen Graph Load Test

The BERT checkpoint provided by Google Research only includes the 
checkpoint data and a config. You will need generate a frozen graph from
the checkpoint as described in "Creating the Frozen Graph" below.

Note that in this case, the checkpoint provided should be the one used
to generate the frozen graph, not the one downloaded from the Google
Research repo.

	$ PYTHONPATH=$PYTHONPATH:. pytest tests/unit/tensorflow/
		--config-path <PATH_TO_BERT_DATA>/bert_config.json \
		--chkpt-path <PATH_TO_CHECKPOINT>/model.ckpt-0 \
		--frozen-path <PATH_TO_EXPORT_DIR>/frozen.pb

### Creating the Frozen Graph

To create the frozen graph, we need a complete checkpoint to feed into the
freeze_graph method - the one provided by Google Research is missing key
information to produce the frozen graph.

1. Clone the Google Research repo:

	$ git clone https://github.com/google-research/bert.git

2. Run your training/fine-tuning to generate a complete checkpoint (see the
   README in the Google Research BERT repo).

3. Take the saved checkpoint and run it through Tensorflow's Freeze Graph
   utility:

	$ freeze_graph --input_meta_graph=<PATH_TO_CHECKPOINT>/model.ckpt-0.meta \ 
		--input_checkpoint=<PATH_TO_CHECKPOINT>/model.ckpt-0 \
		--output_node_names="add_1" \
		--clear_devices \
		--output_graph=<PATH_TO_EXPORT_DIR>/frozen.pb \ 
		--input_binary=True

Note that this example assumes running the pretraining to generate the
checkpoint, otherwise output nodes may vary.
