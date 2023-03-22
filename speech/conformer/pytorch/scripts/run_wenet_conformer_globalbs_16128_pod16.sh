python3
    main.py  \
    train  \
    --trainer.log_every_n_step 1  \
    --train_dataset.use_generated_data true  \
    --trainer.num_epochs=2  \
    --ipu_options.num_replicas=4  \
    --ipu_options.gradient_accumulation=1008  \
    --train_iterator.batch_size=4  \
