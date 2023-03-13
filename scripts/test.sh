#!/usr/bin/env bash
cd ../
python main.py    --logging-dir /workspaces/code/neurips_code/logging/ \
                  --model-dir /workspaces/code/neurips_code/models/ \
                  --model-name augmented_language_model_repeated\
                  --augment-data 0\
                  --load-model 1\
                  --version 2 \
                  --data_path webnlg-dataset/release_v3.0/en \
                  --num_data_workers 10 \
                  --run test \
                  --gpu-devices 2 \
                  --batch_size 55 \
                  --learning-rate 1e-4 \
                  --max_epochs 100 \
                  --accelerator gpu \
                  --num_nodes 1 \
                  --num_sanity_val_steps 0 \
                  --fast_dev_run 0 \
                  --overfit_batches 0 \
                  --limit_train_batches 1.0 \
                  --limit_val_batches 1.0 \
                  --limit_test_batches 1.0 \
                  --accumulate_grad_batches 2 \
                  --detect_anomaly True \
                  --log_every_n_steps 100 \
