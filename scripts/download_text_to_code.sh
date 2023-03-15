#!/usr/bin/env bash

cd $1
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
unzip python.zip
gzip -d python/final/jsonl/test/python_test_0.jsonl.gz
gzip -d python/final/jsonl/valid/python_valid_0.jsonl.gz
gzip -d python/final/jsonl/train/python_train_7.jsonl.gz
gzip -d python/final/jsonl/train/python_train_6.jsonl.gz
gzip -d python/final/jsonl/train/python_train_12.jsonl.gz
gzip -d python/final/jsonl/train/python_train_13.jsonl.gz
gzip -d python/final/jsonl/train/python_train_0.jsonl.gz
gzip -d python/final/jsonl/train/python_train_1.jsonl.gz
gzip -d python/final/jsonl/train/python_train_4.jsonl.gz
gzip -d python/final/jsonl/train/python_train_5.jsonl.gz
gzip -d python/final/jsonl/train/python_train_9.jsonl.gz
gzip -d python/final/jsonl/train/python_train_8.jsonl.gz
gzip -d python/final/jsonl/train/python_train_11.jsonl.gz
gzip -d python/final/jsonl/train/python_train_10.jsonl.gz
gzip -d python/final/jsonl/train/python_train_3.jsonl.gz
gzip -d python/final/jsonl/train/python_train_2.jsonl.gz
mv python/final/jsonl/test/python_test_0.jsonl  $1
mv python/final/jsonl/valid/python_valid_0.jsonl  $1
mv python/final/jsonl/train/python_train_7.jsonl  $1
mv python/final/jsonl/train/python_train_6.jsonl  $1
mv python/final/jsonl/train/python_train_12.jsonl  $1
mv python/final/jsonl/train/python_train_13.jsonl  $1
mv python/final/jsonl/train/python_train_0.jsonl  $1
mv python/final/jsonl/train/python_train_1.jsonl  $1
mv python/final/jsonl/train/python_train_4.jsonl  $1
mv python/final/jsonl/train/python_train_5.jsonl  $1
mv python/final/jsonl/train/python_train_9.jsonl  $1
mv python/final/jsonl/train/python_train_8.jsonl  $1
mv python/final/jsonl/train/python_train_11.jsonl  $1
mv python/final/jsonl/train/python_train_10.jsonl  $1
mv python/final/jsonl/train/python_train_3.jsonl  $1
mv python/final/jsonl/train/python_train_2.jsonl  $1
rm -r python
rm python.zip
