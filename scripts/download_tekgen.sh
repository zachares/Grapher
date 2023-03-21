#!/usr/bin/env bash

mkdir $1/tekgen
wget https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/quadruples-train.tsv -P $1/tekgen/
wget https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/quadruples-validation.tsv -P $1/tekgen/
wget https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/quadruples-test.tsv -P $1/tekgen/
mv $1/tekgen/quadruples-validation.tsv $1/tekgen/quadruples-val.tsv
python process_tekgen_data.py --dataset-path $1/tekgen/
rm $1/tekgen/quadruples-train.tsv
rm $1/tekgen/quadruples-val.tsv
rm $1/tekgen/quadruples-test.tsv
