#!/usr/bin/env bash
!/usr/bin/env bash

mkdir $1/webnlg2020

git clone https://gitlab.com/webnlg/corpus-reader.git corpusreader
git clone https://gitlab.com/shimorina/webnlg-dataset.git
pip install networkx

python process_webnlg2020_data.py --dataset-path $1/webnlg2020

rm -r corpusreader
rm -r webnlg-dataset

mv $1/webnlg2020/dev.json $1/webnlg2020/val.json
