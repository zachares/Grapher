#!/usr/bin/env bash

mkdir $1/genwiki
pip install gdown
gdown 1zGyjKHCYJBkM7EtF9halhI2wuggaxLMp -O $1/genwiki/dataset.zip
unzip $1/genwiki/dataset.zip -d $1/genwiki/
mv $1/genwiki/genwiki/* $1/genwiki
rm -r $1/genwiki/genwiki/
python process_genwiki_data.py --dataset-path $1/genwiki/
rm -r $1/genwiki/train/
rm -r $1/genwiki/test/
rm $1/genwiki/dataset.zip
