!/usr/bin/env bash

git clone https://github.com/deepmind/deepmind-research.git
pip install -r deepmind-research/wikigraphs/requirements.txt
pip install -e deepmind-research/wikigraphs/
BASE_DIR=$1/wikigraphs/
mkdir ${BASE_DIR}

# wikitext-103
TARGET_DIR=${BASE_DIR}/wikitext-103
mkdir -p ${TARGET_DIR}
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -P ${TARGET_DIR}
unzip ${TARGET_DIR}/wikitext-103-v1.zip -d ${TARGET_DIR}
mv ${TARGET_DIR}/wikitext-103/* ${TARGET_DIR}
rm -rf ${TARGET_DIR}/wikitext-103 ${TARGET_DIR}/wikitext-103-v1.zip

# wikitext-103-raw
TARGET_DIR=${BASE_DIR}/wikitext-103-raw
mkdir -p ${TARGET_DIR}
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip -P ${TARGET_DIR}
unzip ${TARGET_DIR}/wikitext-103-raw-v1.zip -d ${TARGET_DIR}
mv ${TARGET_DIR}/wikitext-103-raw/* ${TARGET_DIR}
rm -rf ${TARGET_DIR}/wikitext-103-raw ${TARGET_DIR}/wikitext-103-raw-v1.zip


# processed freebase graphs
FREEBASE_TARGET_DIR=$1/wikigraphs/freebase/
mkdir -p ${FREEBASE_TARGET_DIR}/packaged/
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uuSS2o72dUCJrcLff6NBiLJuTgSU-uRo' -O ${FREEBASE_TARGET_DIR}/packaged/max256.tar
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nOfUq3RUoPEWNZa2QHXl2q-1gA5F6kYh' -O ${FREEBASE_TARGET_DIR}/packaged/max512.tar
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uuJwkocJXG1UcQ-RCH3JU96VsDvi7UD2' -O ${FREEBASE_TARGET_DIR}/packaged/max1024.tar

for version in max1024 max512 max256
do
  output_dir=${FREEBASE_TARGET_DIR}/freebase/${version}/
  mkdir -p ${output_dir}
  tar -xvf ${FREEBASE_TARGET_DIR}/packaged/${version}.tar -C ${output_dir}
done
rm -rf ${FREEBASE_TARGET_DIR}/packaged
python freebase_preprocess.py --freebase_dir=${FREEBASE_TARGET_DIR}/freebase/max1024 --output_dir=${BASE_DIR}/max1024 --text_dir=${BASE_DIR}/wikitext-103

python process_wikigraphs_data.py --dataset-path $1/wikigraphs/

rm -r $1/wikigraphs/wikitext-103/
rm -r $1/wikigraphs/wikitext-103-raw/
rm -r $1/wikigraphs/freebase/
rm -r deepmind-research
rm -r $1/wikigraphs/max1024/
mv $1/wikigraphs/valid.json $1/wikigraphs/val.json
