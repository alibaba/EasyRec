#! /bin/bash
if [ "$(uname)" == "Darwin" ]; then
    curl -O http://files.grouplens.org/datasets/movielens/ml-1m.zip
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    wget -c http://files.grouplens.org/datasets/movielens/ml-1m.zip
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
    curl -O http://files.grouplens.org/datasets/movielens/ml-1m.zip
fi
unzip ml-1m.zip
mv ml-1m.zip ml-1m/
python process_ml_1m.py
