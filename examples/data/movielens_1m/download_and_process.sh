#! /bin/bash

wget -c http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
mv ml-1m.zip ml-1m/
python process_ml_1m.py
