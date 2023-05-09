#! /bin/bash

wget -c https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz
mkdir criteo_kaggle_display
tar -zxvf criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz -C criteo_kaggle_display
mv criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz criteo_kaggle_display

python process_criteo_kaggle.py
