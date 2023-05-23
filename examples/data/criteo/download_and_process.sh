#! /bin/bash
if [ "$(uname)" == "Darwin" ]; then
    curl -O https://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/criteo_kaggle/kaggle-display-advertising-challenge-dataset.tar.gz
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    wget -c https://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/criteo_kaggle/kaggle-display-advertising-challenge-dataset.tar.gz
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
    curl -O https://easy-rec.oss-cn-hangzhou.aliyuncs.com/data/criteo_kaggle/kaggle-display-advertising-challenge-dataset.tar.gz
fi
mkdir criteo_kaggle_display
tar -zxvf kaggle-display-advertising-challenge-dataset.tar.gz -C criteo_kaggle_display
mv kaggle-display-advertising-challenge-dataset.tar.gz criteo_kaggle_display
python process_criteo_kaggle.py
