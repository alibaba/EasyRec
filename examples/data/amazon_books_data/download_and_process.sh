#! /bin/bash
if [ "$(uname)" == "Darwin" ]; then
    curl -O https://ali-rec-sln.oss-cn-hangzhou.aliyuncs.com/resources/AmazonBooksData.tar.gz
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    wget -c https://ali-rec-sln.oss-cn-hangzhou.aliyuncs.com/resources/AmazonBooksData.tar.gz
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
    curl -O https://ali-rec-sln.oss-cn-hangzhou.aliyuncs.com/resources/AmazonBooksData.tar.gz
fi
tar -zxvf AmazonBooksData.tar.gz
mv AmazonBooksData.tar.gz AmazonBooksData/
python process_amazon.py
