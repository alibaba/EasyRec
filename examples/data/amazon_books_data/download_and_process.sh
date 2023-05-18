wget -c https://ali-rec-sln.oss-cn-hangzhou.aliyuncs.com/resources/AmazonBooksData.tar.gz
tar -zxvf AmazonBooksData.tar.gz
mv AmazonBooksData.tar.gz AmazonBooksData/
python process_amazon.py
