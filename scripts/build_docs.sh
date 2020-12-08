# Make sure in python3 environment
DATE=`date +'%Y%m%d'`

# make proto
bash scripts/gen_proto.sh
PATH=./protoc/bin/ protoc/bin/protoc --doc_out=html,proto.html:docs/source easy_rec/python/protos/*.proto
sed -i 's#<p>#<pre>#g;s#</p>#</pre>#g' docs/source/proto.html

# install requirements
python setup.py install
pip install -r requirements/docs.txt

# make sphinx
cd docs
rm -rf build
make html
rm -rf build/html/_modules
