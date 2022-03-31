# Make sure in python3 environment

# remove old protos
find easy_rec/python/protos/ -name \"*_pb2.py\" | xargs rm -rf
# make proto
bash scripts/gen_proto.sh
PATH=./protoc/bin/ protoc/bin/protoc --doc_out=html,proto.html:docs/source easy_rec/python/protos/*.proto
sed -i 's#<p>#<pre>#g;s#</p>#</pre>#g' docs/source/proto.html

pip3 install tensorflow==2.3
pip install tensorflow==2.3
