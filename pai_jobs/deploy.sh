curr_path=`readlink -f $0`
curr_dir=`dirname $curr_path`
root_dir=`dirname $curr_dir`

cd $root_dir
sh scripts/gen_proto.sh
if [ $? -ne 0 ]
then
  echo "generate proto file failed"
  exit 1
fi

cd $curr_dir
rm -rf easy_rec
ln -s $root_dir/easy_rec ./
find -L easy_rec -name "*.pyc" | xargs rm -rf
tar -cvzhf easy_rec.tar.gz easy_rec run.py
