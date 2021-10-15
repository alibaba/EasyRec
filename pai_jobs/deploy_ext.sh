#/bin/bash

curr_path=`greadlink -f $0`
curr_dir=`dirname $curr_path`
root_dir=`dirname $curr_dir`
build_dir="$root_dir/build"

VERSION=""
ODPSCMD=odpscmd
resource_only=0
odps_config=""

while getopts 'V:C:Oc:' OPT; do
    case $OPT in
        V)
            VERSION="$OPTARG";;
        C)
            ODPSCMD="$OPTARG";;
        c)
            odps_config="$OPTARG";;
        O)
            resource_only=1;;
        ?)
            echo "Usage: `basename $0` -V VERSION [-C odpscmd_path] [-c odps_config_path] [-O]"
            echo " -O: only update easy_rec resource file"
            echo " -c: odps_config file path"
            echo " -C: odpscmd file path, default to: odpscmd, so in default odpscmd must be in PATH"
            echo " -V: algorithm version, chars must be in [0-9A-Za-z_-]"
            exit 1
    esac
done

if [ -z "$VERSION" ]
then
  echo "algorithm version(-V) is not set."
  exit 1
fi

ODPSCMD=`which $ODPSCMD`
if [ $? -ne 0 ]
then
   echo "$ODPSCMD is not in PATH"
   exit 1
fi

if [ -z $odps_config ]
then
  odpscmd_path=`greadlink -f $ODPSCMD`
  odpscmd_dir=`dirname $odpscmd_path`
  odpscmd_dir=`dirname $odpscmd_dir`
  odps_config=$odpscmd_dir/conf/odps_config.ini
fi
if [ ! -e $odps_config ]
then
  echo "$odps_config does not exist"
  exit 1
fi
odps_config=`greadlink -f $odps_config`

cd $root_dir
bash scripts/gen_proto.sh
if [ $? -ne 0 ]
then
  echo "generate proto file failed"
  exit 1
fi

if [ ! -d $build_dir ]
then
  mkdir $build_dir
fi
cd $build_dir
if [ $? -ne 0 ]
then
  echo "cannot get to $build_dir"
  exit 1
fi

RES_PATH=easy_rec_ext_${VERSION}_res.tar.gz
rm -rf ./easy_rec
cp -R $root_dir/easy_rec ./easy_rec
cp easy_rec/__init__.py easy_rec/__init__.py.bak
sed -i -e "s/\[VERSION\]/$VERSION/g" easy_rec/__init__.py
find -L easy_rec -name "*.pyc" | xargs rm -rf
cp ../requirements.txt ./requirements.txt
cp $curr_dir/run.py ./run.py
tar -cvzhf $RES_PATH easy_rec run.py requirements.txt
mv easy_rec/__init__.py.bak easy_rec/__init__.py
${ODPSCMD} --config=$odps_config -e "add file $RES_PATH -f;"
if [ $? -ne 0 ]
then
  echo "add $RES_PATH failed"
  exit 1
fi
if [ $resource_only -gt 0 ]
then
  echo "add $RES_PATH succeed, version=${VERSION}"
  echo "[WARNING] will not update xflow"
  echo "   your must specify -Dversion=${VERSION} when run pai -name easy_rec_ext"
  exit 0
fi
#rm -rf $RES_PATH

# cd easy_rec_flow_ex
# sed -i -e "s/parameter name=\"version\" use=\"optional\" default=\"[0-9A-Za-z_-]\+\"/parameter name=\"version\" use=\"optional\" default=\"$VERSION\"/g" easy_rec_ext.xml
# tar -cvzf easy_rec_flow_ex.tar.gz easy_rec_ext.lua  easy_rec_ext.xml
# cd ../xflow-deploy
# package=../easy_rec_flow_ex/easy_rec_flow_ex.tar.gz
# python xflow_deploy.py conf=${odps_config} package=$package
# if [ $? -ne 0 ]
# then
#    echo "deploy $package failed"
#    exit 1
# else
#    echo "deploy $package succeed"
# fi
# rm -rf ../easy_rec_flow_ex/easy_rec_flow_ex ../easy_rec_flow_ex/easy_rec_flow_ex.tar.gz
