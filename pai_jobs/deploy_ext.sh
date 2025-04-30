greadlink=`which greadlink 2>/dev/null`
if [ -n "$greadlink" ]  && [ -e "$greadlink" ]
then
  curr_path=`greadlink -f $0`
else
  curr_path=`readlink -f $0`
  if [ -z "$curr_path" ]
  then
    echo "for mac: brew install coreutils"
    exit 1
  fi
fi
curr_dir=`dirname $curr_path`
root_dir=`dirname $curr_dir`

VERSION=`grep -o "[0-9]\.[0-9]\.[0-9]" easy_rec/version.py`

ODPSCMD=odpscmd
# 0: deploy resources and xflow
# 1: deploy resources only
# 2: generate resources and xflow only, not deploy
mode=0
odps_config=""

is_tf15=0
is_py3=0
while getopts 'V:C:OGc:D:P' OPT; do
    case $OPT in
        V)
            VERSION="$OPTARG";;
        C)
            ODPSCMD="$OPTARG";;
        c)
            odps_config="$OPTARG";;
        O)
            mode=1;;
        G)
            mode=2;;
        D)
            is_tf15=1;;
        P)
            is_py3=1;;
        ?)
            echo "Usage: `basename $0` -V VERSION [-C odpscmd_path] [-c odps_config_path] [-O]"
            echo " -O: only update easy_rec resource file"
            echo " -G: generate resource file and xflow, but not deploy"
            echo " -c: odps_config file path"
            echo " -C: odpscmd file path, default to: odpscmd, so in default odpscmd must be in PATH"
            echo " -D: use tf1.15 or deeprec"
            echo " -P: use tf1.12_py3"
            echo " -V: algorithm version, chars must be in [0-9A-Za-z_-], default: version info in easy_rec/version.py"
            exit 1
    esac
done

if [ -z "$VERSION" ]
then
  echo "algorithm version(-V) is not set."
  exit 1
fi

ODPSCMD=`which $ODPSCMD`
if [ $? -ne 0 ] && [ $mode -ne 2 ]
then
   echo "$ODPSCMD is not in PATH"
   exit 1
fi

if [ ! -e "$odps_config" ] && [ $mode -ne 2 ]
then
  if [ -z "$odps_config" ]
  then
      echo "odps_config is not set"
  else
      echo "odps_config[$odps_config] does not exist"
  fi
  exit 1
fi
if [ -e "$odps_config" ]
then
  odps_config=`readlink -f $odps_config`
fi

cd $root_dir
sh scripts/gen_proto.sh
if [ $? -ne 0 ]
then
  echo "generate proto file failed"
  exit 1
fi

cd $curr_dir

RES_PATH=easy_rec_ext_${VERSION}_res.tar.gz

if [ -e easy_rec ]
then
  rm -rf easy_rec
fi
cp -R $root_dir/easy_rec ./easy_rec
sed -i -e "s/\[VERSION\]/$VERSION/g" easy_rec/__init__.py
find -L easy_rec -name "*.pyc" | xargs rm -rf

if [ ! -d "datahub" ]
then
  if [ ! -e "pydatahub.tar.gz" ]
  then
    wget http://easyrec.oss-cn-beijing.aliyuncs.com/third_party/pydatahub.tar.gz
    if [ $? -ne 0 ]
    then
      echo "datahub download failed."
    fi
  fi
  tar -zvxf pydatahub.tar.gz
  rm -rf pydatahub.tar.gz
fi

if [ ! -d "kafka" ]
then
  if [ ! -e "kafka.tar.gz" ]
  then
    wget http://easyrec.oss-cn-beijing.aliyuncs.com/third_party/kafka.tar.gz
    if [ $? -ne 0 ]
    then
      echo "kafka download failed."
    fi
  fi
  tar -zvxf kafka.tar.gz
  rm -rf kafka.tar.gz
fi

if [ ! -d "faiss" ]
then
  if [ ! -e "faiss.tar.gz" ]
  then
    wget http://easyrec.oss-cn-beijing.aliyuncs.com/third_party/faiss.tar.gz
    if [ $? -ne 0 ]
    then
      echo "faiss download failed."
    fi
  fi
  tar -zvxf faiss.tar.gz
  rm -rf faiss.tar.gz
fi

if [ -d "tensorflow_probability" ]
then
  rm -rf tensorflow_probability
fi
if [ $is_tf15 -gt 0 ]; then
  tfp_version='0.8.0'
else
  tfp_version='0.5.0'
fi
if [ ! -e "tensorflow_probability" ]
then
  wget http://easyrec.oss-cn-beijing.aliyuncs.com/3rdparty/probability-${tfp_version}.tar.gz
  if [ $? -ne 0 ]
  then
    echo "tensorflow_probability download failed."
  fi
fi
tar -xzvf probability-${tfp_version}.tar.gz --strip-components=1 probability-${tfp_version}/tensorflow_probability
rm -rf tensorflow_probability/examples
rm -rf tensorflow_probability/g3doc
rm -rf probability-${tfp_version}.tar.gz

tar -cvzhf $RES_PATH easy_rec datahub lz4 cprotobuf kafka faiss tensorflow_probability run.py

# 2 means generate only
if [ $mode -ne 2 ]
then
  ${ODPSCMD} --config=$odps_config -e "add archive $RES_PATH -f;"
  if [ $? -ne 0 ]
  then
    echo "add $RES_PATH failed"
    exit 1
  fi
fi

# deploy resource only
if [ $mode -eq 1 ]
then
  echo "add $RES_PATH succeed, version=${VERSION}"
  echo "[WARNING] will not update xflow"
  echo "   your must specify -Dversion=${VERSION} when run pai -name easy_rec_ext"
  exit 0
fi

cd easy_rec_flow_ex
sed -i -e "s/parameter name=\"version\" use=\"optional\" default=\"[0-9A-Za-z_-]\+\"/parameter name=\"version\" use=\"optional\" default=\"$VERSION\"/g" easy_rec_ext.xml

if [ $is_tf15 -gt 0 ]
then
  echo "will deploy DeepRec(TF1.15) version"
  sed -i -e "s/name=\"easy_rec_ext\"/name=\"easy_rec_ext15\"/g" easy_rec_ext.xml
  sed -i -e "s/tensorflow1120_ext/tensorflow1150_ext/g" easy_rec_ext.xml
fi

if [ $is_py3 -gt 0 ]
then
  echo "will deploy TF1.12_py3 version"
  sed -i -e "s/name=\"easy_rec_ext\"/name=\"easy_rec_py3_ext\"/g" easy_rec_ext.xml
  sed -i -e "s/tensorflow1120_ext/tensorflow1120_py3_ext/g" easy_rec_ext.xml
fi

tar -cvzf easy_rec_flow_ex.tar.gz easy_rec_ext.lua  easy_rec_ext.xml

git checkout ./easy_rec_ext.xml

if [ $mode -ne 2 ]
then
  cd ../xflow-deploy
  package=../easy_rec_flow_ex/easy_rec_flow_ex.tar.gz
  python xflow_deploy.py conf=${odps_config} package=$package
  if [ $? -ne 0 ]
  then
     echo "deploy $package failed"
     exit 1
  else
     echo "deploy $package succeed"
  fi
  rm -rf ../easy_rec_flow_ex/easy_rec_flow_ex ../easy_rec_flow_ex/easy_rec_flow_ex.tar.gz
else
  cd $curr_dir
  echo "Generated RESOURCES: $RES_PATH"
  echo "Generated XFLOW: easy_rec_flow_ex/easy_rec_flow_ex.tar.gz"
fi
