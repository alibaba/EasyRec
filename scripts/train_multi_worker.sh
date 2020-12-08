# distributed train script, train on one machine with 2gpus
EXP="multi_worker"
LOG_DIR="logs/"

PIPELINE_CONFIG=$1
shift 1

if [ -z "$PIPELINE_CONFIG" ]
then
  echo "usage: sh $0 pipeline_config_path ... "
  exit 1
fi

if [ ! -e $PIPELINE_CONFIG ]
then
  echo "config file: $PIPELINE_CONFIG does not exits"
  exit 1
fi

if [ ! -e $LOG_DIR ]
then
  mkdir $LOG_DIR
fi

WAIT_DONE=$1
if [ "$WAIT_DONE" == "WAIT_DONE" ]
then
  echo "wait for process done"
  shift 1
fi

worker_hosts="localhost:2224,localhost:2223"
gpus="4,5"

echo "worker_hosts=${worker_hosts}"
echo "gpus=${gpus}"

worker_hosts=`echo $worker_hosts | awk -v FS=',' '{ a=""; for(i = 1; i <= NF; i++) { if (a!="") a = a","; a = a"\""$i"\"" } print a }'`

worker_num=`echo $worker_hosts | awk -v FS="," '{ print NF }'`
for worker_id in `seq 0 $((worker_num-1))`
do
  # Worker Process
  export TF_CONFIG="{
                      \"cluster\":
                      {
                          \"worker\": [$worker_hosts]
                      },
                      \"task\":
                      {
                          \"type\": \"worker\",
                          \"index\": $worker_id
                      }
                 }"

  gpu_id=`echo $gpus | awk -v FS=',' -v worker_id=$((worker_id+1)) '{ print $worker_id }'`
  echo "start worker=$worker_id gpu_id=$gpu_id"
  CUDA_VISIBLE_DEVICES=$gpu_id nohup python -m easy_rec.python.train_eval \
          --pipeline_config_path $PIPELINE_CONFIG $@\
          > $LOG_DIR/log_${EXP}_worker_${worker_id}.txt 2>&1 &
  worker_pids[$worker_id]=$!
  echo "    pid=$!"
done

echo "logs:"
ls -lh ${LOG_DIR}/log_${EXP}_*.txt

if [ "$WAIT_DONE" == "WAIT_DONE" ]
then
  for worker_id in `seq 0 $((worker_num-1))`
  do
    wait ${worker_pids[$worker_id]}
    error_code=$?
    if [ $error_code -ne 0 ]
    then
      echo "worker $worker_id ${worker_pids[$worker_id]} failed[error_code=$error_code]"
      exit $error_code
    else
      echo "worker $worker_id done"
    fi
  done
fi
