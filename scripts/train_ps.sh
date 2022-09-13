# distributed train script, train on one machine with 2gpus
EXP="ps"
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

ps_hosts="localhost:2327"
chief_hosts="localhost:2323"
worker_hosts="localhost:2324"
gpus=""

echo "ps_hosts=${ps_hosts}"
echo "chief_hosts=${chief_hosts}"
echo "worker_hosts=${worker_hosts}"
echo "gpus=${gpus}"

# add quotes
ps_hosts=`echo $ps_hosts | awk -v FS=',' '{ a=""; for(i = 1; i <= NF; i++) { if (a!="") a = a","; a = a"\""$i"\"" } print a }'`
chief_hosts="\"${chief_hosts}\""
worker_hosts=`echo $worker_hosts | awk -v FS=',' '{ a=""; for(i = 1; i <= NF; i++) { if (a!="") a = a","; a = a"\""$i"\"" } print a }'`


cluster_str="\"cluster\":
              {
                  \"ps\": [$ps_hosts],
                  \"chief\": [$chief_hosts],
                  \"worker\": [$worker_hosts]
              }"

# Parameter Server Process
ps_num=`echo $ps_hosts | awk -v FS=',' '{ print NF }'`
for ps_id in `seq 0 $((ps_num-1))`
do
  export TF_CONFIG="{
                      $cluster_str,
                      \"task\":
                      {
                          \"type\": \"ps\",
                          \"index\": $ps_id
                      }
                 }"
  CUDA_VISIBLE_DEVICES='' nohup python -m easy_rec.python.train_eval \
          --pipeline_config_path $PIPELINE_CONFIG $@\
          > $LOG_DIR/log_${EXP}_ps_${ps_id}.txt &
  echo "ps[$ps_id] started pid=$!"
done


# Master Worker Process
export TF_CONFIG="{
                    $cluster_str,
                    \"task\":
                    {
                        \"type\": \"chief\",
                        \"index\": 0
                    }
               }"
gpu_id=`echo $gpus | awk -v FS=',' -v worker_id=1 '{ print $worker_id }'`
CUDA_VISIBLE_DEVICES=$gpu_id nohup python -m easy_rec.python.train_eval \
        --pipeline_config_path $PIPELINE_CONFIG $@\
        > $LOG_DIR/log_${EXP}_chief.txt 2>&1 &
chief_pid=$!
echo "chief started gpu_id=$gpu_id pid=$chief_pid"

worker_num=`echo $worker_hosts | awk -v FS="," '{ print NF }'`
for worker_id in `seq 0 $((worker_num-1))`
do
  gpu_id=`echo $gpus | awk -v FS=',' -v worker_id=$((worker_id+2)) '{ print $worker_id }'`
  # Master Worker Process
  export TF_CONFIG="{
                      $cluster_str,
                      \"task\":
                      {
                          \"type\": \"worker\",
                          \"index\": $worker_id
                      }
                 }"
  CUDA_VISIBLE_DEVICES=$gpu_id nohup python -m easy_rec.python.train_eval \
          --pipeline_config_path $PIPELINE_CONFIG $@\
          > $LOG_DIR/log_${EXP}_worker_${worker_id}.txt  2>&1 &
  worker_pids[$worker_id]=$!
  echo "worker[$worker_id] gpu_id=$gpu_id pid=$! started"
done

echo "logs:"
ls -lh ${LOG_DIR}/log_${EXP}_*.txt

if [ "$WAIT_DONE" == "WAIT_DONE" ]
then
  wait $chief_pid
  error_code=$?
  if [ $error_code -ne 0 ]
  then
    echo "wait chief $chief_pid failed[error_code=$error_code]"
    exit 1
  fi

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
