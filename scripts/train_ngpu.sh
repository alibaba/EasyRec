LOG_DIR="logs/"

START_GPU=0
START_PORT=2001
WORKER_NUM=8
PS_NUM=2
HOST='localhost'

usage() {
  echo "Usage: `basename $0` -c criteo.config -s gpu_id -p start_port \
-m model_dir -f fine_tune_ckpt -W worker_num -P ps_num -E extra_args \
-H hostname -N exp_name"
}

args="--continue_train"

while getopts "c:s:p:m:f:P:W:E:H:N:" arg; do
  case $arg in
    c)
      PIPELINE_CONFIG=$OPTARG 
      ;;
    s)
      START_GPU=$OPTARG
      ;;
    p)
      START_PORT=$OPTARG
      ;;
    m)
      args="$args --model_dir $OPTARG"
      ;;
    f)
      args="$args --fine_tune_checkpoint $OPTARG"
      ;;
    W)
      WORKER_NUM=$OPTARG
      ;;
    P) 
      PS_NUM=$OPTARG
      ;;
    E)
      args="$args $OPTARG" 
      ;;
    H)
      HOST=$OPTARG
      ;;
    N)
      EXP=$OPTARG
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

shift $(($OPTIND - 1))

if [ -n "$@" ]
then 
  args="$args $@"
fi

if [ -z "$EXP" ]
then
  EXP="easy_rec"
fi

EXP=${EXP}_${START_GPU}_${START_PORT}

if [ -z "$PIPELINE_CONFIG" ]
then
  usage
  exit 1
fi

if [ ! -e $PIPELINE_CONFIG ]
then
  usage
  exit 1
fi

if [ ! -e $LOG_DIR ]
then
  mkdir $LOG_DIR
fi

echo "pipeline config: ${PIPELINE_CONFIG}"
echo "start gpu: ${START_GPU}"
echo "start port: ${START_PORT}"
echo "worker_num: ${WORKER_NUM}"
echo "ps_num: ${PS_NUM}"
echo "host: ${HOST}"
echo "more args: ${args}"
echo "exp: ${EXP}"


ps_hosts="\"$HOST:$START_PORT\""
for ps_id in `seq 1 $((PS_NUM-1))`
do
  ps_hosts=${ps_hosts}",\"$HOST:$((START_PORT+ps_id))\""
done

master_hosts=\"$HOST:$((START_PORT+PS_NUM))\"

worker_hosts="\"$HOST:$((START_PORT+PS_NUM+1))\""
for worker_id in `seq 1 $((WORKER_NUM-2))`
do
  worker_hosts=${worker_hosts}",\"$HOST:$((START_PORT+PS_NUM+worker_id+1))\""
done

echo "ps_hosts: $ps_hosts"
echo "master_hosts: $master_hosts"
echo "worker_hosts: $worker_hosts"

cluster_spec="{
                \"ps\": [$ps_hosts],
                \"master\": [$master_hosts],
                \"worker\": [$worker_hosts]
              }"

for ps_id in `seq 0 $((PS_NUM-1))`
do
  echo "start ps: ${ps_id}"
  log_file=$LOG_DIR/log_${EXP}_ps_${ps_id}.txt
  # Parameter Server Process
  export TF_CONFIG="{
                      \"cluster\":$cluster_spec,
                      \"task\":
                      {
                          \"type\": \"ps\",
                          \"index\": ${ps_id}
                      }
                 }"
  CUDA_VISIBLE_DEVICES='' nohup python -m easy_rec.python.train_eval \
          --pipeline_config_path $PIPELINE_CONFIG $args \
          > $log_file &
done


log_file_1=$LOG_DIR/log_${EXP}_master.txt
export TF_CONFIG="{
                    \"cluster\":$cluster_spec,
                    \"task\":
                    {
                        \"type\": \"master\",
                        \"index\": 0
                    }
               }"
CUDA_VISIBLE_DEVICES=$START_GPU nohup python -m easy_rec.python.train_eval \
        --pipeline_config_path $PIPELINE_CONFIG $args\
        > $log_file_1 &
echo $log_file_1

for worker_id in `seq 0 $((WORKER_NUM-2))`
do
  log_file_2=$LOG_DIR/log_${EXP}_worker_$((worker_id)).txt
  export TF_CONFIG="{
                      \"cluster\":$cluster_spec,
                      \"task\":
                      {
                          \"type\": \"worker\",
                          \"index\": $worker_id
                      }
                 }"
  ((START_GPU++))
  CUDA_VISIBLE_DEVICES=$START_GPU nohup python -m easy_rec.python.train_eval \
          --pipeline_config_path $PIPELINE_CONFIG $args\
          > $log_file_2 &
  echo $log_file_2
done
