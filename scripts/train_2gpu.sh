# distributed train script, train on one machine with 2gpus
# local we do not support chief, worker, evaluator mode

EXP="2gpu"
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

# Parameter Server Process
export TF_CONFIG="{
                    \"cluster\":
                    {
                        \"ps\": [\"localhost:2227\"],
                        \"master\": [\"localhost:2223\"],
                        \"worker\": [\"localhost:2224\"]
                    },
                    \"task\":
                    {
                        \"type\": \"ps\",
                        \"index\": 0
                    }
               }"
CUDA_VISIBLE_DEVICES='' nohup python -m easy_rec.python.train_eval \
        --pipeline_config_path $PIPELINE_CONFIG $@\
        > $LOG_DIR/log_${EXP}_ps.txt &


# Master Worker Process
export TF_CONFIG="{
                    \"cluster\":
                    {
                        \"ps\": [\"localhost:2227\"],
                        \"master\": [\"localhost:2223\"],
                        \"worker\": [\"localhost:2224\"]
                    },
                    \"task\":
                    {
                        \"type\": \"master\",
                        \"index\": 0
                    }
               }"
CUDA_VISIBLE_DEVICES='' nohup python -m easy_rec.python.train_eval \
        --pipeline_config_path $PIPELINE_CONFIG $@\
        > $LOG_DIR/log_${EXP}_master.txt &

# Master Worker Process
export TF_CONFIG="{
                    \"cluster\":
                    {
                        \"ps\": [\"localhost:2227\"],
                        \"master\": [\"localhost:2223\"],
                        \"worker\": [\"localhost:2224\"]
                    },
                    \"task\":
                    {
                        \"type\": \"worker\",
                        \"index\": 0
                    }
               }"

CUDA_VISIBLE_DEVICES='' nohup python -m easy_rec.python.train_eval \
        --pipeline_config_path $PIPELINE_CONFIG $@\
        > $LOG_DIR/log_${EXP}_worker.txt &


echo "logdir: $LOG_DIR/log_${EXP}_ps.txt"
echo "logdir: $LOG_DIR/log_${EXP}_master.txt"
echo "logdir: $LOG_DIR/log_${EXP}_worker.txt"
