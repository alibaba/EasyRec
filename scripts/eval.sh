export TF_CONFIG="{\"cluster\": {\"chief\":[\"localhost:2221\"],\"evaluator\":[\"localhost:2220\"]}, \"task\": {\"type\": \"evaluator\", \"index\": 0}}"

CUDA_VISIBLE_DEVICES=0 python -m easy_rec.python.eval --pipeline_config_path $@
