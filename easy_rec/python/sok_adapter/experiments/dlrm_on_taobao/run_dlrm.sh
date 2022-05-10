rm -rf experiments/dlrm/taobao_ckpt/*
python -m easy_rec.python.train_eval \
    --pipeline_config_path experiments/dlrm/dlrm_on_taobao.config
