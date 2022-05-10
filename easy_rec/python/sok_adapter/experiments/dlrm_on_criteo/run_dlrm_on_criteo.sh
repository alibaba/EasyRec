rm -rf experiments/dlrm_on_criteo/ckpt/*
python -m easy_rec.python.train_eval \
    --pipeline_config_path experiments/dlrm_on_criteo/dlrm_on_criteo.config