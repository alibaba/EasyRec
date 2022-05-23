# How to run SOK in EasyRec

First, `git checkout sok_integration`

1. Build docker

```bash
bash easy_rec/python/sok_adapter/experiments/docker/build_docker_easyrec_enhanced.sh
```

2. Run docker

```bash
PROJECT_ROOT=easyrec_path_on_host
CONTAINER_WORKDIR=/root/EasyRec
IMAGE=easyrec:latest

docker run \
    -it \
    --rm \
    --name easy_rec \
    --gpus="all" \
    -v ${PROJECT_ROOT}:${CONTAINER_WORKDIR} \
    --workdir ${CONTAINER_WORKDIR} \
    ${IMAGE} \
    bash
```

3. Install SOK inside

```bash
cd sparse_operation_kit && pip -v install .
```

4. Copy modified estimator files to system-installed estimator package

```bash
bash easy_rec/python/sok_adapter/edit/edit.sh
```

Currently we are using some hack method to make SOK run, some important SOK related settings are in `easy_rec/python/sok_adapter/edit/estimator.py`. You can modify accordingly. For example, you can comment out sparse or dense sok embedding, and use `dense_sok()` or `sparse_sok()` in `easy_rec/python/layers/input_layer.py`

5. setup envs

```bash
export PYTHONPATH="/root/EasyRec:${PYTHONPATH}"
```

6. run sok example

```bash
mkdir -p easy_rec/python/sok_adapter/experiments/dlrm_on_criteo/ckpt
rm -rf easy_rec/python/sok_adapter/experiments/dlrm_on_criteo/ckpt/*
python -m easy_rec.python.train_eval \
    --pipeline_config_path easy_rec/python/sok_adapter/experiments/dlrm_on_criteo/dlrm_on_criteo.config
```