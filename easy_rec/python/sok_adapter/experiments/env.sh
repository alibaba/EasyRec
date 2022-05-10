SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

EASYREC_PTYHON_DIR="${SCRIPT_DIR}/../"
export PYTHONPATH="${EASYREC_PTYHON_DIR}:${PYTHONPATH}"
export TF_PATH="/usr/local/lib/python3.8/dist-packages/tensorflow_core"