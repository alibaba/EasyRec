SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

#EASYREC_PTYHON_DIR="${SCRIPT_DIR}/../"
#
#export PYTHONPATH="${EASYREC_PTYHON_DIR}:${PYTHONPATH}"

# setup sok
cd ${SCRIPT_DIR}/../sparse_operation_kit
#python setup.py install
pip -v install .
cd ..
