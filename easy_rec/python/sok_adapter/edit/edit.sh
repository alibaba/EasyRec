SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

sudo cp ${SCRIPT_DIR}/estimator.py /usr/local/lib/python3.8/dist-packages/tensorflow_estimator/python/estimator/estimator.py
sudo cp ${SCRIPT_DIR}/saver.py /usr/local/lib/python3.8/dist-packages/tensorflow_core/python/training/saver.py
sudo cp ${SCRIPT_DIR}/session_manager.py /usr/local/lib/python3.8/dist-packages/tensorflow_core/python/training/session_manager.py