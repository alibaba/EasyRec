SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd ${SCRIPT_DIR}

docker build \
    -f ${SCRIPT_DIR}/dockerfiles/easyrec_enhanced.Dockerfile \
    --tag easyrec_enhanced:latest \
    ${SCRIPT_DIR}

