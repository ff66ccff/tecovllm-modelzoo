#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=$(cd -- "${MODEL_DIR}/../.." && pwd)

# The SDK setup references optional unset variables.
set +u
source /opt/tecoai/setvars.sh
set -u

# Keep the runner and evalscope CLI on the vendor interpreter.
export PYTHON=/home/py312/bin/python
export PATH="/home/py312/bin:${PATH}"
if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: required vendor interpreter ${PYTHON} is unavailable" >&2
    exit 1
fi

# The CI/evalscope client needs only tokenizer metadata normalization. The
# service run.sh prepends its own overlay/runtime and binds the model itself.
unset GEMMA4_OVERLAY
export GEMMA4_EVAL_TOKENIZER=1
export PYTHONPATH="${MODEL_DIR}/eval_overlay${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec "${PYTHON}" "${REPO_ROOT}/tools/ci_pipline/run_ci.py" \
    "${MODEL_DIR}/run.sh" "$@"
