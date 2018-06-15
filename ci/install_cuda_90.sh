# /bin/bash

export CUDA_URL="https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run"

curl -sL -o cuda.run "$CUDA_URL" \
    && chmod +x cuda.run \
    && ./cuda.run --toolkit --silent --toolkitpath="$CUDA_HOST"

