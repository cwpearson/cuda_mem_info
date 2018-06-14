# /bin/bash

export CUDA_HOST=${HOME}/cuda
export CUDA_URL="https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda_9.2.88_396.26_linux"

curl -sL -o cuda.run "$CUDA_URL" \
    && chmod +x cuda.run \
    && ./cuda.run --toolkit --silent --toolkitpath="$CUDA_HOST"

