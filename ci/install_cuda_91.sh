# /bin/bash

export CUDA_URL="https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_387.26_linux"

curl -sL -o cuda.run "$CUDA_URL" \
    && chmod +x cuda.run \
    && ./cuda.run --toolkit --silent --toolkitpath="$CUDA_HOST"

