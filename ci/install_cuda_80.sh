# /bin/bash

export CUDA_URL="https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run"

curl -sL -o cuda.run "$CUDA_URL" \
    && chmod +x cuda.run \
    && ./cuda.run --toolkit --silent --toolkitpath="$CUDA_HOST"

