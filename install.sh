#!/bin/bash

pip install -r requirements.txt
yes | conda install -c conda-forge cudatoolkit=11.8.0
yes | conda install cuda-nvcc=11.8 cuda-nvtx=11.8 cuda-libraries-dev=11.8 cuda-cupti=11.8 -c nvidia
yes | conda install nvidia::cuda-cudart-dev=11.8
ln -s ${CONDA_PREFIX}/lib ${CONDA_PREFIX}/lib64
mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

cd systems/ && pip install -e . && \
    cd watermarkers/networks/ptw && pip install -e . && \
    cd ../stable_sig && pip install -e . && \
    cd ../../../../modules/attack/unmark && pip install -e . \
    && cd ../../..
