# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 DimensionLab s.r.o.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_CONTAINER=nvcr.io/nvidia/modulus/modulus:24.01
FROM $BASE_CONTAINER as builder

FROM mltooling/ml-workspace:latest as modulus-mlhub

ARG ARG_WORKSPACE_FLAVOR="gpu"
ENV WORKSPACE_FLAVOR=$ARG_WORKSPACE_FLAVOR

USER root

# Adding this for hosts with libnvidia-container <1.3 
ENV NVIDIA_DISABLE_REQUIRE=true

ENV NVARCH x86_64
ENV NV_CUDA_CUDART_VERSION 12.3.101-1
ENV NV_CUDA_COMPAT_PACKAGE cuda-compat-12-3

ENV NVIDIA_REQUIRE_CUDA "CUDA>=12.3"
ENV NV_CUDA_CUDART_VERSION 12.3.101-1


### NVIDIA CUDA BASE ###
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.3.2/ubuntu2004/base/Dockerfile
RUN \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH}/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    # Cleanup - cannot use cleanup script here, otherwise too much is removed
    apt-get clean && \
    rm -rf $HOME/.cache/* && \
    rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 12.3.2

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-12-3=${NV_CUDA_CUDART_VERSION} \
    ${NV_CUDA_COMPAT_PACKAGE} \
    && ln -s cuda-11.3 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/* && \
    # Cleanup - cannot use cleanup script here, otherwise too much is removed
    apt-get clean && \
    rm -rf $HOME/.cache/* && \
    rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# nvidia-container-runtime
# https://github.com/NVIDIA/nvidia-container-runtime#environment-variables-oci-spec
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
# Some cloud servers fail to start this container when NVIDIA_REQUIRE_CUDA has too high CUDA version
# ENV NVIDIA_REQUIRE_CUDA "cuda>=11.4 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=450,driver<451"


# ### CUDA RUNTIME ###
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.3.2/ubuntu2004/runtime/Dockerfile

ENV NV_CUDA_LIB_VERSION 12.3.2-1

ENV NV_NVTX_VERSION 12.3.101-1
ENV NV_LIBNPP_VERSION 12.2.3.2-1
ENV NV_LIBNPP_PACKAGE libnpp-12-3=${NV_LIBNPP_VERSION}
ENV NV_LIBCUSPARSE_VERSION 12.2.0.103-1

ENV NV_LIBCUBLAS_PACKAGE_NAME libcublas-12-3
ENV NV_LIBCUBLAS_VERSION 12.3.4.1-1
ENV NV_LIBCUBLAS_PACKAGE ${NV_LIBCUBLAS_PACKAGE_NAME}=${NV_LIBCUBLAS_VERSION}

ENV NV_LIBNCCL_PACKAGE_NAME libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION 2.20.3-1
ENV NCCL_VERSION 2.20.3-1
ENV NV_LIBNCCL_PACKAGE ${NV_LIBNCCL_PACKAGE_NAME}=${NV_LIBNCCL_PACKAGE_VERSION}+cuda12.3


RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-12-3=${NV_CUDA_LIB_VERSION} \
    ${NV_LIBNPP_PACKAGE} \
    cuda-nvtx-12-3=${NV_NVTX_VERSION} \
    libcusparse-12-3=${NV_LIBCUSPARSE_VERSION} \
    ${NV_LIBCUBLAS_PACKAGE} \
    ${NV_LIBNCCL_PACKAGE} \
    && rm -rf /var/lib/apt/lists/* \
    # Cleanup - cannot use cleanup script here, otherwise too much is removed
    && apt-get clean \
    && rm -rf $HOME/.cache/* \
    && rm -rf /tmp/* \
    && rm -rf /var/lib/apt/lists/*

RUN apt-mark hold ${NV_LIBCUBLAS_PACKAGE_NAME} ${NV_LIBNCCL_PACKAGE_NAME}

### END CUDA RUNTIME ###

### CUDA DEVEL ###
# https://gitlab.com/nvidia/container-images/cuda/-/tree/master/dist/12.3.2/ubuntu2004/devel

ENV NV_CUDA_CUDART_DEV_VERSION 12.3.101-1
ENV NV_NVML_DEV_VERSION 12.3.101-1
ENV NV_LIBCUSPARSE_DEV_VERSION 12.2.0.103-1
ENV NV_LIBNPP_DEV_VERSION 12.2.3.2-1
ENV NV_LIBNPP_DEV_PACKAGE libnpp-dev-12-3=${NV_LIBNPP_DEV_VERSION}

ENV NV_LIBCUBLAS_DEV_VERSION 12.3.4.1-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME libcublas-dev-12-3
ENV NV_LIBCUBLAS_DEV_PACKAGE ${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}

ENV NV_CUDA_NSIGHT_COMPUTE_VERSION 12.3.2-1
ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE cuda-nsight-compute-12-3=${NV_CUDA_NSIGHT_COMPUTE_VERSION}

ENV NV_NVPROF_VERSION 12.3.101-1
ENV NV_NVPROF_DEV_PACKAGE cuda-nvprof-12-3=${NV_NVPROF_VERSION}

ENV NV_LIBNCCL_DEV_PACKAGE_NAME libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION 2.20.3-1
ENV NCCL_VERSION 2.20.3-1
ENV NV_LIBNCCL_DEV_PACKAGE ${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda12.3


RUN apt-get update && apt-get install -y --no-install-recommends \
    libtinfo5 libncursesw5 \
    cuda-cudart-dev-12-3=${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-command-line-tools-12-3=${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-12-3=${NV_CUDA_LIB_VERSION} \
    cuda-libraries-dev-12-3=${NV_CUDA_LIB_VERSION} \
    cuda-nvml-dev-12-3=${NV_NVML_DEV_VERSION} \
    ${NV_NVPROF_DEV_PACKAGE} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    libcusparse-dev-12-3=${NV_LIBCUSPARSE_DEV_VERSION} \
    ${NV_LIBCUBLAS_DEV_PACKAGE} \
    ${NV_LIBNCCL_DEV_PACKAGE} \
    ${NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE} \
    # Cleanup - cannot use cleanup script here, otherwise too much is removed
    && apt-get clean && \
    rm -rf $HOME/.cache/* && \
    rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/*

# # Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME}
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# ### END CUDA DEVEL ###

# ### CUDANN9 DEVEL ###
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.3.2/ubuntu2004/devel/cudnn9/Dockerfile

ENV NV_CUDNN_VERSION 9.0.0.312
ENV NV_CUDNN_PACKAGE_NAME "libcudnn9-cuda-12"

ENV NV_CUDNN_PACKAGE "libcudnn9-cuda-12=${NV_CUDNN_VERSION}-1"
ENV NV_CUDNN_PACKAGE_DEV "libcudnn9-dev-cuda-12=${NV_CUDNN_VERSION}-1"


RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    # Cleanup
    && apt-get clean && \
    rm -rf /root/.cache/* && \
    rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/*

### END CUDANN9 ###

### GUI TOOLS ###

# Install Paraview
COPY ml-workspace/resources/tools/paraview.sh $RESOURCES_PATH/tools/paraview.sh
RUN \
    /bin/bash $RESOURCES_PATH/tools/paraview.sh --install && \
    # Cleanup
    clean-layer.sh

### END GUI TOOLS ###

# ### NVIDIA MODULUS ###

# Update pip and setuptools
RUN pip install "pip==23.2.1" "setuptools==68.2.2"  

# Setup git lfs, graphviz gl1(vtk dep)
RUN apt-get update && \
    apt-get install -y git-lfs graphviz libgl1 && \
    git lfs install && \
    # install vtk
    echo "Installing vtk for: linux/amd64" && \
	pip install --no-cache-dir "vtk>=9.2.6" && \ 
    # Install modulus sym dependencies
    pip install --no-cache-dir "hydra-core>=1.2.0" "termcolor>=2.1.1" "chaospy>=4.3.7" "Cython==0.29.28" "numpy-stl==2.16.3" "opencv-python==4.5.5.64" \
    "scikit-learn==1.0.2" "symengine>=0.10.0" "sympy==1.12" "timm==0.5.4" "torch-optimizer==0.3.0" "transforms3d==0.3.1" \
    "typing==3.7.4.3" "pillow==10.0.1" "notebook==6.4.12" "mistune==2.0.3" "pint==0.19.2" "tensorboard>=2.8.0" && \
    # Cleanup
    clean-layer.sh

# Install tiny-cuda-nn
ENV TCNN_CUDA_ARCHITECTURES="60;70;75;80;86;90"

# Build and install pysdf
COPY --from=builder /external /external
COPY libpysdf.so /external/lib/libpysdf.so
COPY libsdf.so /external/lib/libsdf.so
# COPY --from=builder /usr/local/lib/python3.10/dist-packages/pysdf /opt/conda/lib/python3.8/site-packages/pysdf
# COPY --from=builder /usr/local/lib/python3.10/dist-packages/pysdf-0.1.dist-info /opt/conda/lib/python3.8/site-packages/pysdf-0.1.dist-info
# Tinycudann
COPY --from=builder /usr/local/lib/python3.10/dist-packages/tinycudann /opt/conda/lib/python3.8/site-packages/tinycudann
COPY --from=builder /usr/local/lib/python3.10/dist-packages/tinycudann-1.7.dist-info /opt/conda/lib/python3.8/site-packages/tinycudann-1.7.dist-info
COPY --from=builder /usr/local/lib/python3.10/dist-packages/tinycudann_bindings /opt/conda/lib/python3.8/site-packages/tinycudann_bindings

# Install Modulus
RUN pip install --upgrade --no-cache-dir git+https://github.com/NVIDIA/modulus-sym.git && \
    # Cleanup
    clean-layer.sh

ENV LD_LIBRARY_PATH=/external/lib:${LD_LIBRARY_PATH}:/usr/local/cuda/extras/CUPTI/lib64 \
    NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility,video \
    _CUDA_COMPAT_TIMEOUT=90

### END NVIDIA MODULUS ###

# install jupyterlab extensions for HW / GPU monitoring
RUN apt-get update && \
    pip install jupyterlab_nvdashboard && \
    # Cleanup
    clean-layer.sh

ENV PYTHONPATH ${RESOURCES_PATH}/paraview_build/lib:$RESOURCES_PATH/paraview_build/lib/python3.8/site-packages
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:$RESOURCES_PATH/paraview_build/lib
ENV PATH ${PATH}:${RESOURCES_PATH}/paraview_build/bin

### Install python libs
COPY ml-workspace/resources/libraries/requirements-dimensionlab.txt $RESOURCES_PATH/libraries/requirements-dimensionlab.txt
RUN \
    apt-get update && \
    apt-get install -y libomp-dev libopenblas-base && \
    # upgrade pip
    pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade --upgrade-strategy only-if-needed scikit-learn==1.3.2 && \
    # Install minimal pip requirements
    pip install --no-cache-dir --upgrade --upgrade-strategy only-if-needed -r ${RESOURCES_PATH}/libraries/requirements-dimensionlab.txt && \
    # Cleanup
    clean-layer.sh

# Fix pysdf
COPY pysdf-0.1-py3.8-linux-x86_64.egg /external/eggs/pysdf-0.1-py3.8-linux-x86_64.egg
RUN pip install setuptools==42.0.0 && \
    cd /external/eggs/ && \
    python -m easy_install pysdf-0.1-py3.8-linux-x86_64.egg && \
    # back to previous version
    pip install setuptools==65.3.0 && \
    # Cleanup
    clean-layer.sh

# Install bison
# RUN apt-get update && \
#     cd $HOME && \
#     wget https://ftp.gnu.org/gnu/bison/bison-3.2.tar.gz && \
#     tar xf bison-3.2.tar.gz && \
#     cd bison-3.2 && \
#     ./configure --prefix=$HOME/bison-install && \
#     make && \
#     make install && \
#     # Cleanup
#     clean-layer.sh

# # Add bison installed path to PATH
# ENV PATH=$HOME/bison-install/bin:$PATH

# Fix glibc -> move to version 2.34
# RUN apt-get update && \
#     mkdir $HOME/glibc/ && cd $HOME/glibc && \
#     wget http://ftp.gnu.org/gnu/libc/glibc-2.34.tar.gz && \
#     tar -xvzf glibc-2.34.tar.gz && \
#     mkdir build && \
#     mkdir glibc-2.34-install && \
#     cd build && \
#     ~/glibc/glibc-2.34/configure --prefix=$HOME/glibc/glibc-2.34-install && \
#     make && \
#     make install && \
#     # Cleanup
#     clean-layer.sh

# ENV PATH=$HOME/glibc/build:$PATH

# RUN conda create -n paraview python==3.8 -y

# ENV PATH ${CONDA_ROOT}/envs/paraview/bin:$PATH

# RUN \
#     # /bin/bash -c "source activate paraview && \
#     conda create -y -n paraview python==3.8 && \
#     conda install -y -n paraview -c conda-forge \
#         ipykernel \
#         paraview=5.8 && \
#     conda run -n paraview ipython kernel install --user --name=paraview && \
#     git clone https://github.com/NVIDIA/ipyparaview.git && \
#     cd ipyparaview && \
#     conda run -n paraview pip install -e . && \
#     conda run -n paraview jupyter nbextension install --py --symlink --sys-prefix ipyparaview && \
#     conda run -n paraview jupyter nbextension enable --py --sys-prefix ipyparaview && \
#     conda run -n paraview jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
#     conda run -n paraview jupyter labextension install js && \
#     fix-permissions.sh $CONDA_ROOT && \
#     clean-layer.sh

RUN \
    conda install -y ipykernel paraview=5.8 -c conda-forge && \
    ipython kernel install --user --name=paraview && \
    git clone https://github.com/NVIDIA/ipyparaview.git && \
    cd ipyparaview && \
    pip install -e . && \
    jupyter nbextension install --py --symlink --sys-prefix ipyparaview && \
    jupyter nbextension enable --py --sys-prefix ipyparaview && \
    fix-permissions.sh $CONDA_ROOT && \
    clean-layer.sh


# Create Desktop Icons for Tooling
COPY ml-workspace/resources/branding $RESOURCES_PATH/branding

# Patch Modulus Sym vtk util
COPY patches/vtk.py /opt/conda/lib/python3.8/site-packages/modulus/sym/utils/io/vtk.py

# Branding of various components
RUN \
    # Jupyter Branding
    cp -f $RESOURCES_PATH/branding/logo.png $CONDA_PYTHON_DIR"/site-packages/notebook/static/base/images/logo.png" && \
    cp -f $RESOURCES_PATH/branding/favicon.ico $CONDA_PYTHON_DIR"/site-packages/notebook/static/base/images/favicon.ico" && \
    cp -f $RESOURCES_PATH/branding/favicon.ico $CONDA_PYTHON_DIR"/site-packages/notebook/static/favicon.ico" && \
    # Fielbrowser Branding
    mkdir -p $RESOURCES_PATH"/filebrowser/img/icons/" && \
    cp -f $RESOURCES_PATH/branding/favicon.ico $RESOURCES_PATH"/filebrowser/img/icons/favicon.ico" && \
    cp -f $RESOURCES_PATH/branding/favicon.ico $RESOURCES_PATH"/filebrowser/img/icons/favicon-32x32.png" && \
    cp -f $RESOURCES_PATH/branding/favicon.ico $RESOURCES_PATH"/filebrowser/img/icons/favicon-16x16.png" && \
    cp -f $RESOURCES_PATH/branding/ml-workspace-logo.svg $RESOURCES_PATH"/filebrowser/img/logo.svg"

ENV TF_FORCE_GPU_ALLOW_GROWTH true

# Overwrites env var from ml-workspace
ENV INCLUDE_TUTORIALS=false

### END CONFIGURATION ###
ARG ARG_WORKSPACE_VERSION

# Overwrite & add Labels
LABEL \
    "name"="Modulus Symbolic for ML Hub" \
    "maintainer"="Michal Takac <hello@dimensionlab.org>" \
    "description"="GPU-based scientific deep learning pipeline for training neural physics simulators" \
    "vendor"="DimensionLab" \
    "url"="https://github.com/DimensionLab/modulus-mlhub" \
    "workspace.version"=$ARG_WORKSPACE_VERSION