#!/bin/bash

# Example usage: docker run -it -v$(pwd)/..:/workspace/TRTorch build_trtorch_wheel /bin/bash /workspace/TRTorch/py/build_whl.sh

export CXX=g++
export CUDA_HOME=/usr/local/cuda-12.1
export PROJECT_DIR=/workspace/project

cp -r $CUDA_HOME /usr/local/cuda

build_wheel() {
    $1/bin/python -m pip install --upgrade pip
    $1/bin/python -m pip wheel . --config-setting="--build-option=--release" --config-setting="--build-option=--ci" -w dist
}

patch_wheel() {
    $2/bin/python -m pip install auditwheel
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$1/torch/lib:$1/tensorrt/:${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs $2/bin/python -m auditwheel repair  $(cat ${PROJECT_DIR}/py/ci/soname_excludes.params) --plat manylinux_2_34_x86_64 dist/torch_tensorrt-*-$3-linux_x86_64.whl
}

py38() {
    cd /workspace/project
    PY_BUILD_CODE=cp38-cp38
    PY_VERSION=3.8
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    build_wheel ${PY_DIR}
    patch_wheel ${PY_PKG_DIR} ${PY_DIR} ${PY_BUILD_CODE}
}

py39() {
    cd /workspace/project
    PY_BUILD_CODE=cp39-cp39
    PY_VERSION=3.9
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    build_wheel ${PY_DIR}
    patch_wheel ${PY_PKG_DIR} ${PY_DIR} ${PY_BUILD_CODE}
}

py310() {
    cd /workspace/project
    PY_BUILD_CODE=cp310-cp310
    PY_VERSION=3.10
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    build_wheel ${PY_DIR}
    patch_wheel ${PY_PKG_DIR} ${PY_DIR} ${PY_BUILD_CODE}
}

py311() {
    cd /workspace/project
    PY_BUILD_CODE=cp311-cp311
    PY_VERSION=3.11
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    build_wheel ${PY_DIR}
    patch_wheel ${PY_PKG_DIR} ${PY_DIR} ${PY_BUILD_CODE}
}

py312() {
    cd /workspace/project
    PY_BUILD_CODE=cp312-cp312
    PY_VERSION=3.12
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    build_wheel ${PY_DIR}
    patch_wheel ${PY_PKG_DIR} ${PY_DIR} ${PY_BUILD_CODE}
}

libtorchtrt() {
    cd /workspace/project
    mkdir -p /workspace/project/py/wheelhouse
    PY_BUILD_CODE=cp310-cp310
    PY_VERSION=3.10
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    ${PY_DIR}/bin/python -m pip install --upgrade pip
    ${PY_DIR}/bin/python -m pip install -r py/requirements.txt
    ${PY_DIR}/bin/python -m pip install setuptools wheel auditwheel
    bazel build //:libtorchtrt --platforms //toolchains:ci_rhel_x86_64_linux -c opt --noshow_progress
    CUDA_VERSION=$(cd ${PROJECT_DIR}/py && ${PY_DIR}/bin/python3 -c "import versions; versions.cuda_version()")
    TORCHTRT_VERSION=$(cd ${PROJECT_DIR}/py && ${PY_DIR}/bin/python3 -c "import versions; versions.torch_tensorrt_version()")
    TRT_VERSION=$(cd ${PROJECT_DIR}/py && ${PY_DIR}/bin/python3 -c "import versions; versions.tensorrt_version()")
    CUDNN_VERSION=$(cd ${PROJECT_DIR}/py && ${PY_DIR}/bin/python3 -c "import versions; versions.cudnn_version()")
    TORCH_VERSION=$(${PY_DIR}/bin/python -c "from torch import __version__;print(__version__.split('+')[0])")
    cp ${PROJECT_DIR}/bazel-bin/libtorchtrt.tar.gz ${PROJECT_DIR}/py/wheelhouse/libtorchtrt-${TORCHTRT_VERSION}-cudnn${CUDNN_VERSION}-tensorrt${TRT_VERSION}-cuda${CUDA_VERSION}-libtorch${TORCH_VERSION}-x86_64-linux.tar.gz
}

libtorchtrt_pre_cxx11_abi() {
    cd /workspace/project/py
    mkdir -p /workspace/project/py/wheelhouse
    PY_BUILD_CODE=cp310-cp310
    PY_VERSION=3.10
    PY_NAME=python${PY_VERSION}
    PY_DIR=/opt/python/${PY_BUILD_CODE}
    PY_PKG_DIR=${PY_DIR}/lib/${PY_NAME}/site-packages/
    ${PY_DIR}/bin/python -m pip install --upgrade pip
    ${PY_DIR}/bin/python -m pip install -r py/requirements.txt
    ${PY_DIR}/bin/python -m pip install setuptools wheel auditwheel
    bazel build //:libtorchtrt --config pre_cxx11_abi --platforms //toolchains:ci_rhel_x86_64_linux -c opt --noshow_progress
    CUDA_VERSION=$(cd ${PROJECT_DIR}/py && ${PY_DIR}/bin/python3 -c "import versions; versions.cuda_version()")
    TORCHTRT_VERSION=$(cd ${PROJECT_DIR}/py && ${PY_DIR}/bin/python3 -c "import versions; versions.torch_tensorrt_version()")
    TRT_VERSION=$(cd ${PROJECT_DIR}/py && ${PY_DIR}/bin/python3 -c "import versions; versions.tensorrt_version()")
    CUDNN_VERSION=$(cd ${PROJECT_DIR}/py && ${PY_DIR}/bin/python3 -c "import versions; versions.cudnn_version()")
    TORCH_VERSION=$(${PY_DIR}/bin/python -c "from torch import __version__;print(__version__.split('+')[0])")
    cp ${PROJECT_DIR}/bazel-bin/libtorchtrt.tar.gz ${PROJECT_DIR}/py/wheelhouse/libtorchtrt-${TORCHTRT_VERSION}-pre-cxx11-abi-cudnn${CUDNN_VERSION}-tensorrt${TRT_VERSION}-cuda${CUDA_VERSION}-libtorch${TORCH_VERSION}-x86_64-linux.tar.gz
}
