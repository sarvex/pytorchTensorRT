#!/bin/bash

# Install dependencies
TRT_VERSION=$(python3 -c "import versions; versions.tensorrt_version()")
yum install -y ninja-build tensorrt-${TRT_VERSION}.*
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.17.0/bazelisk-linux-amd64 \
    && mv bazelisk-linux-amd64 /usr/bin/bazel \
    && chmod +x /usr/bin/bazel

cp ${dirname}/toolchains/ci_workspaces/WORKSPACE.x86_64.release.rhel ${dirname}/WORKSPACE
echo -e "READY"