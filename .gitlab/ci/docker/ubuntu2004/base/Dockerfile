FROM ubuntu:20.04
LABEL maintainer "Vicente Adolfo Bolea Sanchez<vicente.bolea@kitware.com>"

ENV TZ=America/New_York

# Base dependencies for building VTK-m projects
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
      cmake \
      curl \
      g++ \
      git \
      git-lfs \
      libmpich-dev \
      libomp-dev \
      libtbb-dev \
      libhdf5-dev \
      mpich \
      ninja-build \
      software-properties-common

# Need to run git-lfs install manually on ubuntu based images when using the
# system packaged version
RUN git-lfs install
