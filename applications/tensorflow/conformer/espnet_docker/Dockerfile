# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
FROM kaldiasr/kaldi:latest
RUN apt-get update && apt-get install -y \
       build-essential \
       cmake \
       git \
       python3 \
       python3-dev \
       python3-pip \
       pkg-config \
       libfreetype6-dev \
       libpng-dev \
       sox \
       wget \
       automake \
       autoconf \
       unzip \
       bc \
       ca-certificates \
       openssl \
       gfortran \
       libtool \
       subversion \
       libsndfile1-dev \
       ffmpeg && rm -rf /var/lib/apt/lists/*

RUN cd /home && git clone https://github.com/espnet/espnet
RUN sed -i '16,23d' /home/espnet/egs/aishell/asr1/path.sh # drop kenlm check which we do not need
RUN ln -s /opt/kaldi/ /home/espnet/tools/
RUN cd /home/espnet/tools && ./setup_anaconda.sh anaconda espnet 3.8 && make CPU_ONLY=0 
RUN cp /home/espnet/utils/data2json.sh /home/espnet/egs/aishell/asr1/utils/
