FROM ubuntu:22.04
USER root
ARG DEBIAN_FRONTEND=noninteractive


# 这一步解决清华源apt https证书问题，参考：https://juejin.cn/post/7033412379727626247
RUN apt update && apt install -y ca-certificates --reinstall && apt autoremove

COPY ./ubuntu.sources.list /etc/apt/sources.list

RUN apt update \
    && apt install -y \
       wget curl vim iputils-ping net-tools telnet build-essential libopenmpi-dev openmpi-bin openmpi-common \
       libglib2.0-0 libgl1-mesa-glx nginx zip unzip \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p /root/miniconda3 \
    && rm ~/miniconda.sh \
    && ln -s /root/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /root/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
    && curl -sL https://deb.nodesource.com/setup_14.x | bash - \
    && apt install -y nodejs \
    && apt autoremove \
    && rm -rf /var/lib/apt/lists/* 

ENV PATH /root/miniconda3/bin:$PATH
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu/openmpi/lib:$LD_LIBRARY_PATH

RUN conda create -y --name py11 python=3.11 && rm -rf /root/miniconda3/envs/py11/compiler_compat/ld

ENV CONDA_DEFAULT_ENV py11
ENV CONDA_PREFIX /root/miniconda3/envs/py11
ENV PATH $CONDA_PREFIX/bin:$PATH
ENV PIP_INDEX https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./requirements.txt /tmp/requirements.txt
RUN conda run -n py11 pip install -i ${PIP_INDEX} -r /tmp/requirements.txt
RUN conda run -n py11 pip install -i ${PIP_INDEX} ollama \
    && wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip -O /tmp/wordnet.zip \
    && wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip -O /tmp/punkt.zip \
    && mkdir -p /root/miniconda3/envs/py11/nltk_data \
    && unzip /tmp/wordnet.zip -d /root/miniconda3/envs/py11/nltk_data/corpora \
    && unzip /tmp/punkt.zip -d /root/miniconda3/envs/py11/nltk_data/tokenizers \
    && rm -rf /tmp/wordnet.zip /tmp/punkt.zip \
    && /root/miniconda3/envs/py11/bin/pip uninstall -y onnxruntime-gpu \
    && /root/miniconda3/envs/py11/bin/pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

ENV HF_ENDPOINT=https://hf-mirror.com
