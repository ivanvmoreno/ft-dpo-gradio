FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# git lfs
RUN sudo apt update && sudo apt install -y git-lfs
RUN git lfs install

# tmux
RUN touch ~/.no_auto_tmux

# python 3 build deps
RUN timedatectl set-timezone Europe/Madrid
RUN apt-get update && apt-get install -y build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    tk-dev \
    libdb-dev \
    libexpat1-dev \
    wget \
    curl \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev

# install pyenv
RUN curl https://pyenv.run | bash

# add to path
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
RUN echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
RUN echo 'alias watch-gpu="watch -n .1 nvidia-smi"' >> ~/.bashrc
RUN . ~/.bashrc

# install python 3.11
RUN pyenv install 3.11
RUN pyenv global 3.11
