#!/bin/sh

# git lfs
sudo apt update
sudo apt install -y git-lfs
git lfs install

# tmux
touch ~/.no_auto_tmux

# python 3 build deps
sudo timedatectl set-timezone Europe/Madrid
sudo apt-get update
sudo apt-get install -y build-essential \
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
curl https://pyenv.run | bash

# add to path
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'alias watch-gpu="watch -n .1 nvidia-smi"' >> ~/.bashrc
. ~/.bashrc

# install python3.11
pyenv install 3.11
pyenv global 3.11

# python env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt