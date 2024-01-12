FROM ubuntu:focal

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common wget build-essential libffi-dev libssl-dev curl git unzip vim \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.9 python3.9-dev python3-pip \
    && ln -s /usr/bin/python3.9 /usr/bin/python

# change the working directory to PaddleOCR
WORKDIR /PaddleOCR

# install miniconda
RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm -rf ~/miniconda3/miniconda.sh
ENV PATH="/root/miniconda3/bin:${PATH}"

# update conda
RUN conda update conda

# create a new conda environment
RUN conda create -n paddleocr python=3.9 -y

# activate the newly created conda environment
RUN echo "source ~/miniconda3/bin/activate paddleocr" >> ~/.bashrc

# install requirements and other dependencies
RUN /bin/bash -c "\
    source ~/miniconda3/bin/activate paddleocr && \
    python -m pip install paddlepaddle_gpu==2.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    python -m pip install opencv-python shapely pyclipper scikit-image imgaug Polygon3 lanms-neo lmdb PyYAML visualdl tensorboard && \
    conda install -c conda-forge Polygon3 pillow==9.5.0 tqdm loguru -y && \
    conda install -c anaconda cudatoolkit cudnn -y && \
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"

# transfer libcudnn.so from the torch installation to the cuda library, where paddleocr expects it
# also update the LD_LIBRARY_PATH to include the torch library
RUN MINICONDA_DIR=$(find / -name miniconda3 -type d | head -n 1) && \
    find $MINICONDA_DIR/envs/paddleocr -name libcudnn.so.8 -exec cp {} /usr/local/cuda/lib64/libcudnn.so \; && \
    echo "export LD_LIBRARY_PATH=$MINICONDA_DIR/envs/paddleocr/lib/python3.9/site-packages/torch/lib/" >> ~/.bashrc

# enable add-apt-repository
RUN apt-get update && apt install -y software-properties-common g++ gcc libgl1-mesa-glx

# update the repos to be able to install system76 cuda drivers
RUN gpg --keyserver keyserver.ubuntu.com --recv-keys 204DD8AEC33A7AFF
RUN gpg --export 204DD8AEC33A7AFF > key2.gpg
RUN mv key2.gpg /etc/apt/trusted.gpg.d/
RUN gpg --keyserver keyserver.ubuntu.com --recv-keys ACD442D1C8B7748B
RUN gpg --export ACD442D1C8B7748B > keyA.gpg
RUN mv keyA.gpg /etc/apt/trusted.gpg.d/
RUN add-apt-repository "deb http://apt.pop-os.org/proprietary focal main"
RUN echo "deb [trusted=yes] http://ppa.launchpad.net/system76-dev/stable/ubuntu focal main" | tee /etc/apt/sources.list.d/system76-dev.list

# install system76 cuda drivers
RUN apt-get update && apt-get install -y system76-cudnn-11.2

# copy the entire contents from PaddleOCR to the image. PaddleOCR is a local folder
COPY . /PaddleOCR

RUN /bin/bash -c "\
    source ~/miniconda3/bin/activate paddleocr && \
    pip install -r requirements.txt"

RUN apt install wormhole -y

RUN chmod +x start.sh

CMD [ "./start.sh" ]