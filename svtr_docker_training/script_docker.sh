docker run --gpus all --name "nguyenpdg_paddleocr_svtr" -v /HDD4T/nguyenpdg/:/code -v /media/a100-5g/nguyenpdg/:/data --shm-size=64G --network=host --ulimit memlock=-1 -it pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime /bin/bash

# To fix GPG key error when running apt-get update
rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install system dependencies for opencv-python and install git, nano, wget
apt-get update && apt-get install -y libgl1 libglib2.0-0 git nano wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setup env
python -m pip install paddlepaddle-gpu -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# Clone source code
cd /code/ \
    && git clone -b nguyenpdg https://github.com/giacnguyenbmt/PaddleOCR.git \
    && cd PaddleOCR \
    && pip install -r requirements.txt

# prepare data
python svtr_docker_training/prepare_data.py /data/lpr_vht_hnc_v4/train /data/lpr_vht_hnc_v4/synth /data/lpr_vht_hnc_v4/docker_train.txt
python svtr_docker_training/prepare_data.py /data/lpr_vht_hnc_v4/validate none /data/lpr_vht_hnc_v4/docker_val.txt

# download pretrained model
mkdir models \
    && cd models \
    && wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_large_none_ctc_ch_train.tar \
    && tar -xf rec_svtr_large_none_ctc_ch_train.tar \
    && cd ..

# train with a specific gpu
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0' tools/train.py -c configs/rec/rec_svtrnet_large_lpr.yml -o \
    Global.pretrained_model=models/rec_svtr_large_none_ctc_ch_train/best_accuracy.pdparams \
    Global.eval_batch_step="[0, 4478]" \
    Train.dataset.data_dir="/" \
    Train.dataset.label_file_list=[/data/lpr_vht_hnc_v4/docker_train.txt] \
    Eval.dataset.data_dir="/" \
    Eval.dataset.label_file_list=[/data/lpr_vht_hnc_v4/docker_val.txt]
