# clone source
cd /HDD4T/nguyenpdg/
git clone -b nguyenpdg https://github.com/giacnguyenbmt/PaddleOCR.git
cd PaddleOCR
# create conda env
yes | conda create --name nguyenpdg python=3.10
conda activate nguyenpdg
# install env
python -m pip install paddlepaddle-gpu -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install -r requirements.txt
# prepare data
python svtr_trainning/prepare_data.py
# download pretrained model
mkdir models
cd models
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_large_none_ctc_ch_train.tar
tar -xf rec_svtr_large_none_ctc_ch_train.tar
cd ..
# training
python tools/train.py -c configs/rec/rec_svtrnet_large_lpr.yml -o \
    Global.pretrained_model=models/rec_svtr_large_none_ctc_ch_train/best_accuracy.pdparams \
    Global.eval_batch_step="[0, 4478]" \
    Train.dataset.data_dir="/" \
    Train.dataset.label_file_list=[/media/lpr_vht_hnc_v4/train.txt] \
    Eval.dataset.data_dir="/" \
    Eval.dataset.label_file_list=[/media/lpr_vht_hnc_v4/val.txt]