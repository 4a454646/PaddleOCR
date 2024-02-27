export LD_LIBRARY_PATH=/root/miniconda3/envs/paddleocr/lib/
export FLAGS_use_cuda_managed_memory=true
python3 tools/default_train.py -c configs/det_r50_vd_db.yml