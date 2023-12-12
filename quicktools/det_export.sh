export LD_LIBRARY_PATH=/root/miniconda3/envs/paddleocr/lib/python3.9/site-packages/torch/lib/
python3 tools/export_model.py \
    -c configs/det/det_r50_vd_db.yml \
    -o Global.pretrained_model="./output/det_r50_vd/best_accuracy" \
    Global.save_inference_dir="./output/custom_detmodel/"