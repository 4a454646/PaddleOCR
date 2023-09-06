export LD_LIBRARY_PATH=/root/miniconda3/envs/paddleocr/lib/python3.9/site-packages/torch/lib/
python3 tools/export_model.py -c ./trials/600_no_ai_5e-4/config.yml -o Global.pretrained_model=./trials/600_no_ai_5e-4  Global.save_inference_dir=./inference/best_no_ai
