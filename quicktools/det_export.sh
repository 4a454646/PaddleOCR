export LD_LIBRARY_PATH=/usr/lib/cuda-11.2/targets/x86_64-linux/lib/:/root/miniconda3/envs/paddleocr/lib/python3.9/site-packages/torch/lib/

while getopts "m:" opt; do
  case $opt in
    m)
      model=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

python3 tools/export_model.py \
    -c "${model}/config.yml" \
    -o Global.pretrained_model="${model}/best_accuracy" \
    Global.save_inference_dir="${model}/model"