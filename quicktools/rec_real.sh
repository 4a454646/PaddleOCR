python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/PP-OCRv3/real.yml -o Global.checkpoints=trials/aug_train4/best_accuracy
