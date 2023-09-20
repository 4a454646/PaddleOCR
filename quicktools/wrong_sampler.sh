python3 quicktools/wrong_sampler.py -m cosine_imgaug_25_rotation -i train_imgs-card_crops-export.txt -ii train_card_crops.txt
python3 quicktools/wrong_sampler.py -m cosine_imgaug_25_rotation -i train_imgs-gc_ori-export.txt -ii train_gc.txt
python3 quicktools/wrong_sampler.py -m cosine_imgaug_25_rotation -i train_imgs-test-export.txt -ii train_icdar.txt
python3 quicktools/wrong_sampler.py -m cosine_imgaug_25_rotation -i train_imgs-train-export.txt -ii train_word_list.txt