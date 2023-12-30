# extract pack.tar.gz
tar -xzf pack.tar.gz
# move the folders nitin_crops and nitin_fullset into the aug_train folder
mv nitin_crops aug_train
mv nitin_fullset aug_train
# use cat to add the contents of the following files:
# export/det_train_set.txt export/det_val_set.txt export/rec_train_set.txt export/rec_val_set.txt
# to aug_train/dets_train.txt aug_train/dets_val.txt aug_train/train_list.txt aug_train/val_list.txt
cat export/det_train_set.txt >> aug_train/dets_train.txt
cat export/det_val_set.txt >> aug_train/dets_val.txt
cat export/rec_train_set.txt >> aug_train/train_list.txt
cat export/rec_val_set.txt >> aug_train/val_list.txt

# clean up folders