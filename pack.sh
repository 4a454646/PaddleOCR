# extract pack.tar.gz
tar -xzf pack.tar.gz
# move the folders nitin_crops and nitin_fullset into the aug_train folder
mv nitin_crops aug_train
mv nitin_fullset aug_train
# use cat to add the contents of the following files:
# export/det_train_set.txt export/det_val_set.txt export/rec_train_set.txt export/rec_val_set.txt
# to train_data/dets_train.txt train_data/dets_val.txt train_data/train_list.txt train_data/val_list.txt
cat export/det_train_set.txt >> train_data/dets_train.txt
cat export/det_val_set.txt >> train_data/dets_val.txt
cat export/rec_train_set.txt >> train_data/train_list.txt
while IFS= read -r line
do
  line=${line%%$'\t'*}
  echo "/PaddleOCR/train_data/$line"
done < train_list > train_list_paths
cat export/rec_train_set.txt >> train_data/alt_list.txt
while IFS= read -r line
do
  line=${line%%$'\t'*}
  echo "/PaddleOCR/train_data/$line"
done < alt_list > alt_list_paths
cat export/rec_val_set.txt >> train_data/val_list.txt
cat export/rec_train_set.txt >> train_data/alt_list.txt
while IFS= read -r line
do
  line=${line%%$'\t'*}
  echo "/PaddleOCR/train_data/$line"
done < val_list > val_list_paths