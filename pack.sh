tar -xzf pack.tar.gz

rm -r train_data/nitin_crops
rm -r train_data/nitin_fullset

mv nitin_crops train_data
mv nitin_fullset train_data

cat export/det_train_set.txt >> train_data/dets_train.txt
cat export/det_val_set.txt >> train_data/dets_val.txt

cat export/rec_train_set.txt >> train_data/train_list.txt
while IFS= read -r line
do
  line=${line%%$'\t'*}
  echo "/PaddleOCR/train_data/$line"
done < train_data/train_list.txt > train_data/train_list_paths.txt

cat export/rec_train_set.txt >> train_data/alt_list.txt
while IFS= read -r line
do
  line=${line%%$'\t'*}
  echo "/PaddleOCR/train_data/$line"
done < train_data/alt_list.txt > train_data/alt_list_paths.txt

cat export/rec_val_set.txt >> train_data/val_list.txt
while IFS= read -r line
do
  line=${line%%$'\t'*}
  echo "/PaddleOCR/train_data/$line"
done < train_data/val_list.txt > train_data/val_list_paths.txt