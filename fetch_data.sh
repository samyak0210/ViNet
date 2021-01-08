#!/usr/bin/env bash

fetch_site='http://cvsp.cs.ntua.gr/research/stavis/data'
# define the path where the data will be stored
data_root='/ssd_scratch/cvit/samyak/data'
# comment each line if you want to download a subset of the datasets
db_names=(
          'AVAD'
          'Coutrot_db1'
          'Coutrot_db2'
          'DIEM'
          'ETMD_av'
          'SumMe'
         )

echo '############################# Downloading Video Frames #############################'
mkdir -p ${data_root}'/video_frames'
for db_name in ${db_names[@]}; do
     echo 'Downloading ' $db_name '  ...'
     wget $fetch_site'/video_frames/'$db_name'.tar.gz' -O $data_root'/video_frames/'$db_name'.tar.gz'
     tar -xzf $data_root'/video_frames/'$db_name'.tar.gz' -C $data_root'/video_frames/'
     rm $data_root'/video_frames/'$db_name'.tar.gz'
done

echo '############################# Downloading Video Audio #############################'
mkdir -p ${data_root}'/video_audio'
for db_name in ${db_names[@]}; do
     echo 'Downloading ' $db_name '  ...'
     wget $fetch_site'/video_audio/'$db_name'.tar.gz' -O $data_root'/video_audio/'$db_name'.tar.gz'
     tar -xzf $data_root'/video_audio/'$db_name'.tar.gz' -C $data_root'/video_audio/'
     rm $data_root'/video_audio/'$db_name'.tar.gz'
done

echo '############################# Downloading Annotations ################################'
mkdir -p ${data_root}'/annotations'
for db_name in ${db_names[@]}; do
     echo 'Downloading ' $db_name '  ...'
     wget $fetch_site'/annotations/'$db_name'.tar.gz' -O $data_root'/annotations/'$db_name'.tar.gz'
     tar -xzf $data_root'/annotations/'$db_name'.tar.gz' -C $data_root'/annotations/'
     rm $data_root'/annotations/'$db_name'.tar.gz'
done

echo '############################# Downloading Fold Lists ################################'
wget $fetch_site'/fold_lists.tar.gz' -O $data_root'/fold_lists.tar.gz'
tar -xzf $data_root'/fold_lists.tar.gz' -C $data_root'/'
rm $data_root'/fold_lists.tar.gz'

echo '############################# Downloading Pretrained Models #############################'
wget $fetch_site'/pretrained_models.tar.gz' -O $data_root'/pretrained_models.tar.gz'
tar -xzf $data_root'/pretrained_models.tar.gz' -C $data_root'/'
rm $data_root'/pretrained_models.tar.gz'
