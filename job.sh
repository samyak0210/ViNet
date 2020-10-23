#!/bin/bash
#SBATCH --job-name=ucf
#SBATCH -A samyak
#SBATCH -n 1
#SBATCH -w gnode24
#SBATCH --gres=gpu:0
#SBATCH -o ../ucf.log
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module load cuda/9.0
module load cudnn/7-cuda-9.0

source activate env3

cat ./job.sh

#rsync -r samyak@gnode08:/ssd_scratch/cvit/samyak/DHF1K /ssd_scratch/cvit/samyak/
#echo "Done"
cd /ssd_scratch/cvit/samyak
rsync -r samyak@gnode08:/ssd_scratch/cvit/samyak/data .
rsync -r samyak@gnode08:/ssd_scratch/cvit/samyak/DHF1K .
# cd /home/navyasri/S3D_Transformer
# python3 train.py --num_hier 1 --model_val_path ./saved_models/1_hier.pt
# python3 train.py --dataset UCF --load_weight ./saved_models/no_trans_upsampling_reduced.pt --train_path_data /ssd_scratch/cvit/samyak/UCF/training/ --val_path_data /ssd_scratch/cvit/samyak/UCF/testing/ --model_val_path ./saved_models/ucf_single.pt
# python3 train.py --dataset UCF --multi_frame 32 --load_weight ./saved_models/new_train_multi_frame.pt --train_path_data /ssd_scratch/cvit/samyak/UCF/training/ --val_path_data /ssd_scratch/cvit/samyak/UCF/testing/ --model_val_path ./saved_models/ucf_multi.pt
#python3 train.py --train_random_idx True --multi_frame 32  --num_decoder_layers 3 --model_val_path new_train_multi_frame.pt
#python3 train.py --frame_no middle --model_val_path ./saved_models/no_trans_upsample_middle_frame_more_epochs.pt --load_weight ./saved_models/no_trans_upsample_middle_frame.pt
