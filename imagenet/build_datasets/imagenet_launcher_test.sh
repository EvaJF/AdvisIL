#!/usr/bin/env bash
#SBATCH --error=/home/users/efeillet/expe/imagenet/imagenet_test.e.log
#SBATCH --output=/home/users/efeillet/expe/imagenet/imagenet_test.o.log
#SBATCH --job-name=imagenet
#SBATCH --partition=gpu-test 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
source /home/users/efeillet/miniconda3/bin/activate
conda activate py37
nvidia-smi 
echo ANIMAL
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /home/users/efeillet/expe/imagenet/imagenet_subsetter.py /home/users/efeillet/expe/imagenet/imagenet_subsetter_animal.cf
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /home/users/efeillet/images_list_files/compute_dataset_mean_from_images_list.py /home/users/efeillet/images_list_files/train100/imagenet_animal/train.lst 
