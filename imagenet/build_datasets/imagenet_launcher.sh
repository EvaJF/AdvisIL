#!/usr/bin/env bash
#SBATCH --error=/home/users/efeillet/expe/imagenet/imagenet.e.log
#SBATCH --output=/home/users/efeillet/expe/imagenet/imagenet.o.log
#SBATCH --job-name=imagenet
#SBATCH --partition=gpu-test
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
source /home/users/efeillet/miniconda3/bin/activate
conda activate py37
nvidia-smi 
echo FOOD
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /home/users/efeillet/expe/imagenet/imagenet_subsetter.py /home/users/efeillet/expe/imagenet/imagenet_subsetter_food.cf
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /home/users/efeillet/images_list_files/compute_dataset_mean_from_images_list.py /home/users/efeillet/images_list_files/train100/imagenet_food/train.lst 
echo FLORA
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /home/users/efeillet/expe/imagenet/imagenet_subsetter.py /home/users/efeillet/expe/imagenet/imagenet_subsetter_flora.cf
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /home/users/efeillet/images_list_files/compute_dataset_mean_from_images_list.py /home/users/efeillet/images_list_files/train100/imagenet_flora/train.lst 
echo FAUNA
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /home/users/efeillet/expe/imagenet/imagenet_subsetter.py /home/users/efeillet/expe/imagenet/imagenet_subsetter_fauna.cf
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /home/users/efeillet/images_list_files/compute_dataset_mean_from_images_list.py /home/users/efeillet/images_list_files/train100/imagenet_fauna/train.lst 
echo RANDOM0
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /home/users/efeillet/expe/imagenet/imagenet_random_subsetter.py /home/users/efeillet/expe/imagenet/imagenet_random_subsetter0.cf
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /home/users/efeillet/images_list_files/compute_dataset_mean_from_images_list.py /home/users/efeillet/images_list_files/train100/imagenet_random_0/train.lst 
echo RANDOM1
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /home/users/efeillet/expe/imagenet/imagenet_random_subsetter.py /home/users/efeillet/expe/imagenet/imagenet_random_subsetter1.cf
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /home/users/efeillet/images_list_files/compute_dataset_mean_from_images_list.py /home/users/efeillet/images_list_files/train100/imagenet_random_1/train.lst 
echo RANDOM2
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /home/users/efeillet/expe/imagenet/imagenet_random_subsetter.py /home/users/efeillet/expe/imagenet/imagenet_random_subsetter2.cf
srun --nodes=1 --ntasks=1 --gres=gpu:1 python /home/users/efeillet/images_list_files/compute_dataset_mean_from_images_list.py /home/users/efeillet/images_list_files/train100/imagenet_random_2/train.lst & wait