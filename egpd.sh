#!/bin/bash
# The interpreter used to execute the script

#"#SBATCH" directives that convey submission options:

#SBATCH --job-name=egpd
#SBATCH --mail-user=zjf@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --time=1-00:00:00
#SBATCH --account=jjparkcv0
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --output=/home/%u/%x-gpu2-%j.log
#SBATCH --cpus-per-task=2

# --account=eecs542s001f23_class
# eecs595f23_class
# jjparkcv0

eval "$(conda shell.bash hook)"
conda init bash
conda activate eg3d
cd /home/zjf/repos/prolificdreamer

export CUDA_HOME=~/miniconda3

python main.py --text "A peach." --iters 25000 --lambda_entropy 10 --scale 7.5 --n_particles 1 --h 256  --w 256 --workspace exp-nerf-stage1/ --cfg=ffhq  --gen_pose_cond=True --data=lalla   --val_radius=3.0 --use_pretrained=/home/zjf/Downloads/ffhqrebalanced512-64.pkl