#! /bin/bash
#SBATCH -J ZJY_MLSelectTest
#SBATCH -p SP-V100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=%j.out
#SBATCH --error=%j.err
###SBATCH --profile=all

source /Share/THUDATA/Softwares/mpich3.3.sh

srun -o output_14.txt docker run --gpus all --rm -v /Share/THUDATA/UserHome/thunder_user/:/home 192.103.2.201:5000/picos_mpi4y:cupy82-cuda92_python37_pytorch17_matplotlib33_scipy15_mrcfile12_skimage017 sh -c "cd /home/zhujianying/Compressed_Sensing/test/zhujianying && python3 test_ml.py"
