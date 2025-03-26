
DIR="runs_outputs"
NAME="vppo"

# This will use 2 GPUs total: 1 process to train the model + 1 SGL GPU to do online inferencing

# TRAIN_PROCS=1    : num training processes
# SGL_BASE_GPU_ID=1: puts SGL server in GPU 1
# SGL_DP_SIZE=1    : means that SGL will use 1 GPU

export TRAIN_PROCS=1
export SGL_BASE_GPU_ID=$TRAIN_PROCS
export SGL_DP_SIZE=1

python run_spawn.py -P $TRAIN_PROCS \
    -m q1.5i \
    -o ${DIR}/${NAME}_42 \
    -t 0.35 --fast \
    -k 5 \
    --lr 1e-5 \
    -a vppo \
    --onlbsz 32 \
    --t500 \
    --template lcot \
    --dataset math \
    --maxtok 1024 \
    --epc 300 \
    --offepc 1 \
    -s 42 \
    --kl_ctl 1e-6
