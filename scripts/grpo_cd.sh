
DIR="runs_outputs"
NAME="grpo"

# This will use 2 GPUs total: 1 process to train the model + 1 SGL GPU to do online inferencing

# TRAIN_PROCS=1    : num training processes
# SGL_BASE_GPU_ID=1: puts SGL server in GPU 1
# SGL_DP_SIZE=1    : means that SGL will use 1 GPU

export TRAIN_PROCS=1
export SGL_BASE_GPU_ID=$TRAIN_PROCS
export SGL_DP_SIZE=1

sudo kill -kill `ps -ax | grep sgl | awk '{print $1}' | xargs`

torchrun --nproc-per-node=${TRAIN_PROCS} run_torch.py \
    -m q1.5i \
    -o ${DIR}/${NAME}_42 \
    -t 1.0 \
    -k 4 \
    --lr 4e-6 \
    -a grpo \
    --dataset cd \
    --onlbsz 8 \
    --maxtok 1024 \
    --maxstp 500 \
    --offepc 1 \
    -s 42 \
    --kl_ctl 1e-6 \
    --evalevery 50
