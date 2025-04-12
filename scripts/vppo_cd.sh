# This will use 2 GPUs total: 1 process to train the model + 1 SGL GPU to do online inferencing

# TRAIN_PROCS=1    : num training processes
# SGL_BASE_GPU_ID=1: puts SGL server in GPU 1
# SGL_DP_SIZE=1    : means that SGL will use 1 GPU

export TRAIN_PROCS=1
export SGL_BASE_GPU_ID=$TRAIN_PROCS
export SGL_DP_SIZE=3

DIR="runs_outputs"
NAME="vppo_cd"

sudo kill -kill `ps -ax | grep sgl | awk '{print $1}' | xargs`

torchrun --nproc-per-node=${TRAIN_PROCS} run_torch.py \
    -m q1.5i \
    -o ${DIR}/${NAME}_42 \
    -t 1.0 \
    -k 4 \
    --lr 5e-6 \
    -a vppo \
    --dataset cd \
    --onlbsz 16 \
    --offbsz 8 \
    --innbsz 1 \
    --maxtok 1024 \
    --maxepochs 1000 \
    --offepc 1 \
    -s 42 \
    --kl_ctl 0.001 \
    --evalevery 25
