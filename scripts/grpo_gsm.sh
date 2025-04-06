
DIR="runs_outputs"
NAME="grpo"

export TRAIN_PROCS=1
export SGL_BASE_GPU_ID=$TRAIN_PROCS
export SGL_DP_SIZE=2

sudo kill -kill `ps -ax | grep sgl | awk '{print $1}' | xargs`

torchrun --nproc-per-node=${TRAIN_PROCS} run_torch.py \
    -m q1.5i \
    -o ${DIR}/${NAME}_42 \
    -t 1.0 \
    -k 14 \
    --lr 5e-6 \
    -a grpo \
    --dataset gsm8k \
    --onlbsz 7 \
    --maxtok 1024 \
    --maxepochs 1000 \
    --offepc 1 \
    -s 42 \
    --kl_ctl 0.04 \
    --evalevery 10
