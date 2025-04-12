DIR="runs_outputs"
NAME="vppo"

export TRAIN_PROCS=1
export SGL_BASE_GPU_ID=$TRAIN_PROCS
export SGL_DP_SIZE=1

torchrun --nproc_per_node=$TRAIN_PROCS run_torch.py \
    -m q1.5i \
    -o ${DIR}/${NAME}_42 \
    -t 1.0 --fast \
    -k 5 \
    --lr 1e-5 \
    -a vppo \
    --onlbsz 32 \
    --t500 \
    --template lcot \
    --dataset math \
    --maxtok 1024 \
    --maxepochs 1000 \
    --offepc 1 \
    -s 42 \
    --kl_ctl 1e-6
