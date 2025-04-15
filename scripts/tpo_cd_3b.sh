# This will use 2 GPUs total: 1 process to train the model + 1 SGL GPU to do online inferencing

export TRAIN_PROCS=1

DIR="/data/alsordon/runs_outputs"
NAME="tpo_features_2prin"

python run_torch.py \
    -m q3i \
    --gcheck 0 \
    -o ${DIR}/${NAME}_42 \
    -t 1.0 \
    -k 4 \
    --lr 5e-6 \
    -a tpo \
    -b vllm \
    --fast \
    --dataset cd \
    --onlbsz 16 \
    --offbsz 8 \
    --innbsz 1 \
    --maxtok 1024 \
    --maxepochs 1000 \
    --offepc 1 \
    -s 42 \
    --kl_ctl 0.001 \
    --evalevery 25 \
    --opt adamw8bit
