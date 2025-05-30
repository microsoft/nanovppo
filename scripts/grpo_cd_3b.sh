DIR="./runs_outputs"
NAME="grpo_cd_3b"

python run_torch.py \
    -m q3i \
    -o ${DIR}/${NAME}_42 \
    -t 1.0 \
    -k 4 \
    --lr 5e-6 \
    -a grpo \
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
