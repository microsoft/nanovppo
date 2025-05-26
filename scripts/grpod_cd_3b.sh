DIR="/data/alsordon/runs_outputs"
NAME="grpod_cd_3b"

python run_torch.py \
    -m q3i \
    -o ${DIR}/${NAME}_42 \
    -t 1.0 \
    -k 4 \
    --lr 5e-6 \
    --guess_ctl 0.1 \
    -a grpod \
    --dataset cd \
    --onlbsz 16 \
    --offbsz 8 \
    --innbsz 4 \
    --maxtok 1024 \
    --maxepochs 1000 \
    --offepc 1 \
    -s 42 \
    --kl_ctl 0.001 \
    --evalevery 25 \
    --opt adamw
