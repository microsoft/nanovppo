
DIR="runs_outputs"
NAME="grpo"

# This will use 2 GPUs total: 1 process to train the model + 1 SGL GPU to do online inferencing

# TRAIN_PROCS=1    : num training processes
# SGL_BASE_GPU_ID=1: puts SGL server in GPU 1
# SGL_DP_SIZE=1    : means that SGL will use 1 GPU

# -P 1        : 1 training process
# -a grpo     : GRPO algorithm
# -k 5        : 5 samples for each answer in GRPO
# --onlbsz 32 : 32 problems (per gpu) at each online sampling batch
# --epc 300   : 300 epochs of online training
# --offepc 4  : 4 epochs of offline training, per each online batch we do 4 epochs of offline training
# -s 42       : seed
# -t500       : test on MATH-500 (speeds up)

export TRAIN_PROCS=1
export SGL_BASE_GPU_ID=$TRAIN_PROCS
export SGL_DP_SIZE=2

sudo kill -kill `ps -ax | grep sgl | awk '{print $1}' | xargs`

torchrun --nproc-per-node=${TRAIN_PROCS} run_torch.py \
    -m q1.5i \
    -o ${DIR}/${NAME}_42 \
    -t 1.0 \
    -k 8 \
    --lr 3e-5 \
    -a grpo \
    --t500 \
    --fast \
    --template lcot \
    --dataset math \
    --onlbsz 16 \
    --offbsz 8 \
    --maxtok 1024 \
    --maxstp 12000 \
    --offepc 1 \
    -s 42 \
    --kl_ctl 1e-6 \
    --evalevery 200
