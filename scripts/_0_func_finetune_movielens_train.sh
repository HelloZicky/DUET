dataset=$1
model=$2
cuda_num=$3

ITERATION=24300
SNAPSHOT=2430
MAX_EPOCH=20
ARCH_CONF_FILE="configs/${dataset}_conf.json"


GRADIENT_CLIP=5                     # !
BATCH_SIZE=1024

######################################################################
LEARNING_RATE=0.001
CHECKPOINT_PATH=../checkpoint/WWW2023/${dataset}_${model}
######################################################################

echo ${CHECKPOINT_PATH}
echo "Model save to ${CHECKPOINT_PATH}"


USER_DEFINED_ARGS="--model=${model} --num_loading_workers=1 --arch_config=${ARCH_CONFIG_CONTENT} --learning_rate=${LEARNING_RATE} \
--max_gradient_norm=${GRADIENT_CLIP} --batch_size=${BATCH_SIZE} --snapshot=${SNAPSHOT} --max_steps=${ITERATION} --checkpoint_dir=${CHECKPOIN\
T_PATH} --arch_config=${ARCH_CONF_FILE}"

dataset="../../../data/${dataset}/ood_generate_dataset_tiny_10_30u30i"


export CUDA_VISIBLE_DEVICES=${cuda_num}
echo ${USER_DEFINED_ARGS}
python ../main/multi_metric_vanilla_finetune_train.py \
--dataset=${dataset} \
${USER_DEFINED_ARGS}

echo "Training done: ${CHECKPOINT_PATH}"

