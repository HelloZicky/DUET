#!/bin/bash
date
dataset_list=("movielens_1m" "douban_book" "douban_music")
echo ${dataset_list}
line_num_list=(7828 21189 30819)
cuda_num_list=(0 1 2 3)
echo ${line_num_list}
length=${#dataset_list[@]}
for ((i=0; i<${length}; i++));
do
{
    dataset=${dataset_list[i]}
    cuda_num=${cuda_num_list[i]}
    for model in din gru4rec sasrec
    do
    {
          for type in _0_func_base_movielens_train _0_func_finetune_movielens_train _0_func_duet_movielens_train
          do
            {
              bash ${type}.sh ${dataset} ${model} ${cuda_num}
            } &
          done
    } &
    done
} &
done
wait # 等待所有任务结束
date
# bash _0_func_base_movielens_train.sh amazon_beauty sasrec 0
# bash _0_func_finetune_movielens_train.sh amazon_beauty sasrec 0
# bash _0_func_duet_movielens_train.sh amazon_beauty sasrec 0