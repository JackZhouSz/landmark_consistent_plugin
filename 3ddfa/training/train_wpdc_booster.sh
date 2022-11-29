#!/usr/bin/env bash

LOG_ALIAS=$1
LOG_DIR="logs"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/${LOG_ALIAS}_`date +'%Y-%m-%d_%H:%M.%S'`.log"
#echo $LOG_FILE

training/train_booster.py --arch="mobilenet_1" \
    --start-epoch=1 \
    --loss=wpdc \
    --snapshot="snapshot/phase1_wpdc" \
    --param-fp-train='/nfs/home/uss00054/projects/landmark_consistent_plugin/3ddfa/train.configs/param_all_norm.pkl' \
    --param-fp-val='/nfs/home/uss00054/projects/landmark_consistent_plugin/3ddfa/train.configs/param_all_norm_val.pkl' \
    --warmup=5 \
    --opt-style=resample \
    --resample-num=132 \
    --batch-size=1 \
    --base-lr=0.002 \
    --booster-weight=0.2 \
    --epochs=10 \
    --milestones=30,40 \
    --print-freq=1 \
    --devices-id=4 \
    --workers=8 \
    --filelists-train="/nfs/home/uss00054/projects/landmark_consistent_plugin/3ddfa/train.configs/train_aug_120x120.list.train" \
    --filelists-val="/nfs/home/uss00054/projects/landmark_consistent_plugin/3ddfa/train.configs/train_aug_120x120.list.val" \
    --root="/nfs/STG/CodecAvatar/lelechen/libingzeng/3DDFA/train_aug_120x120" \
    --log-file="${LOG_FILE}" \
    --resume="/nfs/home/uss00054/projects/landmark_consistent_plugin/3ddfa/models/phase1_wpdc_vdc.pth.tar"
