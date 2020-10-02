#!/bin/sh

# for split in jump around_right; do
#   for nproto in 1; do
#     for vae in novae vae; do
#         screen -X -S  ${split}_${vae}_${nproto}proto.sh quit
#     done
#   done
# done

takejob () {
    export RECOMB_TASK=SCAN
    export RECOMB_CHECKPOINT_DIR=checkpoints/
    export RECOMB_DATA_DIR=data/
    if [ ! -d ${RECOMB_CHECKPOINT_DIR}/SCANDataSet/logs ]; then
                mkdir -p ${RECOMB_CHECKPOINT_DIR}/SCANDataSet/logs
    fi
    echo "$1"
    salloc --gres=gpu:volta:1 --time=48:00:00 --constraint=xeon-g6 --cpus-per-task=5 --qos=high  srun $1
}
export -f takejob

for split in around_right jump; do
  for nproto in 2 1 0; do
    for vae in novae vae; do
        screen -S ${split}_${vae}_${nproto}proto.sh  -d -m bash -c  "takejob ./${split}_${vae}_${nproto}proto.sh"
    done
  done
done

# takejob='salloc --immediate=60 -p gpu --gres=gpu:volta:1 --time=24:00:00 --constraint=xeon-g6 --cpus-per-task=5 --qos=high  srun'
# for split in jump around_right; do
#   for nproto in 0; do
#     for vae in vae; do
#         screen -S ${split}_${vae}_${nproto}proto.sh  -d -m bash -c  "$takejob ./${split}_${vae}_${nproto}proto.sh"
#     done
#   done
# done
