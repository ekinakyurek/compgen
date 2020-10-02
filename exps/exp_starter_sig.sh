#!/bin/bash

#  for lang  in spanish turkish swahili; do
#    for nproto in 0 1 2; do
#      for vae in vae novae; do
#        screen -X -S  runsinglegpu_sig_${lang}_${vae}_${nproto}proto.sh quit
# # 	screen -X -S runsinglegpu_sig_${lang}_baseline.sh quit
#      done
#    done
#  done


takejob () {
    export RECOMB_TASK=MORPH
    export RECOMB_CHECKPOINT_DIR=checkpoints/
    export RECOMB_DATA_DIR=data/
    export RECOMB_SIG_SUBDIR=SIGDataSet/
    if [ ! -d ../${RECOMB_CHECKPOINT_DIR}/SIGDataSet/turkish/logs ]; then
        for lang in turkish spanish swahili; do
            mkdir -p ../${RECOMB_CHECKPOINT_DIR}/SIGDataSet/${lang}/logs
        done
    fi
    echo "$1"
    salloc --gres=gpu:volta:1 --time=48:00:00 --constraint=xeon-g6 --cpus-per-task=5 --qos=high  srun $1
}
export -f takejob

# Baselines
# for lang  in spanish turkish swahili; do
#     screen -S runsinglegpu_sig_${lang}_baseline.sh -d -m bash -c  "takejob ./runsinglegpu_sig_baseline_${lang}.sh"
# done

# Models
for lang  in turkish spanish swahili; do
  for nproto in 1 0 2; do
      for vae in novae vae; do
          screen -S runsinglegpu_sig_${lang}_${vae}_${nproto}proto.sh  -d -m bash -c  "takejob ./runsinglegpu_sig_${lang}_${vae}_${nproto}proto.sh"
      done
  done
done
