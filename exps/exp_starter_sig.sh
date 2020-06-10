#!/bin/sh

# for lang  in spanish turkish; do
#   for nproto in 2; do
#     for vae in vae novae; do
#         screen -X -S  runsinglegpu_sig_${lang}_${vae}_${nproto}proto.sh quit
#     done
#   done
# done

# takejob='salloc --immediate=60 -p gpu --gres=gpu:volta:1 --time=24:00:00 --constraint=xeon-g6 --cpus-per-task=5 --qos=high  srun'
# for lang  in haida; do
#   for nproto in 0 1 2; do
#     for vae in novae vae; do
#         screen -S runsinglegpu_sig_${lang}_${vae}_${nproto}proto.sh  -d -m bash -c  "$takejob ./runsinglegpu_sig_${lang}_${vae}_${nproto}proto.sh"
#     done
#   done
# done

takejob='salloc --immediate=60 -p gpu --gres=gpu:volta:1 --time=24:00:00 --constraint=xeon-g6 --cpus-per-task=5 --qos=high  srun'
for lang  in haida; do
  # for nproto in 0 1 2; do
  #    for vae in novae vae; do
        screen -S runsinglegpu_sig_${lang}_baseline.sh  -d -m bash -c  "$takejob ./runsinglegpu_sig_baseline_${lang}.sh"
   #   done
   # done
done
