#!/bin/sh
# julia runexperiments.jl $SLURM_ARRAY_TASK_ID > ${SLURM_ARRAY_TASK_ID}_scan.log 2> ${SLURM_ARRAY_TASK_ID}_scan.error
for hints in 4 8 16 32; do
  for seed in 0 1 2 3 4; do
          julia runexperiments.jl --seed $seed \
                                --hints $hints \
                                --config configs/recombine_sig.jl \
                                --condconfig configs/seq2seq_sig.jl \
                                --copy \
																--subtask reinflection \
																--seperate_emb \
																--generate \
                                > checkpoints/SIGDataSet/logs/evaluate.hints-$hints.$seed.augmented.out \
                                2> checkpoints/SIGDataSet/logs/evaluate.hints-$hints.$seed.augmented.err
    done
done

#
# julia runexperiments.jl --seed 3 \
#                         --hints 4 \
#                         --config configs/recombine_sig.jl \
#                         --condconfig configs/seq2seq_sig.jl \
