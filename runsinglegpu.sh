#!/bin/sh
# julia runexperiments.jl $SLURM_ARRAY_TASK_ID > ${SLURM_ARRAY_TASK_ID}_scan.log 2> ${SLURM_ARRAY_TASK_ID}_scan.error
for hints in 4; do #8 16 32; d
  for seed in 0 1 2 3; do
      for H in 512; do
        for E in 16; do
            julia runexperiments.jl --seed $seed \
                                  --hints $hints \
                                  --config configs/recombine_sig.jl \
                                  --condconfig configs/seq2seq_sig.jl \
                                  --H $H \
                                  --E $E \
                                  --copy \
                                  > checkpoints/SIGDataSet/logs/evaluate.hints-$hints.$seed.$H.$E.perm.out \
                                  2> checkpoints/SIGDataSet/logs/evaluate.hints-$hints.$seed.$H.$E.perm.err

        done
      done
  done
done

#
# julia runexperiments.jl --seed 3 \
#                         --hints 4 \
#                         --config configs/recombine_sig.jl \
#                         --condconfig configs/seq2seq_sig.jl \
