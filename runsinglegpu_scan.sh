#!/bin/sh
# julia runexperiments.jl $SLURM_ARRAY_TASK_ID > ${SLURM_ARRAY_TASK_ID}_scan.log 2> ${SLURM_ARRAY_TASK_ID}_scan.error
for seed in 0 1 2 3 4; do
            julia runexperiments_scan.jl --seed $seed \
                                  --config configs/recombine_scan.jl \
                                  --condconfig configs/seq2seq_scan.jl \
                                  --copy \
				  											  --outdrop_test \
																	--generate \
                                  > checkpoints/SCANDataSet/logs/evaluate.seed.$seed.out \
                                  2> checkpoints/SCANDataSet/logs/evaluate.seed.$seed.err

done

#
# julia runexperiments_scan.jl --seed 3 \
#                          --config configs/recombine_scan.jl \
#                          --condconfig configs/seq2seq_scan.jl \
                                                   
