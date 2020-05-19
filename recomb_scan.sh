#!/bin/sh
# julia runexperiments.jl $SLURM_ARRAY_TASK_ID > ${SLURM_ARRAY_TASK_ID}_scan.log 2> ${SLURM_ARRAY_TASK_ID}_scan.error
for seed in 0 1; do
            julia runexperiments_scan.jl --seed $seed \
			                                  --config configs/recombine_scan.jl \
			                                  --copy \
																				--nproto 2 \
																				--epoch 4 \
							  											  --outdrop_test \
																				--generate \
			                                  > checkpoints/SCANDataSet/logs/evaluate.recomb.seed.$seed.out \
			                                  2> checkpoints/SCANDataSet/logs/evaluate.recomb.seed.$seed.err

						julia runexperiments_scan.jl --seed $seed \
																				--config configs/recombine_scan.jl \
																				--condconfig configs/seq2seq_scan.jl \
																				--usegenerated \
																				--nproto 2 \
																				--loadprefix checkpoints/SCANDataSet/Recombine_nproto_2_seed_${seed} \
																				> checkpoints/SCANDataSet/logs/evaluate.recomb.seed.$seed.cond.out \
																				2> checkpoints/SCANDataSet/logs/evaluate.recomb.seed.$seed.cond.err

done

#
# julia runexperiments_scan.jl --seed 3 \
#                          --config configs/recombine_scan.jl \
#                          --condconfig configs/seq2seq_scan.jl \
                                                   
