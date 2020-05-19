#!/bin/sh
# julia runexperiments.jl $SLURM_ARRAY_TASK_ID > ${SLURM_ARRAY_TASK_ID}_scan.log 2> ${SLURM_ARRAY_TASK_ID}_scan.error
for seed in 2 3 4; do
						julia runexperiments_scan.jl --seed $seed \
																				--config configs/recombine_scan.jl \
																				--copy \
																				--nproto 1 \
																				--outdrop_test \
																			  --epoch 4 \
																				--generate \
																				> checkpoints/SCANDataSet/logs/evaluate.proto.seed.$seed.out \
																				2> checkpoints/SCANDataSet/logs/evaluate.proto.seed.$seed.err


						julia runexperiments_scan.jl --seed $seed \
																				--config configs/recombine_scan.jl \
																				--condconfig configs/seq2seq_scan.jl \
																				--usegenerated \
																				--nproto 1 \
																				--loadprefix checkpoints/SCANDataSet/Recombine_nproto_1_seed_${seed} \
																				> checkpoints/SCANDataSet/logs/evaluate.proto.seed.$seed.cond.out \
																				2> checkpoints/SCANDataSet/logs/evaluate.proto.seed.$seed.cond.err


done

#
# julia runexperiments_scan.jl --seed 3 \
#                          --config configs/recombine_scan.jl \
#                          --condconfig configs/seq2seq_scan.jl \
                                                   
