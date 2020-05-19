#!/bin/sh
# julia runexperiments.jl $SLURM_ARRAY_TASK_ID > ${SLURM_ARRAY_TASK_ID}_scan.log 2> ${SLURM_ARRAY_TASK_ID}_scan.error
for seed in 2 3 4; do
          julia runexperiments_scan.jl --seed $seed \
			                                  --config configs/recombine_scan.jl \
																				--generate \
																				--nproto 0 \
																				--writedrop 0.0 \
																				--outdrop 0.5 \
																				--epoch 4 \
			                                  > checkpoints/SCANDataSet/logs/evaluate.rnnvae.seed.$seed.out \
			                                  2> checkpoints/SCANDataSet/logs/evaluate.rnnvae.seed.$seed.err


					julia runexperiments_scan.jl --seed $seed \
																			--config configs/recombine_scan.jl \
																			--condconfig configs/seq2seq_scan.jl \
																			--usegenerated \
																			--nproto 0 \
																			--loadprefix checkpoints/SCANDataSet/Recombine_nproto_0_seed_${seed} \
																			> checkpoints/SCANDataSet/logs/evaluate.rnnvae.seed.$seed.cond.out \
																			2> checkpoints/SCANDataSet/logs/evaluate.rnnvae.seed.$seed.cond.err

done

#
# julia runexperiments_scan.jl --seed 3 \
#                          --config configs/recombine_scan.jl \
#                          --condconfig configs/seq2seq_scan.jl \
                                                   
