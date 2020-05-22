#!/bin/sh
# julia runexperiments.jl $SLURM_ARRAY_TASK_ID > ${SLURM_ARRAY_TASK_ID}_scan.log 2> ${SLURM_ARRAY_TASK_ID}_scan.error
export CUDA_VISIBLE_DEVICES=3
for seed in 1 2 3 4; do
            julia runexperiments_scan.jl --seed $seed \
			                                  --config configs/recombine_scan.jl \
			                                  --copy \
																				--nproto 2 \
																				--epoch 3 \
							  											  --outdrop_test \
																				--split template \
																				--splitmodifier around_right \
																				--generate \
																				--B 64 \
																				--optim 'Adam(lr=0.001)' \
																				--Nsamples 500 \
																				--N 300 \
																				--paug 0.2 \
																				--temp 0.2 \
			                                  > checkpoints/SCANDataSet/logs/evaluate.right.recomb.vae.seed.$seed.out \
			                                  2> checkpoints/SCANDataSet/logs/evaluate.right.recomb.vae.seed.$seed.err

						julia runexperiments_scan.jl --seed $seed \
																				--config configs/recombine_scan.jl \
																				--condconfig configs/seq2seq_scan.jl \
																				--usegenerated \
																		  	--split template \
																				--splitmodifier around_right \
																				--nproto 2 \
																				--paug 0.2 \
																				--temp 0.2 \
																				--loadprefix checkpoints/SCANDataSet/Recombine_nproto_2_vae_true_template_seed_${seed} \
																				> checkpoints/SCANDataSet/logs/evaluate.right.recomb.vae.seed.$seed.cond.out \
																				2> checkpoints/SCANDataSet/logs/evaluate.right.recomb.vae.seed.$seed.cond.err

done

#
# julia runexperiments_scan.jl --seed 3 \
#                          --config configs/recombine_scan.jl \
#                          --condconfig configs/seq2seq_scan.jl \
                                                   
