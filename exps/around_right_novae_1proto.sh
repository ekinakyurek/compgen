#!/bin/sh
cd ../
# export CUDA_VISIBLE_DEVICES=3
export RECOMB_TASK=SCAN
VAE=false
for nproto in 1; do
	for seed in 0 1 2 3 4; do
		julia --project runexperiments_scan.jl --seed $seed \
		--config configs/recombine_scan.jl \
		--copy \
		--nproto $nproto \
		--epoch 3 \
		--outdrop_test \
		--split template \
		--splitmodifier around_right \
		--generate \
		--kill_edit \
		--B 64 \
		--optim 'Adam(lr=0.001)' \
		--Nsamples 500 \
		--N 300 \
		--paug 0.2 \
		--temp 0.5 \
		> ${RECOMB_CHECKPOINT_DIR}/SCANDataSet/logs/${nproto}proto.vae.${VAE}.around_right.seed.$seed.log \
		2> ${RECOMB_CHECKPOINT_DIR}/SCANDataSet/logs/${nproto}proto.vae.${VAE}.around_right.seed.$seed.err

		julia --project runexperiments_scan.jl --seed $seed \
		--config configs/recombine_scan.jl \
		--condconfig configs/seq2seq_scan.jl \
		--usegenerated \
		--split template \
		--splitmodifier around_right \
		--nproto  $nproto  \
		--paug 0.2 \
		--temp 0.5 \
		--baseline \
		--kill_edit \
		--usegenerated \
		--loadprefix ${RECOMB_CHECKPOINT_DIR}/SCANDataSet/${nproto}proto.vae.${VAE}.around_right.seed.${seed}. \
		> ${RECOMB_CHECKPOINT_DIR}/SCANDataSet/logs/${nproto}proto.vae.${VAE}.around_right.seed.$seed.cond.log  \
		2> ${RECOMB_CHECKPOINT_DIR}/SCANDataSet/logs/${nproto}proto.vae.${VAE}.around_right.seed.$seed.cond.err
	done
done

#
# julia --project runexperiments_scan.jl --seed 3 \
#                          --config configs/recombine_scan.jl \
#                          --condconfig configs/seq2seq_scan.jl \
                                                   
