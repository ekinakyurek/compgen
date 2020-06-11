#!/bin/sh
cd ../
# export CUDA_VISIBLE_DEVICES=3
export RECOMB_TASK=SCAN
VAE=true
for nproto in 2; do
	for seed in 0 1 2 3 4; do
		julia --project runexperiments_scan.jl --seed $seed \
		--config configs/recombine_scan.jl \
		--copy \
		--nproto 2 \
		--epoch 8 \
		--outdrop_test \
		--split add_prim \
		--splitmodifier jump \
		--generate \
		--beam \
		--paug 0.01 \
		> checkpoints/SCANDataSet/logs/${nproto}proto.vae.${VAE}.jump.seed.$seed.log \
		2> checkpoints/SCANDataSet/logs/${nproto}proto.vae.${VAE}.jump.seed.$seed.err

		julia --project runexperiments_scan.jl --seed $seed \
		--config configs/recombine_scan.jl \
		--condconfig configs/seq2seq_scan.jl \
		--usegenerated \
		--split add_prim \
		--splitmodifier jump \
		--nproto 2 \
		--paug 0.01 \
		--baseline \
		--usegenerated \
		--loadprefix checkpoints/SCANDataSet/${nproto}proto.vae.${VAE}.jump.seed.${seed}. \
		> checkpoints/SCANDataSet/logs/${nproto}proto.vae.${VAE}.jump.seed.$seed.cond.log  \
		2> checkpoints/SCANDataSet/logs/${nproto}proto.vae.${VAE}.jump.seed.$seed.cond.err
	done
done
