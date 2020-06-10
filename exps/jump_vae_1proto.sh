#!/bin/sh
cd ../
# export CUDA_VISIBLE_DEVICES=3
export RECOMB_TASK=SCAN
VAE=true
for nproto in 1; do
	for seed in 0 1 2 3 4; do
		julia runexperiments_scan.jl --seed $seed \
		--config configs/recombine_scan.jl \
		--copy \
		--nproto $nproto \
		--epoch 8 \
		--outdrop_test \
		--split add_prim \
		--splitmodifier jump \
		--generate \
		--paug 0.01 \
		--temp 0.4 \
		> checkpoints/SCANDataSet/logs/${nproto}proto.vae.${VAE}.jump.seed.$seed.log \
		2> checkpoints/SCANDataSet/logs/${nproto}proto.vae.${VAE}.jump.seed.$seed.err

		julia runexperiments_scan.jl --seed $seed \
		--config configs/recombine_scan.jl \
		--condconfig configs/seq2seq_scan.jl \
		--usegenerated \
		--split add_prim \
		--splitmodifier jump \
		--nproto $nproto \
		--paug 0.01 \
		--baseline \
		--usegenerated \
		--loadprefix checkpoints/SCANDataSet/${nproto}proto.vae.${VAE}.jump.seed.${seed}. \
		> checkpoints/SCANDataSet/logs/${nproto}proto.vae.${VAE}.jump.seed.$seed.cond.log  \
		2> checkpoints/SCANDataSet/logs/${nproto}proto.vae.${VAE}.jump.seed.$seed.cond.err
	done
done