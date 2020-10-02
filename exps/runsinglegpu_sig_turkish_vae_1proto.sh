#!/bin/sh
cd ../
# export CUDA_VISIBLE_DEVICES=3
export RECOMB_TASK=MORPH
VAE=true
LANG=turkish
for nproto in 1; do
	for hints in 4 8 16; do
		for seed in 0 1 2 3 4; do
			julia --project runexperiments.jl --seed $seed \
			--hints $hints \
			--config configs/recombine_sig.jl \
			--copy \
			--nproto $nproto \
			--epoch 25 \
			--subtask reinflection \
			--seperate_emb \
			--generate \
			--temp 1.0 \
			--lang ${LANG} \
			--Nsamples 180 \
			--N 180 \
			--rare_token \
			> ${RECOMB_CHECKPOINT_DIR}/SIGDataSet/${LANG}/logs/${nproto}proto.vae.${VAE}.hints.${hints}.seed.$seed.log \
			2> ${RECOMB_CHECKPOINT_DIR}/SIGDataSet/${LANG}/logs/${nproto}proto.vae.${VAE}.hints.${hints}.seed.$seed.err

			julia --project runexperiments.jl --seed $seed \
			--hints $hints \
			--config configs/recombine_sig.jl \
			--condconfig configs/seq2seq_sig.jl \
			--copy \
			--nproto $nproto \
			--subtask analyses \
			--seperate_emb \
			--lang ${LANG} \
			--baseline \
			--usegenerated \
			--loadprefix ${RECOMB_CHECKPOINT_DIR}/SIGDataSet/${LANG}/${nproto}proto.vae.${VAE}.hints.${hints}.seed.${seed}. \
			> ${RECOMB_CHECKPOINT_DIR}/SIGDataSet/${LANG}/logs/${nproto}proto.vae.${VAE}.hints.${hints}.seed.$seed.cond.log  \
			2> ${RECOMB_CHECKPOINT_DIR}/SIGDataSet/${LANG}/logs/${nproto}proto.vae.${VAE}.hints.${hints}.seed.$seed.cond.err
		done
	done
done
