#!/bin/sh
cd ../
# export CUDA_VISIBLE_DEVICES=3
export RECOMB_TASK=MORPH
VAE=false
LANG=swahili
for nproto in 2; do
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
			--kill_edit \
			--temp 0.5 \
			--lang ${LANG} \
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
			--kill_edit \
			--loadprefix ${RECOMB_CHECKPOINT_DIR}/SIGDataSet/${LANG}/${nproto}proto.vae.${VAE}.hints.${hints}.seed.${seed}. \
			> ${RECOMB_CHECKPOINT_DIR}/SIGDataSet/${LANG}/logs/${nproto}proto.vae.${VAE}.hints.${hints}.seed.$seed.cond.log  \
			2> ${RECOMB_CHECKPOINT_DIR}/SIGDataSet/${LANG}/logs/${nproto}proto.vae.${VAE}.hints.${hints}.seed.$seed.cond.err
		done
	done
done
