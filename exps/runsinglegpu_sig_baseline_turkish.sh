#!/bin/sh
cd ../
# export CUDA_VISIBLE_DEVICES=3
export RECOMB_TASK=MORPH
LANG=turkish
for hints in 4 8 16; do
	for seed in 0 1 2 3 4; do
		julia --project runexperiments.jl --seed $seed \
		--hints $hints \
		--config configs/recombine_sig.jl \
		--condconfig configs/seq2seq_sig.jl \
		--copy \
		--nproto 2 \
		--subtask analyses \
		--seperate_emb \
		--lang ${LANG} \
		--baseline \
		> ${RECOMB_CHECKPOINT_DIR}/SIGDataSet/${LANG}/logs/baseline.hints.${hints}.seed.$seed.cond.log  \
		2> ${RECOMB_CHECKPOINT_DIR}/SIGDataSet/${LANG}/logs/baseline.hints.${hints}.seed.$seed.cond.err
	done
done
