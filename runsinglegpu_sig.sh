#!/bin/sh
# julia runexperiments.jl $SLURM_ARRAY_TASK_ID > ${SLURM_ARRAY_TASK_ID}_scan.log 2> ${SLURM_ARRAY_TASK_ID}_scan.error
LANG=turkish

for hints in 4 8 16; do
  for seed in 0 1 2 3 4; do
          julia runexperiments.jl --seed $seed \
                                  --hints $hints \
                                  --config configs/recombine_sig.jl \
                                  --copy \
																	--nproto 2 \
															  	--subtask reinflection \
															  	--seperate_emb \
																	--kill_edit \
															  	--generate \
																	--lang ${LANG} \
	                                > checkpoints/SIGDataSet/${LANG}/logs/evaluate.hints-$hints.$seed.gen.out \
	                                2> checkpoints/SIGDataSet/${LANG}/logs/evaluate.hints-$hints.$seed.gen.err

					julia runexperiments.jl --seed $seed \
                                --hints $hints \
                                --config configs/recombine_sig.jl \
                                --condconfig configs/seq2seq_sig.jl \
                                --copy \
																--nproto 2 \
																--subtask analyses \
																--seperate_emb \
																--lang ${LANG} \
																--baseline \
																--usegenerated \
																--loadprefix checkpoints/SIGDataSet/${LANG}/Recombine_hints_${hints}_seed_${seed} \
                                > checkpoints/SIGDataSet/${LANG}/logs/evaluate.hints-$hints.$seed.cond.out \
                                2> checkpoints/SIGDataSet/${LANG}/logs/evaluate.hints-$hints.$seed.cond.err
    done
done

#
# julia runexperiments.jl --seed 3 \
#                         --hints 4 \
#                         --config configs/recombine_sig.jl \
#                         --condconfig configs/seq2seq_sig.jl \
