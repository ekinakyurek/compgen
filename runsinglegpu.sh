#!/bin/bash
# A shell script to print each number five times.
GPUID="$1"
_B=512
_H=512
_EPOCH=30
_CONCATZ=true
_AEPOCH=20
cnt=0
for E in 8 16 32
do
    for  Z in 8 16 32
    do
	for kl_rate in 0.1 0.05
	do	    
	    for pdrop in 0.0 0.4
	    do
		for fb_rate in 2.0 4.0 8.0
		do
		    for lr in 0.001 0.002 0.004
		    do
			if (( ($cnt % 8) == $GPUID ))
			then
			CUDA_VISIBLE_DEVICES=$GPUID julia runexperiments.jl $GPUID  --B $B\
					    --H $H --E $E --Z $Z --kl_rate $kl_rate --fb_rate $fb_rate\
					    --lr $lr --epoch $_EPOCH --concatz $_CONCATZ --aepoch $_AEPOCH
			fi
			let "cnt+=1"
			if (( $cnt == 1))
			then
			    break
			fi
		    done
		done
	    done
	done
    done
done

