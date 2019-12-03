#!/bin/bash
# A shell script to print each number five times.
GPUID="$1"
_B=16
_H=512
_EPOCH=30
#_CONCATZ=true
#_AEPOCH=15
cnt=0
for lang in spanish turkish
do
	for concatz in true false
	do
		for E in 8 16 32
	    do
				for  Z in 8 16 32
				do
		    	for kl_rate in 0.1 0.05
		    	do
						for pdrop in 0.2 0.4 0.5
						do
			    		for fb_rate in 2.0 4.0 8.0
			    		do
								for lr in 0.002 0.004 0.005
								do
									for aepoch in 10 15 25
									do
						    		if (( $cnt%8==$GPUID))
									  then
											CUDA_VISIBLE_DEVICES=$GPUID julia runexperiments.jl $GPUID --lang $lang --B $_B \
															    --H $_H --E $E --Z $Z --kl_rate $kl_rate --fb_rate $fb_rate \
															    --lr $lr --epoch $_EPOCH --concatz $concatz --aepoch $aepoch

									  fi
										let "cnt+=1"
								done
							done
			    	done
					done
		    done
			done
	  done
	done
done
