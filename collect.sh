#!/bin/bash
std(){
    awk '{sum+=$1; sumsq+=$1*$1}END{print "\\pm " sqrt(sumsq/NR - (sum/NR)**2)}'
}

mean(){
    awk 'BEGIN { sum=0 } { sum+=$1 } END {print sum / NR}'
}

stdmean(){
    echo -n "TEST:"
    mu=$(printf "$1" |  sed -n 'p;n'  | mean | tr -d '\n')
    sigma=$(printf "$1" | sed -n 'p;n' | std | tr -d '\n')
    echo -n "$mu ($sigma)"
    echo -n "\tVAL: "
    mu=$(printf "$1" |  sed -n 'n;p'  | mean | tr -d '\n')
    sigma=$(printf "$1" | sed -n 'n;p' | std | tr -d '\n')
    echo -n "$mu ($sigma)"
    echo
}

stdmean(){
    echo -n "TEST:"
    mu=$(printf "$1" |  sed -n 'p;n' | tr  '\n' ',')
    echo -n  "[$mu]"
    echo -n "\tVAL: "
    mu=$(printf "$1" |  sed -n 'n;p'  | tr '\n' ',')
    echo -n "[$mu]"
    echo
}



rootfolder=$(pwd)
for lang in spanish turkish swahili;do
    cd ${rootfolder}/checkpoints/SIGDataSet/${lang}/
    nproto=2
    model=baseline
    vae=true
    echo "BASELINE:"
    for i in 4 8 16; do
	numbers1=$(grep -o 'acc = [^,]*' $nproto*vae.${vae}*${model}condconfig | grep "hints.${i}" | awk -F'= ' '{print $2}'  2>/dev/null)
	line1=$(echo "$numbers1" | wc -l)
	if [ "$line1" != 10 ]; then
	    continue
	fi
	echo -n "${lang}.baseline.hints.${i}.acc:"
	stdmean "$numbers1"
	numbers2=$(grep -o 'f1 = [^\)]*' $nproto*vae.${vae}*${model}condconfig | grep "hints.${i}"  | awk -F'= ' '{print $2}'  2> /dev/null)
	echo -n "${lang}.baseline.hints.${i}.f1:"
	stdmean "$numbers2"

	numbers3=$(grep -o 'acc = [^,]*' $nproto*vae.${vae}*${model}cond_easy_config | grep "hints.${i}" | awk -F'= ' '{print $2}'  2>/dev/null)
	line3=$(echo "$numbers3" | wc -l)
	if [ "$line3" != 10 ]; then
	    continue
	fi
	echo -n "${lang}.baseline.hints.${i}.acc.easy:"
	stdmean "$numbers3"
	numbers4=$(grep -o 'f1 = [^\)]*' $nproto*vae.${vae}*${model}cond_easy_config | grep "hints.${i}"  | awk -F'= ' '{print $2}'  2> /dev/null)
	echo -n "${lang}.baseline.hints.${i}.f1.easy:"
	stdmean "$numbers4"
    done

    model=augmented
    for nproto in 0 1 2 ;do
	for vae in false true; do
	    echo "MODEL: ${nproto}proto.vae.${vae}"
	    for i in 4 8 16; do
		numbers5=$(grep -o 'acc = [^,]*' $nproto*vae.${vae}*${model}condconfig | grep "hints.${i}" | awk -F'= ' '{print $2}'  2>/dev/null)
		line5=$(echo "$numbers5" | wc -l)
		if [ "$line5" != 10 ]; then
		    continue
		fi
		echo -n "${lang}.${nproto}proto.vae.${vae}.hints.${i}.acc:"
		stdmean "$numbers5"
		numbers6=$(grep -o 'f1 = [^\)]*' $nproto*vae.${vae}*${model}condconfig | grep "hints.${i}"  | awk -F'= ' '{print $2}'  2> /dev/null)
		echo -n "${lang}.${nproto}proto.vae.${vae}.hints.${i}.f1:"
		stdmean "$numbers6"


		numbers7=$(grep -o 'acc = [^,]*' $nproto*vae.${vae}*${model}cond_easy_config | grep "hints.${i}" | awk -F'= ' '{print $2}'  2>/dev/null)
		line7=$(echo "$numbers7" | wc -l)
		if [ "$line7" != 10 ]; then
                    continue
		fi
		echo -n "${lang}.${nproto}proto.vae.${vae}.hints.${i}.acc.easy:"
		stdmean "$numbers7"
		numbers8=$(grep -o 'f1 = [^\)]*' $nproto*vae.${vae}*${model}cond_easy_config | grep "hints.${i}"  | awk -F'= ' '{print $2}'  2> /dev/null)
		echo -n "${lang}.${nproto}proto.vae.${vae}.hints.${i}.f1.easy:"
		stdmean "$numbers8"
	    done
	done
    done
done
	    
echo "====SCAN Results===="
cd ${rootfolder}/checkpoints/SCANDataSet/
for split in jump around_right; do
    for nproto in 0 1 2;do
	for vae in false true;do
  	    numbers8=$(grep -o 'acc = [^,]*' $nproto*augmentedcondconfig | grep "vae.${vae}.${split}.seed" | awk -F'= ' '{print $2}'  2> /dev/null)
	    line8=$(echo "$numbers8" | wc -l)
	    if [ "$line8" != 10 ]; then
		 continue
	    fi
	    echo "MODEL: ${split}.${nproto}proto.vae.${vae}"
	    echo -n "${split}.${nproto}proto.vae.${vae}.acc:"
	    stdmean "$numbers8"
	    numbers9=$(grep -o 'f1 = [^\)]*' $nproto*augmentedcondconfig | grep "vae.${vae}.${split}.seed" | awk -F'= ' '{print $2}' 2> /dev/null)
	    echo -n "${split}.${nproto}proto.vae.${vae}.f1:"
	    stdmean "$numbers9"
	done
    done
done




