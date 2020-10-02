#!/bin/bash
cecho() {
  local code="\033["
  case "$1" in
    black  | bk) color="${code}0;30m";;
    red    |  r) color="${code}1;31m";;
    green  |  g) color="${code}1;32m";;
    yellow |  y) color="${code}1;33m";;
    blue   |  b) color="${code}1;34m";;
    purple |  p) color="${code}1;35m";;
    cyan   |  c) color="${code}1;36m";;
    gray   | gr) color="${code}0;37m";;
    *) local text="$1"
  esac
  [ -z "$text" ] && local text="$color$2${code}0m"
  echo "$text"
}

jlversion=`julia -v | grep -Eo '[0-9]+\.[0-9]+\.[0-9]'`
if [ "$jlversion" != "1.2.0" ]; then
    cecho y "Your julia version ${jlversion} is not compatible with requirements,\
Do you want me to install another julia with version 1.2.0, if yes, please give a path, if no, please type no [path/no]: "
    read answer
    if [ "$answer" = "no" ]; then
	cecho y "quits..."
	exit 1
    else
	path=$answer
	cecho y "julia 1.2.0 will be installed to ${path}/julia-1.2.0/; Julia also creates a package directory at ~/.julia, if there is no"
	cecho y "Do you want to continue? press any button, or kill: "
	read answer
	sh ./install-julia.sh 1.2.0 ${path}/julia-1.2.0
	jlpath=${path}/julia-1.2.0/bin/julia
	cecho y $jlpath
	if [ -f $jlpath ]; then
	    cecho y "Julia is succesfully installed to ${path}/julia-1.2.0/; Type following on your bash before running experiments. "
	    binpath=$(realpath ${path}/julia-1.2.0/bin)
	    cecho y "export PATH=$binpath:\$PATH"
	else
	    cecho y "Julia installation is broken"
	fi
    fi
else
    cecho y "You've correct version of Julia: ${jlversion}"
fi
cecho y "Checking out required Julia packages..."
binpath=$(realpath ${path}/julia-1.2.0/bin)
export PATH=$binpath:$PATH
julia --project -e 'using Pkg; Pkg.instantiate();'

cecho y "Clonning raw dataset files from the original sources..."
#FIXME: Check if raw files exists
julia --project -L src/parser.jl -e 'download(SIGDataSet); download(SCANDataSet)'

cecho y "Do you want to download checkpoints(2MB)? \
Strongly recommended for replication and to see the experimental results.\
Type yes or no: "
read answer
if [ "$answer" = "yes" ]; then
    if [ -d checkpoints.bak/ ]; then
	cecho y "checkpoints exists, skipping the download..."
    else
	cecho y "Downloading checkpoint logs 2MB"
	curl --url https://recomb.s3.us-east-2.amazonaws.com/checkpoints.tar.gz --output ./checkpoints.tar.gz
	tar -xvf checkpoints.tar.gz
    fi
else
    cecho y "skipping checkpoints"
fi

mkdir -p checkpoints/SCANDataSet/logs
if [ -f data/SCANDataSet/jump.jld2 ] && [ -f data/SCANDataSet/around_right.jld2 ]; then
    cecho y "SCAN preprocessed data files exists, skipping the download..."
else
    cecho y "Downloading SCAN preprocessed files 80MB"
    mkdir -p data/SCANDataSet
    curl --url https://recomb.s3.us-east-2.amazonaws.com/jump.jld2 --output data/SCANDataSet/jump.jld2
    curl --url https://recomb.s3.us-east-2.amazonaws.com/around_right.jld2 --output data/SCANDataSet/around_right.jld2
fi

mkdir -p checkpoints/SIGDataSet/
if [ -d data/SIGDataSet/spanish ]; then
    cecho y "Morphology preprocessed data files exists, skipping the download..."
else
    cecho y "Downloading morphology preprocessed files 1.8MB"
    curl --url https://recomb.s3.us-east-2.amazonaws.com/morph_data.tar.gz --output ./morph_data.tar.gz
    tar -xvf morph_data.tar.gz
    rm morph_data.tar.gz
fi

for lang in turkish spanish swahili; do
    mkdir -p checkpoints/SIGDataSet/${lang}/logs
    #python data/prep.py data/SIGDataSet ${lang}
done


cecho y "Do you want to download pretrained SCAN models (138MB)? Type yes or no: "
read answer
if [ "$answer" = "yes" ]; then
    curl --url https://recomb.s3.us-east-2.amazonaws.com/scan_pretrained.tar.gz  --output ./scan_pretrained.tar.gz
    tar -xvf ./scan_pretrained.tar.gz
    rm ./scan_pretrained.tar.gz
else
cecho y "skipping pretrained SCAN models"
fi

cecho y "Do you want to download pretrained morphology models (904MB)? Type yes or no: "
read answer
if [ "$answer" = "yes" ]; then
    curl --url https://recomb.s3.us-east-2.amazonaws.com/morph_pretrained.tar.gz  --output ./morph_pretrained.tar.gz
    tar -xvf ./morph_pretrained.tar.gz
    rm ./morph_pretrained.tar.gz
else
cecho y "skipping pretrained SCAN models"
fi

cecho y "If you didn't get any error till this point, you've succesfully setup this repo for the experiments."
cecho y "If you installed a local Julia during the setup, add Julia to your path before running anything. "
cecho y "export PATH=$binpath:\$PATH"
#cecho y -n "\"\$pip install -r requirement.txt\" will be called. Do you want to continue? press any button, or kill"
#read answer
#pip install -r requirements.txt
#cecho y -n "It will preprocess  data files for morphology. Could take 30 minutes. Do you want to continue? press any button, or kill"
#read answer
#for lang in turkish spanish swahili; do
#  mkdir checkpoints/SIGDataSet/${lang}
#  mkdir checkpoints/SIGDataSet/${lang}/logs
#  python data/prep.py data/SIGDataSet ${lang}
#done
