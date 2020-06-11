#!/bin/bash
jlversion=`julia -v | grep -Eo '[0-9]+\.[0-9]+\.[0-9]'`
if [ "$jlversion" != "1.2.0" ]; then
  echo -n "Your julia version ${jlversion} is not compatible with requirements,
           do you want me to install another julia with version 1.2.0, if yes, please give a path, if no, please type no [path/no]: "
  read answer
  if [ "$answer" = "no" ]; then
    echo "quits..."
    exit 1
  else
    path=$answer
    echo "julia 1.2.0 will be installed to ${path}/julia-1.2.0/"
    echo -n "Do you want to continue? press any button, or kill: "
    read answer
    sh ./install-julia.sh 1.2.0 $path
    jlpath=${path}/julia-1.2.0/bin/julia
    echo $jlpath
    if [ -f $jlpath ]; then
       echo "Julia is succesfully installed to ${path}/julia-1.2.0/"
       echo "type following on your bash before running experiments. "
       binpath=$(realpath ${path}/julia-1.2.0/bin)
       echo "export PATH=$binpath:\$PATH"
    else
       echo "Julia installation is broken"
    fi
  fi
else
  echo "You've correct version of Julia: ${jlversion}"
fi
echo "Checking out required Julia packages..."
binpath=$(realpath ${path}/julia-1.2.0/bin)
export PATH=$binpath:$PATH
julia --project -e 'using Pkg; Pkg.instantiate();'

echo "Clonning raw dataset files from the original sources..."
#FIXME: Check if raw files exists
julia --project -L src/parser.jl -e 'download(SIGDataSet); download(SCANDataSet)'

mkdir -p checkpoints/SCANDataSet/logs
if [ -f data/SCANDataSet/jump.jld2 ] && [ -f data/SCANDataSet/around_right.jld2 ]; then
    echo "SCAN preprocessed data files exists, skipping the download..."
else 
    echo "Downloading SCAN preprocessed files 80MB"
    curl --url https://recomb.s3.us-east-2.amazonaws.com/jump.jld2 --output data/SCANDataSet/jump.jld2
    curl --url https://recomb.s3.us-east-2.amazonaws.com/around_right.jld2 --output data/SCANDataSet/around_right.jld2
fi

mkdir -p checkpoints/SIGDataSet/
if [ -d data/SIGDataSet/spanish ]; then
    echo "Morphology preprocessed data files exists, skipping the download..."
else
    echo "Downloading morphology preprocessed files 1.8MB"
    curl --url https://recomb.s3.us-east-2.amazonaws.com/morph_data.tar.gz --output ./morph_data.tar.gz
    tar -xvf morph_data.tar.gz
    rm morph_data.tar.gz
fi

for lang in turkish spanish swahili; do
  mkdir -p checkpoints/SIGDataSet/${lang}/logs
  #python data/prep.py data/SIGDataSet ${lang}
done


echo -n "Do you want to download pretrained SCAN models (138MB)? Type yes or no: "
read answer
if [ "$answer" = "yes" ]; then
    curl --url https://recomb.s3.us-east-2.amazonaws.com/scan_pretrained.tar.gz  --output ./scan_pretrained.tar.gz
    tar -xvf ./scan_pretrained.tar.gz
    rm ./scan_pretrained.tar.gz
else
    echo "skipping pretrained SCAN models"
fi

echo -n "Do you want to download pretrained morphology models (904MB)? Type yes or no: "
read answer
if [ "$answer" = "yes" ]; then
    curl --url https://recomb.s3.us-east-2.amazonaws.com/morph_pretrained.tar.gz  --output ./morph_pretrained.tar.gz
    tar -xvf ./morph_pretrained.tar.gz
    rm ./morph_pretrained.tar.gz
else
    echo "skipping pretrained SCAN models"
fi

echo "If you didn't get any error till this point, you've succesfully setup this repo for the experiments."
#echo -n "\"\$pip install -r requirement.txt\" will be called. Do you want to continue? press any button, or kill"
#read answer
#pip install -r requirements.txt
#echo -n "It will preprocess  data files for morphology. Could take 30 minutes. Do you want to continue? press any button, or kill"
#read answer
#for lang in turkish spanish swahili; do
#  mkdir checkpoints/SIGDataSet/${lang}
#  mkdir checkpoints/SIGDataSet/${lang}/logs
#  python data/prep.py data/SIGDataSet ${lang}
#done
