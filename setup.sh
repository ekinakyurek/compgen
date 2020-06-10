#!/bin/bash
jlversion=`julia -v | grep -Eo '[0-9]+\.[0-9]+\.[0-9]'`
if [ "$jlversion" != "1.2.0" ]; then
  echo -n "Your julia version ${jlversion} is not compatible with requirements,
           do you want me to install another julia with version 1.2.0 ? If yes, give a path, if no write no [path/no]: "
  read answer
  if [ "$answer" = "no" ]; then
    echo "exitting"
    exit 1
  else
    path=$answer
    echo "julia 1.2.0 will be installed to ${path}"
    echo -n "Do you want to continue? press any button"
    read answer
    ./install-julia.sh 1.2.0 $path
    jlpath=${path}/julia-1.2.0/bin/julia
    echo $jlpath
    if [ -f $jlpath ]; then
       echo "Julia is succesfully installed to ${path}/julia-1.2.0/"
       echo "type following on your bash before running experiments. "
       echo "export julia=${jlpath}"
    else
       echo "Julia installation is broken"
    fi
  fi
else
  echo "You've correct version of Julia"
fi
julia --project -e 'using Pkg; Pkg.instantiate();'
julia --project -L parser.jl -e 'download(SIGDataSet); download(SCANDataSet)'
#FIXME: Needs anonymity
curl --url https://people.csail.mit.edu/akyurek/recomb/jump.jld2 --output data/SCANDataSet/jump.jld2
curl --url https://people.csail.mit.edu/akyurek/recomb/around_right.jld2 --output data/SCANDataSet/around_right.jld2
mkdir checkpoints
mkdir checkpoints/SIGDataSet/
for lang in turkish spanish swahili; do
  mkdir checkpoints/SIGDataSet/${lang}
  mkdir checkpoints/SIGDataSet/${lang}/logs
  python data/prep.py data/SIGDataSet ${lang}
done
