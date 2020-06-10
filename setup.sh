#!/bin/bash
jlversion=`julia -v | grep -Eo '[0-9]+\.[0-9]+\.[0-9]'`
if [ "$jlversion" != "1.2.0" ]; then
  echo -n "Your julia version ${jlversion} is not compatible with requirements,
           do you want me to install another julia with version 1.2.0 ? If yes, give a path, if no write n [path/NO]: "
  read answer
  if [ "$answer" = "n" ]; then
    echo "exitting"
    exit 1
  else
    path=$answer
    echo "julia 1.2.0 will be installed to ${path}"
    # ./install-julia.sh 1.2.0 $path
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
#julia --project -e 'using Pkg; Pkg.instantiate();'
#julia --project -L parser.jl -e 'download(SIGDataSet); download(SCANDataSet)'
#curl --url https://people.csail.mit.edu/akyurek/recomb/jump.jld2 --output data/SCANDataSet/jump.jld2
#curl --url https://people.csail.mit.edu/akyurek/recomb/around_right.jld2 --output data/SCANDataSet/around_right.jld2
# python data/prep.py data/SIGDataSet turkish
# mkdir checkpoints/SIGDataSet/turkish
# mkdir checkpoints/SIGDataSet/turkish/logs
# cp checkpoints/SIGDataSet/collect.sh checkpoints/SIGDataSet/turkish/
# python data/prep.py data/SIGDataSet spanish
# mkdir checkpoints/SIGDataSet/spanish
# mkdir checkpoints/SIGDataSet/spanish/logs/
# cp checkpoints/SIGDataSet/collect.sh checkpoints/SIGDataSet/spanish/
