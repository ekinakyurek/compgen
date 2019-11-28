for i in 0 1 2 3 4 5 6 7
do
  CUDA_VISIBLE_DEVICES=$i julia runexperiments.jl $i > $i.txt 2>&1 &
done
