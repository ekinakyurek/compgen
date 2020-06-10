#!/bin/sh
cd ../
# export CUDA_VISIBLE_DEVICES=3
export RECOMB_TASK=SCAN
VAE=false
for nproto in 2; do
    for seed in 0 1 2 3 4; do
	julia --project evalgen.jl checkpoints/SCANDataSet/${nproto}proto.vae.${VAE}.around_right.seed.$seed.model.jld2 \
	      > checkpoints/SCANDataSet/logs/${nproto}proto.vae.${VAE}.around_right.seed.$seed.eval.log \
	      2> checkpoints/SCANDataSet/logs/${nproto}proto.vae.${VAE}.around_right.seed.$seed.eval.err
    done
done
