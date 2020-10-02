# Learning to Recombine Data for Compositional Generalization

![Recombination Model](./recomb.png "Recomb Network")

## Dependencies
- **OS**: Linux or macOS
- **Language**: Julia 1.2.0 (if not available, the setup script will automatically install a local copy.)
- **Hardware**: NVIDIA GPU (that supports CUDA and cuDNN), with network connection, 3GB disk space including Julia installation. (we only tested with 32GB V100s)
- **Libraries**: CUDA Runtime Library and cuDNN Developer Toolkits (tested with CUDA: 10.1.105_418.39 and cuDNN: 7.5.0)
  - If you don't have them, you might get a warning about GPU functionality which means you are not able to run the code with a GPU. If this is the case, _follow the [instructions](https://stackoverflow.com/a/47503155)_ by using the below download links selecting the abovementioned versions. This is a local installation and will not affect your system.
  ```
  CUDA: https://developer.nvidia.com/cuda-10.1-download-archive-base
  CUDNN: https://developer.nvidia.com/rdp/cudnn-archive
  ```
  Do remember to add the below commands to your `.bashrc` replacing `$CUDAPATH` with your installation path.
  ```SHELL
  export PATH=$CUDAPATH/bin:$PATH
  export LD_LIBRARY_PATH=$CUDAPATH/lib64:$LD_LIBRARY_PATH
  ```
- **Optional**:
  - Jupyter Notebook with Python 3 (Used for analysis of results.)


Note that since this codebase is for reproducibility purposes you might require specific versions of the dependencies as described above. Additionally, we find AWS AMI: Knet-1.3.0 (ami-0469b38d93e8ab9da) to be compatible with the requirements here for convenience.

## Requirements

To install requirements:
```SHELL
   sh setup.sh
```
`setup.sh` performs the following steps interactively:
1. If Julia 1.2.0 is not available, it downloads and installs it locally.
2. Installs the exact versions of the required Julia packages to a local environment.
3. Downloads raw dataset files of `SCAN` and `SIGMORPHON 2018` to the [data/](data/) folder.
4. Downloads `SCAN` and `SIGMORPHON 2018` preprocessed neighborhood files from server.

  **Optional**:

5. Downloads pre-trained generative models for `SCAN` and `SIGMORPHON 2018` along with generated samples to the the checkpoints folders.

Note that if there are issues with any of the steps 1 through 4, the experiments might fail.


## Training

To verify the results presented in the paper, you may run the scripts to train models and see the evaluations. During training logs will be created at [checkpoints/](checkpoints/) folder.

All experiment scripts are found at [exps/](exps/)

For example, to run the recomb-2 model with VAE on the `jump` split of `SCAN`, use

```SHELL
cd exps
export RECOMB_CHECKPOINT_DIR=checkpoints/
./jump_vae_2proto.sh
```
which runs the entire pipeline (train generative model -> generate samples -> train conditional model with augmented data). The logs and saved models can be found under `RECOMB_CHECKPOINT_DIR` folder.

> 📋 Note that the experiments are tested on NVIDIA 32GB V100 Volta GPUs. For some models GPU requirements might be high. Assuming the same setup, each experiment should run less than one hour.

## Evaluation

After running an experiment, evaluation results can be found under [checkpoints/](checkpoints/) at the end of the files named `*condconfig`. After running multiple experiments, we provide a convenience script which collates the results in shell:

```SHELL
sh collect.sh path/to/checkpoints
```

Moreover, after running all experiments, one can refer to `analyze_results.ipynb` Jupyter Notebook to obtain the figures and tables provided in the paper.


## Pre-trained Models

`setup.sh` optionally downloads the pre-trained models. See **Requirements** section.


## Trouble Shooting

If you get warnings about the GPU such examples in below, you should refer to requirements section
```
Warning: Knet cannot use the GPU: ErrorException("curandCreateGenerator: 203: Initialization of CUDA failed"
[ Info: CUDAdrv.jl failed to initialize, GPU functionality unavailable (set JULIA_CUDA_SILENT or JULIA_CUDA_VERBOSE to silence or expand this message)
```
If you get `ERROR: LoadError: cudnnRNNBackwardData: 8: CUDNN_STATUS_EXECUTION_FAILED`, you need more GPU memory.
