# Recombination Networks

The code provides the models described in Learning Data Recombination and Mutation forCompositional Generalization paper.

## Dependencies
  - Linux or MACOS
  - Julia 1.2 (if not don't worry, the setup script will ask for a local installation)
  - CUDA Run Time Library and CUDNN Developer Toolkits (tested with cuda: 10.1.105_418.39, cudnn: 7.5.0)
        If you don't have them, you might get a warning about gpu functionality which mean you are not able to run this code with gpu.
	If this is the case, follow this [instructions](https://stackoverflow.com/a/47503155) after downloading required versions from the below links.
	        -- CUDA: https://developer.nvidia.com/cuda-10.1-download-archive-base
		-- CUDNN: https://developer.nvidia.com/rdp/cudnn-archive
        In the final state, add below commands to your bashrc replacing CUDAPATH with your installation path.
	```SHELL
		export PATH=$CUDAPATH/bin:$PATH
		export LD_LIBRARY_PATH=$CUDAPATH/lib64:$LD_LIBRARY_PATH
	```
		

  - Python 3
  - Network connection


## Setup

```SHELL
   sh setup.sh
```

## Experiments

To verify the results presented in the paper, you may run the scripts to train models and see the evaluations. During training logs will be created at [checkpoints/](checkpoints/) folder.

All the experiments can be found at [exps/](exps/)

**Note**: We only tested the experiments on a Nvidia Volta GPU, for some models GPU requirements might be high.
