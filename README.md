# Recombination Networks

The code provides the models described in Learning Data Recombination and Mutation forCompositional Generalization paper.

## Dependencies
  - Linux or MACOS
  - Julia 1.2 (if not don't worry, the setup script will ask for a local installation)
  - CUDA and CUDNN Developer Toolkits
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
