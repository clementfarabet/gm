# GM: Graphical Models for Torch/LuaJIT

This package provides standard functions to create (arbitrary) 
undirected graphical models, using adjacency matrices.

A graph is described by an adjacency matrix, node potentials
and edge potentials. 

## Inference, Decoding, Sampling, Parameter estimation

Four common tasks are implemented:

* Decoding: finding the joint configuration of the variables with the highest 
joint probability;

* Inference: computing the normalization constant Z (partition function), as 
well as the probabilities of each variable taking each possible state (the
marginal probabilities);

* Sampling: given a model, sampling generates likely configurations for 
each node of the graph;

* Training (or parameter estimation): the task of computing the potentials 
that maximize the likelihood of a set of data (MAP estimation).

Note 1: parameter estimation is implemented for CRF and MRF objectives.

Note 2: this code is heavily based on 
[UGM](http://www.di.ens.fr/~mschmidt/Software/UGM.html), 
from Mark Schmidt.

## Install 

``` sh
$ [sudo] luarocks install gm
```

## Use

First run torch, and load gm:

``` sh
$ th
``` 

``` lua
> require 'gm'
```

Once loaded, see and run the examples:

``` lua
> gm.examples.simple()
> gm.examples.trainMRF()
> gm.examples.trainCRF()
```
