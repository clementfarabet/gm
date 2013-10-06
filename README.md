# gm: graphical models
## inference, decoding, parameter estimation

This package provides standard functions to create (arbitrary) 
undirected graphical models, using adjacency matrices.

A graph is described by an adjacency matrix, node potentials
and edge potentials. Three common tasks are implemented:

* Decoding: finding the joint configuration of the variables with the highest probability.
* Inference: computing the normalization constant Z, as well as the probabilities of each variable taking each state.
* Training (or parameter estimation): the task of computing the potentials that maximize the likelihood of a set of data.

Note: training is only implemented for CRF objectives (conditional random fields), not MRFs.

Note 2: this code is heavily based on 
[UGM](http://www.di.ens.fr/~mschmidt/Software/UGM.html), 
from Mark Schmidt.

## Install 

``` sh
$ [sudo] torch-rocks install gm
```

## Use

First run torch, and load gm:

``` sh
$ torch
``` 

``` lua
> require 'gm'
```

Once loaded, run the examples:

``` lua
> gm.examples.simple()
> gm.examples.trainCRF()
```
