(Note: this package is provided for compatibility with public tutorials.
 It is not maintained anymore.)

# GM: Graphical Models for Torch/LuaJIT

This package provides standard functions to create (arbitrary) 
undirected graphical models, using adjacency matrices.

A graph is described by an adjacency matrix, node potentials
and edge potentials. 

## Install 

``` sh
$ git clone ...
$ [sudo] luarocks make
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
