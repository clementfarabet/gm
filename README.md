# gm: a package to create and use graphical models

This package provides standard functions to create (arbitrary) 
undirected graphical models, and decode/infer their optimal 
(maximum potential) state.

Note: this code is heavily based on 
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
> gm.examples.trainMRF()
> gm.examples.trainCRF()
```
