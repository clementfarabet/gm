# gm: a package to create and use graphical models

This package provides standard functions to create (arbitrary) 
undirected graphical models, and decode/infer their optimal 
(maximum potential) state.

Note: this code is heavily based on 
[UGM](http://www.di.ens.fr/~mschmidt/Software/UGM.html), 
from Mark Schmidt.

## Install 

1/ Torch7 is required:

Dependencies, on Linux (Ubuntu > 9.04):

``` sh
$ apt-get install gcc g++ git libreadline5-dev cmake wget libqt4-core libqt4-gui libqt4-dev
```

Dependencies, on Mac OS (Leopard, or more), using [Homebrew](http://mxcl.github.com/homebrew/):

``` sh
$ brew install git readline cmake wget qt
```

Then on both platforms:

``` sh
$ git clone https://github.com/andresy/torch
$ cd torch
$ mkdir build; cd build
$ cmake ..
$ make
$ [sudo] make install
```

2/ Once Torch7 is available, install this package:

``` sh
$ [sudo] torch-pkg install gm
```

## Use the library

First run torch, and load gm:

``` sh
$ torch
``` 

``` lua
> require 'gm'
```

Once loaded, run the little example:

``` lua
> gm.testme()
```
