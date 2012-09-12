#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/gm.c"
#else

static inline THTensor * torch_(Tensor)(lua_State *L, long idx, bool contiguous) {
  THTensor *t = (THTensor *)luaT_checkudata(L, idx, torch_Tensor);
  return THTensor_(newContiguous)(t);
}

static int gm_(maxproduct)(lua_State *L) {
  // get args
  THTensor *matrix = (THTensor *)luaT_checkudata(L, 1, torch_Tensor);
  THTensor *vector = (THTensor *)luaT_checkudata(L, 2, torch_Tensor);

  // dims
  long rows = matrix->size[0];
  long cols = matrix->size[1];

  // alloc output
  THTensor *result = THTensor_(newWithSize1d)(rows);
  THTensor_(zero)(result);

  // get contiguous tensors
  matrix = THTensor_(newContiguous)(matrix);
  vector = THTensor_(newContiguous)(vector);

  // raw pointers
  real *matrix_d = THTensor_(data)(matrix);
  real *vector_d = THTensor_(data)(vector);
  real *result_d = THTensor_(data)(result);

  // matrix vector max product
  for (long i = 0; i < rows; i++) {
    for (long j = 0; j < cols; j++) {
      real product = matrix_d[i*cols + j] * vector_d[j];
      if (product > result_d[i]) {
        result_d[i] = product;
      }
    }
  }

  // clean up
  THTensor_(free)(matrix);
  THTensor_(free)(vector);

  // return result
  luaT_pushudata(L, result, torch_Tensor);
  return 1;
}

static int gm_(getPotentialForConfig)(lua_State *L) {
  // args
  THTensor *np = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 1, torch_Tensor));
  THTensor *ep = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 2, torch_Tensor));
  THTensor *ee = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 3, torch_Tensor));
  THTensor *yy = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 4, torch_Tensor));

  // dims
  long nNodes = np->size[0];
  long nEdges = ep->size[0];

  // raw pointers
  real *nodePot = THTensor_(data)(np);
  real *edgePot = THTensor_(data)(ep);
  real *edgeEnds = THTensor_(data)(ee);
  real *Y = THTensor_(data)(yy);

  // potential
  real pot = 1;

  // node potentials
  for (long n = 0; n < nNodes; n++) {
    pot *= nodePot[n*np->stride[0]+(long)(Y[n]-1)];
  }

  // edge potentials
  for (long e = 0; e < nEdges; e++) {
    long n1 = edgeEnds[e*2+0]-1;
    long n2 = edgeEnds[e*2+1]-1;
    pot *= edgePot[e*ep->stride[0]+(long)(Y[n1]-1)*ep->stride[1]+(long)(Y[n2]-1)];
  }

  // cleanup
  THTensor_(free)(np);
  THTensor_(free)(ep);
  THTensor_(free)(ee);
  THTensor_(free)(yy);

  // return potential
  lua_pushnumber(L,pot);
  return 1;
}

static int gm_(getLogPotentialForConfig)(lua_State *L) {
  // args
  THTensor *np = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 1, torch_Tensor));
  THTensor *ep = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 2, torch_Tensor));
  THTensor *ee = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 3, torch_Tensor));
  THTensor *yy = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 4, torch_Tensor));

  // dims
  long nNodes = np->size[0];
  long nEdges = ep->size[0];

  // raw pointers
  real *nodePot = THTensor_(data)(np);
  real *edgePot = THTensor_(data)(ep);
  real *edgeEnds = THTensor_(data)(ee);
  real *Y = THTensor_(data)(yy);

  // potential
  accreal logpot = 0;

  // node potentials
  for (long n = 0; n < nNodes; n++) {
    logpot += log(nodePot[n*np->stride[0]+(long)(Y[n]-1)]);
  }

  // edge potentials
  for (long e = 0; e < nEdges; e++) {
    long n1 = edgeEnds[e*2+0]-1;
    long n2 = edgeEnds[e*2+1]-1;
    logpot += log(edgePot[e*ep->stride[0]+(long)(Y[n1]-1)*ep->stride[1]+(long)(Y[n2]-1)]);
  }

  // cleanup
  THTensor_(free)(np);
  THTensor_(free)(ep);
  THTensor_(free)(ee);
  THTensor_(free)(yy);

  // return potential
  lua_pushnumber(L,logpot);
  return 1;
}

static const struct luaL_Reg gm_(methods__) [] = {
  {"maxproduct", gm_(maxproduct)},
  {"getPotentialForConfig", gm_(getPotentialForConfig)},
  {"getLogPotentialForConfig", gm_(getLogPotentialForConfig)},
  {NULL, NULL}
};

static void gm_(Init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, gm_(methods__), "gm");
  lua_pop(L,1);
}

#endif
