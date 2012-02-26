#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/gm.c"
#else

static int gm_(maxproduct)(lua_State *L) {
  // get args
  THTensor *matrix = (THTensor *)luaT_checkudata(L, 1, torch_(Tensor_id));
  THTensor *vector = (THTensor *)luaT_checkudata(L, 2, torch_(Tensor_id));

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
  luaT_pushudata(L, result, torch_(Tensor_id));
  return 1;
}

static const struct luaL_Reg gm_(methods__) [] = {
  {"maxproduct", gm_(maxproduct)},
  {NULL, NULL}
};

static void gm_(Init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, gm_(methods__), "gm");
  lua_pop(L,1);
}

#endif
