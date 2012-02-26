#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/gm_infer.c"
#else

static int gm_infer_(bpComputeMessages)(lua_State *L) {
  // get args
  const void *id = torch_(Tensor_id);
  THTensor *np = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 1, id));
  THTensor *ep = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 2, id));
  THTensor *ee = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 3, id));
  THTensor *ns = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 4, id));
  THTensor *EE = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 5, id));
  THTensor *VV = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 6, id));
  THTensor *msg = (THTensor *)luaT_checkudata(L, 7, id);
  bool maxprod = lua_toboolean(L, 8);

  // dims
  long nNodes = np->size[0];
  long nEdges = ep->size[0];

  // raw pointers
  real *nodePot = THTensor_(data)(np);
  real *edgePot = THTensor_(data)(ep);
  real *nStates = THTensor_(data)(ns);
  real *edgeEnds = THTensor_(data)(ee);
  real *E = THTensor_(data)(EE);
  real *V = THTensor_(data)(VV);
  real *message = THTensor_(data)(msg);

  // temp structures
  THTensor *pot_ij_src = THTensor_(new)();
  THTensor *pot_ij = THTensor_(new)();
  THTensor *prod_src = THTensor_(new)();
  THTensor *prod = THTensor_(new)();
  THTensor *messg = THTensor_(new)();

  // belief propagation = message passing
  for (long n = 0; n < nNodes; n++) {
    // find neighbors of node n (Lua: local edges = graph:getEdgesOf(n)
    real *edges = E + ((long)(V[n])-1);
    long nEdgesOfNode = (long)(V[n+1]-V[n]);

    // send a message to each neighbor of node n
    for (long k = 0; k < nEdgesOfNode; k++) {
      // get edge of interest, and its nodes
      long e = edges[k]-1;
      long n1 = edgeEnds[e*2+0]-1;
      long n2 = edgeEnds[e*2+1]-1;

      // get joint potential
      THTensor_(select)(pot_ij_src, ep, 0, e);
      THTensor_(narrow)(pot_ij_src, NULL, 0, 0, nStates[n1]);
      THTensor_(narrow)(pot_ij_src, NULL, 1, 0, nStates[n2]);
      THTensor_(resizeAs)(pot_ij, pot_ij_src);
      if (n == n1) {
        THTensor_(transpose)(pot_ij, pot_ij_src, 0, 1);
      } else {
        THTensor_(copy)(pot_ij, pot_ij_src);
      }

      // compute product of all incoming messages except j
      THTensor_(select)(prod_src, np, 0, n);
      THTensor_(narrow)(prod_src, NULL, 0, 0, nStates[n]);
      THTensor_(resizeAs)(prod, prod_src);
      THTensor_(copy)(prod, prod_src);
      for (long kk = 0; kk < nEdgesOfNode; kk++) {
        long ee = edges[kk]-1;
        long nn1 = edgeEnds[ee*2+0]-1;
        long nn2 = edgeEnds[ee*2+1]-1;
        if (ee != e) {
          if (n == nn1) {
            THTensor_(select)(messg, msg, 0, ee+nEdges);
            THTensor_(narrow)(messg, NULL, 0, 0, nStates[n]);
          } else {
            THTensor_(select)(messg, msg, 0, ee);
            THTensor_(narrow)(messg, NULL, 0, 0, nStates[n]);
          }
          THTensor_(cmul)(prod, prod, messg);
        }
      }

      // compute new message
      if (n == n1) {
        THTensor_(select)(messg, msg, 0, e);
        THTensor_(narrow)(messg, NULL, 0, 0, nStates[n2]);
      } else {
        THTensor_(select)(messg, msg, 0, e+nEdges);
        THTensor_(narrow)(messg, NULL, 0, 0, nStates[n1]);
      }

      // either do a max or products, or a sum of products
      THTensor_(zero)(messg);
      if (maxprod) {
        // do max product on raw pointers
        real *matrix_d = THTensor_(data)(pot_ij);
        real *vector_d = THTensor_(data)(prod);
        real *result_d = THTensor_(data)(messg);
        long rows = pot_ij->size[0];
        long cols = pot_ij->size[1];

        // matrix vector max product
        for (long i = 0; i < rows; i++) {
          for (long j = 0; j < cols; j++) {
            real product = matrix_d[i*cols + j] * vector_d[j];
            if (product > result_d[i]) {
              result_d[i] = product;
            }
          }
        }
      } else {
        // do sum of products = regular matrix*vector
        THTensor_(addmv)(messg, 1, messg, 1, pot_ij, prod);
      }

      // normalize message
      accreal sum = 0;
      real *messg_d = THTensor_(data)(messg);
      for (long i = 0; i < messg->size[0]; i++) sum += messg_d[i];
      THTensor_(div)(messg, messg, sum);
    }
  }

  // clean up
  THTensor_(free)(np);
  THTensor_(free)(ep);
  THTensor_(free)(ee);
  THTensor_(free)(EE);
  THTensor_(free)(VV);
  THTensor_(free)(ns);
  THTensor_(free)(pot_ij_src);
  THTensor_(free)(pot_ij);
  THTensor_(free)(prod_src);
  THTensor_(free)(prod);
  THTensor_(free)(messg);
  return 0;
}

static int gm_infer_(bpComputeNodeBeliefs)(lua_State *L) {
  // get args
  const void *id = torch_(Tensor_id);
  THTensor *np = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 1, id));
  THTensor *nb = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 2, id));
  THTensor *ee = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 3, id));
  THTensor *ns = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 4, id));
  THTensor *EE = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 5, id));
  THTensor *VV = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 6, id));
  THTensor *pd = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 7, id));
  THTensor *msg = (THTensor *)luaT_checkudata(L, 8, id);

  // dims
  long nNodes = np->size[0];
  long nEdges = ee->size[0];

  // raw pointers
  real *E = THTensor_(data)(EE);
  real *V = THTensor_(data)(VV);
  real *edgeEnds = THTensor_(data)(ee);
  real *nStates = THTensor_(data)(ns);

  // temp structures
  THTensor *prod = THTensor_(new)();
  THTensor *messg = THTensor_(new)();
  THTensor *nodeBel = THTensor_(new)();

  // init products
  THTensor_(copy)(pd, np);

  // compute node beliefs
  for (long n = 0; n < nNodes; n++) {
    // find neighbors of node n (Lua: local edges = graph:getEdgesOf(n)
    real *edges = E + ((long)(V[n])-1);
    long nEdgesOfNode = (long)(V[n+1]-V[n]);

    // get potentials
    THTensor_(select)(prod, pd, 0, n);
    THTensor_(narrow)(prod, NULL, 0, 0, nStates[n]);

    // send a message to each neighbor of node n
    for (long k = 0; k < nEdgesOfNode; k++) {
      // get edge of interest, and its nodes
      long e = edges[k]-1;
      long n1 = edgeEnds[e*2+0]-1;
      long n2 = edgeEnds[e*2+1]-1;

      // 
      if (n == n1) {
        THTensor_(select)(messg, msg, 0, e+nEdges);
        THTensor_(narrow)(messg, NULL, 0, 0, nStates[n]);
      } else {
        THTensor_(select)(messg, msg, 0, e);
        THTensor_(narrow)(messg, NULL, 0, 0, nStates[n]);
      }
      THTensor_(cmul)(prod, prod, messg);
    }

    // update node belief
    THTensor_(select)(nodeBel, nb, 0, n);
    THTensor_(narrow)(nodeBel, NULL, 0, 0, nStates[n]);
    THTensor_(copy)(nodeBel, prod);
    accreal sum = 0;
    real *prod_d = THTensor_(data)(prod);
    for (long i = 0; i < prod->size[0]; i++) sum += prod_d[i];
    THTensor_(div)(nodeBel, nodeBel, sum);
  }

  // clean up
  THTensor_(free)(np);
  THTensor_(free)(nb);
  THTensor_(free)(ee);
  THTensor_(free)(EE);
  THTensor_(free)(VV);
  THTensor_(free)(ns);
  THTensor_(free)(pd);
  THTensor_(free)(nodeBel);
  THTensor_(free)(prod);
  THTensor_(free)(messg);
  return 0;
}

static int gm_infer_(bpComputeEdgeBeliefs)(lua_State *L) {
  // get args
  const void *id = torch_(Tensor_id);
  THTensor *ep = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 1, id));
  THTensor *eb = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 2, id));
  THTensor *nb = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 3, id));
  THTensor *ee = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 4, id));
  THTensor *ns = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 5, id));
  THTensor *EE = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 6, id));
  THTensor *VV = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 7, id));
  THTensor *msg = (THTensor *)luaT_checkudata(L, 8, id);

  // dims
  long nEdges = ep->size[0];

  // raw pointers
  real *E = THTensor_(data)(EE);
  real *V = THTensor_(data)(VV);
  real *edgeEnds = THTensor_(data)(ee);
  real *nStates = THTensor_(data)(ns);

  // temp structures
  THTensor *tmp1 = THTensor_(new)();
  THTensor *tmp2 = THTensor_(new)();
  THTensor *belN1 = THTensor_(new)();
  THTensor *belN2 = THTensor_(new)();
  THTensor *b1 = THTensor_(new)();
  THTensor *b2 = THTensor_(new)();
  THTensor *bs = THTensor_(new)();
  THTensor *edgeBel = THTensor_(new)();
  THTensor *edgePot = THTensor_(new)();

  // compute node beliefs
  for (long e = 0; e < nEdges; e++) {
    // get edge of interest, and its nodes
    long n1 = edgeEnds[e*2+0]-1;
    long n2 = edgeEnds[e*2+1]-1;

    // get beliefs for node 1
    THTensor_(select)(tmp1, nb, 0, n1);
    THTensor_(narrow)(tmp1, NULL, 0, 0, nStates[n1]);
    THTensor_(select)(tmp2, msg, 0, e+nEdges);
    THTensor_(narrow)(tmp2, NULL, 0, 0, nStates[n1]);
    THTensor_(resizeAs)(belN1, tmp1);
    THTensor_(cdiv)(belN1, tmp1, tmp2);

    // get beliefs for node 2
    THTensor_(select)(tmp1, nb, 0, n2);
    THTensor_(narrow)(tmp1, NULL, 0, 0, nStates[n2]);
    THTensor_(select)(tmp2, msg, 0, e);
    THTensor_(narrow)(tmp2, NULL, 0, 0, nStates[n2]);
    THTensor_(resizeAs)(belN2, tmp1);
    THTensor_(cdiv)(belN2, tmp1, tmp2);

    // copy beliefs
    THTensor_(resize2d)(b1, nStates[n1], nStates[n2]);
    THTensor_(resize2d)(b2, nStates[n1], nStates[n2]);
    for (long k = 0; k < nStates[n2]; k++) {
      THTensor_(select)(bs, b1, 1, k);
      THTensor_(copy)(bs, belN1);
    }
    for (long k = 0; k < nStates[n1]; k++) {
      THTensor_(select)(bs, b2, 0, k);
      THTensor_(copy)(bs, belN2);
    }

    // compute edge beliefs
    THTensor_(select)(edgeBel, eb, 0, e);
    THTensor_(narrow)(edgeBel, NULL, 0, 0, nStates[n1]);
    THTensor_(narrow)(edgeBel, NULL, 0, 0, nStates[n2]);
    THTensor_(select)(edgePot, ep, 0, e);
    THTensor_(narrow)(edgePot, NULL, 0, 0, nStates[n1]);
    THTensor_(narrow)(edgePot, NULL, 0, 0, nStates[n2]);
    THTensor_(cmul)(edgeBel, b1, b2);
    THTensor_(cmul)(edgeBel, edgeBel, edgePot);

    // normalize
    accreal sum = 0;
    real *eb_d = THTensor_(data)(edgeBel);
    for (long i = 0; i < edgeBel->size[0]; i++) {
      for (long j = 0; j < edgeBel->size[1]; j++) {
        sum += eb_d[i*edgeBel->stride[0]+j];
      }
    }
    THTensor_(div)(edgeBel, edgeBel, sum);
  }

  // clean up
  THTensor_(free)(ep);
  THTensor_(free)(eb);
  THTensor_(free)(nb);
  THTensor_(free)(ee);
  THTensor_(free)(EE);
  THTensor_(free)(VV);
  THTensor_(free)(ns);
  THTensor_(free)(edgeBel);
  THTensor_(free)(edgePot);
  THTensor_(free)(tmp1);
  THTensor_(free)(tmp2);
  THTensor_(free)(belN1);
  THTensor_(free)(belN2);
  THTensor_(free)(b1);
  THTensor_(free)(b2);
  THTensor_(free)(bs);
  return 0;
}

static const struct luaL_Reg gm_infer_(methods__) [] = {
  {"bpComputeMessages", gm_infer_(bpComputeMessages)},
  {"bpComputeNodeBeliefs", gm_infer_(bpComputeNodeBeliefs)},
  {"bpComputeEdgeBeliefs", gm_infer_(bpComputeEdgeBeliefs)},
  {NULL, NULL}
};

static void gm_infer_(Init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, gm_infer_(methods__), "gm");
  lua_pop(L,1);
}

#endif
