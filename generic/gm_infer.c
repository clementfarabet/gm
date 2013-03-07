#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/gm_infer.c"
#else

#ifdef _OPENMP
#include "omp.h"
#endif

static int gm_infer_(bpInitMessages)(lua_State *L) {
  // get args
  THTensor *ee = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 1, torch_Tensor));
  THTensor *ns = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 2, torch_Tensor));
  THTensor *msg = (THTensor *)luaT_checkudata(L, 3, torch_Tensor);

  // dims
  long nEdges = ee->size[0];

  // raw pointers
  real *edgeEnds = THTensor_(data)(ee);
  real *nStates = THTensor_(data)(ns);
  real *message = THTensor_(data)(msg);

  // propagate state normalizations
#pragma omp parallel for
  for (long e = 0; e < nEdges; e++) {
    // get edge of interest, and its nodes
    long n1 = edgeEnds[e*2+0]-1;
    long n2 = edgeEnds[e*2+1]-1;

    // propagate
    for (long s = 0; s < nStates[n2]; s++) {
      message[e*msg->stride[0]+s*msg->stride[1]] = 1/nStates[n2]; //  n1 ==> n2
    }
    for (long s = 0; s < nStates[n1]; s++) {
      message[(e+nEdges)*msg->stride[0]+s*msg->stride[1]] = 1/nStates[n1]; //  n2 ==> n1
    }
  }

  // clean up
  THTensor_(free)(ee);
  THTensor_(free)(ns);
  return 0;
}

static int gm_infer_(bpComputeMessages)(lua_State *L) {
  // get args
  THTensor *np = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 1, torch_Tensor));
  THTensor *ep = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 2, torch_Tensor));
  THTensor *ee = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 3, torch_Tensor));
  THTensor *ns = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 4, torch_Tensor));
  THTensor *EE = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 5, torch_Tensor));
  THTensor *VV = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 6, torch_Tensor));
  THTensor *msg = (THTensor *)luaT_checkudata(L, 7, torch_Tensor);
  bool maxprod = lua_toboolean(L, 8);

  // dims
  long nNodes = np->size[0];
  long nEdges = ep->size[0];

  // raw pointers
  real *nStates = THTensor_(data)(ns);
  real *edgeEnds = THTensor_(data)(ee);
  real *E = THTensor_(data)(EE);
  real *V = THTensor_(data)(VV);

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
      if (sum == 0) THError("numeric precision too low, can't compute messages");
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
  THTensor *np = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 1, torch_Tensor));
  THTensor *nb = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 2, torch_Tensor));
  THTensor *ee = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 3, torch_Tensor));
  THTensor *ns = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 4, torch_Tensor));
  THTensor *EE = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 5, torch_Tensor));
  THTensor *VV = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 6, torch_Tensor));
  THTensor *pd = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 7, torch_Tensor));
  THTensor *msg = (THTensor *)luaT_checkudata(L, 8, torch_Tensor);

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

      // compute component-wise product
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
    if (sum == 0) THError("numeric precision too low, can't compute node beliefs");
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
  THTensor *ep = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 1, torch_Tensor));
  THTensor *eb = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 2, torch_Tensor));
  THTensor *nb = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 3, torch_Tensor));
  THTensor *ee = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 4, torch_Tensor));
  THTensor *ns = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 5, torch_Tensor));
  THTensor *EE = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 6, torch_Tensor));
  THTensor *VV = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 7, torch_Tensor));
  THTensor *msg = (THTensor *)luaT_checkudata(L, 8, torch_Tensor);

  // dims
  long nEdges = ep->size[0];

  // raw pointers
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
    if (sum == 0) THError("numeric precision too low, can't compute edge beliefs");
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

static int gm_infer_(bpComputeLogZ)(lua_State *L) {
  // get args
  THTensor *np = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 1, torch_Tensor));
  THTensor *ep = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 2, torch_Tensor));
  THTensor *nb = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 3, torch_Tensor));
  THTensor *eb = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 4, torch_Tensor));
  THTensor *ee = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 5, torch_Tensor));
  THTensor *ns = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 6, torch_Tensor));
  THTensor *EE = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 7, torch_Tensor));
  THTensor *VV = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 8, torch_Tensor));

  // dims
  long nNodes = np->size[0];
  long nEdges = ep->size[0];

  // raw pointers
  real *edgeEnds = THTensor_(data)(ee);
  real *nStates = THTensor_(data)(ns);
  real *V = THTensor_(data)(VV);

  // add epsilon to beliefs
  real eps = 1e-15;
  THTensor_(add)(nb, nb, eps);
  THTensor_(add)(eb, eb, eps);

  // vars
  accreal eng1 = 0;
  accreal eng2 = 0;
  accreal ent1 = 0;
  accreal ent2 = 0;
  accreal sum;
  real *tmp_d;

  // temp structures
  THTensor *nodeBel = THTensor_(new)();
  THTensor *nodePot = THTensor_(new)();
  THTensor *edgeBel = THTensor_(new)();
  THTensor *edgePot = THTensor_(new)();
  THTensor *tmp = THTensor_(new)();

  // wrt nodes
  for (long n = 0; n < nNodes; n++) {
    // find neighbors of node n (Lua: local edges = graph:getEdgesOf(n)
    long nEdgesOfNode = (long)(V[n+1]-V[n]);

    // node entropy
    THTensor_(select)(nodeBel, nb, 0, n);
    THTensor_(narrow)(nodeBel, NULL, 0, 0, nStates[n]);
    THTensor_(resizeAs)(tmp, nodeBel);
    THTensor_(log)(tmp, nodeBel);
    THTensor_(cmul)(tmp, tmp, nodeBel);
    sum = 0;
    tmp_d = THTensor_(data)(tmp);
    for (long i = 0; i < tmp->size[0]; i++) sum += tmp_d[i];
    ent1 += (nEdgesOfNode-1) * sum;

    // node energy
    THTensor_(select)(nodePot, np, 0, n);
    THTensor_(narrow)(nodePot, NULL, 0, 0, nStates[n]);
    THTensor_(resizeAs)(tmp, nodePot);
    THTensor_(log)(tmp, nodePot);
    THTensor_(cmul)(tmp, tmp, nodeBel);
    sum = 0;
    tmp_d = THTensor_(data)(tmp);
    for (long i = 0; i < tmp->size[0]; i++) sum += tmp_d[i];
    eng1 -= sum;
  }

  // wrt edges
  for (long e = 0; e < nEdges; e++) {
    // get edge of interest, and its nodes
    long n1 = edgeEnds[e*2+0]-1;
    long n2 = edgeEnds[e*2+1]-1;

    // edge entropy
    THTensor_(select)(edgeBel, eb, 0, e);
    THTensor_(narrow)(edgeBel, NULL, 0, 0, nStates[n1]);
    THTensor_(narrow)(edgeBel, NULL, 0, 0, nStates[n2]);
    THTensor_(resizeAs)(tmp, edgeBel);
    THTensor_(log)(tmp, edgeBel);
    THTensor_(cmul)(tmp, tmp, edgeBel);
    sum = 0;
    tmp_d = THTensor_(data)(tmp);
    for (long i = 0; i < edgeBel->size[0]; i++) {
      for (long j = 0; j < edgeBel->size[1]; j++) {
        sum += tmp_d[i*tmp->stride[0]+j];
      }
    }
    ent2 -= sum;

    // edge energy
    THTensor_(select)(edgePot, ep, 0, e);
    THTensor_(narrow)(edgePot, NULL, 0, 0, nStates[n1]);
    THTensor_(narrow)(edgePot, NULL, 0, 0, nStates[n2]);
    THTensor_(resizeAs)(tmp, edgePot);
    THTensor_(log)(tmp, edgePot);
    THTensor_(cmul)(tmp, tmp, edgeBel);
    sum = 0;
    tmp_d = THTensor_(data)(tmp);
    for (long i = 0; i < edgeBel->size[0]; i++) {
      for (long j = 0; j < edgeBel->size[1]; j++) {
        sum += tmp_d[i*tmp->stride[0]+j];
      }
    }
    eng2 -= sum;
  }

  // free energy
  accreal F = (eng1+eng2) - (ent1+ent2);
  accreal logZ = -F;

  // clean up
  THTensor_(free)(np);
  THTensor_(free)(ep);
  THTensor_(free)(nb);
  THTensor_(free)(eb);
  THTensor_(free)(ee);
  THTensor_(free)(ns);
  THTensor_(free)(EE);
  THTensor_(free)(VV);
  THTensor_(free)(nodeBel);
  THTensor_(free)(nodePot);
  THTensor_(free)(edgeBel);
  THTensor_(free)(edgePot);
  THTensor_(free)(tmp);

  // return logZ
  lua_pushnumber(L, logZ);
  return 1;
}

static const struct luaL_Reg gm_infer_(methods__) [] = {
  {"bpInitMessages", gm_infer_(bpInitMessages)},
  {"bpComputeMessages", gm_infer_(bpComputeMessages)},
  {"bpComputeNodeBeliefs", gm_infer_(bpComputeNodeBeliefs)},
  {"bpComputeEdgeBeliefs", gm_infer_(bpComputeEdgeBeliefs)},
  {"bpComputeLogZ", gm_infer_(bpComputeLogZ)},
  {NULL, NULL}
};

static void gm_infer_(Init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, gm_infer_(methods__), "gm");
  lua_pop(L,1);
}

#endif
