#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/gm_energies.c"
#else

#ifdef _OPENMP
#include "omp.h"
#endif

static int gm_energies_(crfGradWrtNodes)(lua_State *L) {
  // get args
  THTensor *xn = (THTensor *)luaT_checkudata(L, 1, torch_Tensor);
  THTensor *nm = (THTensor *)luaT_checkudata(L, 2, torch_Tensor);
  THTensor *ns = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 4, torch_Tensor));
  THTensor *yy = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 5, torch_Tensor));
  THTensor *nb = (THTensor *)luaT_checkudata(L, 6, torch_Tensor);
  THTensor *gd = (THTensor *)luaT_checkudata(L, 7, torch_Tensor);

  // dims
  long nNodes = nm->size[0];
  long nNodeFeatures= xn->size[0];

  // raw pointers
  real *Xnode = THTensor_(data)(xn);
  real *nodeMap = THTensor_(data)(nm);
  real *nodeBel = THTensor_(data)(nb);
  real *nStates = THTensor_(data)(ns);
  real *Y = THTensor_(data)(yy);
  real *grad = THTensor_(data)(gd);

  // compute gradients wrt nodes
  for (long n = 0; n < nNodes; n++) {
    long label = (long)Y[n]-1;
    for (long s = 0; s < nStates[n]; s++) {
      for (long f = 0; f < nNodeFeatures; f++) {
        long map = nodeMap[n*nm->stride[0]+s*nm->stride[1]+f*nm->stride[2]];
        if (map > 0) {
          real obs = (s == label) ? 1 : 0;
          grad[map-1] += Xnode[f*xn->stride[0]+n*xn->stride[1]] 
                                * (nodeBel[n*nb->stride[0]+s*nb->stride[1]] - obs);
        }
      }
    }
  }

  // clean up
  THTensor_(free)(ns);
  THTensor_(free)(yy);
  return 0;
}

static int gm_energies_(crfGradWrtEdges)(lua_State *L) {
  // get args
  THTensor *xe = (THTensor *)luaT_checkudata(L, 1, torch_Tensor);
  THTensor *em = (THTensor *)luaT_checkudata(L, 2, torch_Tensor);
  THTensor *ee = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 4, torch_Tensor));
  THTensor *ns = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 5, torch_Tensor));
  THTensor *yy = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 6, torch_Tensor));
  THTensor *eb = (THTensor *)luaT_checkudata(L, 7, torch_Tensor);
  THTensor *gd = (THTensor *)luaT_checkudata(L, 8, torch_Tensor);

  // dims
  long nEdges = em->size[0];
  long nEdgeFeatures = xe->size[0];

  // raw pointers
  real *Xedge = THTensor_(data)(xe);
  real *edgeMap = THTensor_(data)(em);
  real *edgeBel = THTensor_(data)(eb);
  real *nStates = THTensor_(data)(ns);
  real *edgeEnds = THTensor_(data)(ee);
  real *Y = THTensor_(data)(yy);

  // partial gradients
  // :
#ifdef _OPENMP
  long maxthreads = omp_get_max_threads();
#else
  long maxthreads = 1;
#endif
  THTensor **gds = (THTensor **)malloc(sizeof(void*)*maxthreads);
  real **grads = (real **)malloc(sizeof(void*)*maxthreads);
  for (int i = 0; i < maxthreads; i++) {
    gds[i] = THTensor_(newWithSize1d)(gd->size[0]);
    grads[i] = THTensor_(data)(gds[i]);
  }

  // compute gradients wrt edges
#pragma omp parallel
{
  // partial gradients
#ifdef _OPENMP
  long id = omp_get_thread_num();
#else
  long id = 0;
#endif
  real *grad = grads[id];
  THTensor_(zero)(gds[id]);

  // map
#pragma omp for
  for (long e = 0; e < nEdges; e++) {
    long n1 = edgeEnds[e*2+0]-1;
    long n2 = edgeEnds[e*2+1]-1;
    long label1 = (long)Y[n1]-1;
    long label2 = (long)Y[n2]-1;
    for (long s1 = 0; s1 < nStates[n1]; s1++) {
      for (long s2 = 0; s2 < nStates[n2]; s2++) {
        for (long f = 0; f < nEdgeFeatures; f++) {
          long map = edgeMap[e*em->stride[0]+s1*em->stride[1]+s2*em->stride[2]+f*em->stride[3]];
          if (map > 0) {
            real obs = ((s1 == label1) && (s2 == label2)) ? 1 : 0;
            grad[map-1] += Xedge[f*xe->stride[0]+e*xe->stride[1]] 
                  * (edgeBel[e*eb->stride[0]+s1*eb->stride[1]+s2*eb->stride[2]] - obs);
          }
        }
      }
    }
  }

  // reduce
#pragma omp barrier
  if (id==0) {
#ifdef _OPENMP
    long nthreads = omp_get_num_threads();
#else
    long nthreads = 1;
#endif
    for (int i = 0; i < nthreads; i++) {
      THTensor_(cadd)(gd, gd, 1.0, gds[i]);
    }
  }
}

  // clean up
  for (int i = 0; i < maxthreads; i++) {
    THTensor_(free)(gds[i]);
  }
  free(gds);
  free(grads);
  THTensor_(free)(ee);
  THTensor_(free)(ns);
  THTensor_(free)(yy);
  return 0;
}

static int gm_energies_(crfMakeNodePotentials)(lua_State *L) {
  // get args
  THTensor *xn = (THTensor *)luaT_checkudata(L, 1, torch_Tensor);
  THTensor *nm = (THTensor *)luaT_checkudata(L, 2, torch_Tensor);
  THTensor *ww = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 3, torch_Tensor));
  THTensor *ns = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 4, torch_Tensor));
  THTensor *np = (THTensor *)luaT_checkudata(L, 5, torch_Tensor);

  // dims
  long nNodes = nm->size[0];
  long nNodeFeatures= xn->size[0];

  // zero output
  THTensor_(zero)(np);

  // raw pointers
  real *Xnode = THTensor_(data)(xn);
  real *nodeMap = THTensor_(data)(nm);
  real *nodePot = THTensor_(data)(np);
  real *nStates = THTensor_(data)(ns);
  real *w = THTensor_(data)(ww);

  // generate node potentials
#pragma omp parallel for
  for (long n = 0; n < nNodes; n++) {
    for (long s = 0; s < nStates[n]; s++) {
      long np_i = n*np->stride[0]+s*np->stride[1];
      for (long f = 0; f < nNodeFeatures; f++) {
        long map = nodeMap[n*nm->stride[0]+s*nm->stride[1]+f*nm->stride[2]];
        if (map > 0) {
          nodePot[np_i] += w[map-1]*Xnode[f*xn->stride[0]+n*xn->stride[1]];

        }
      }
      nodePot[np_i] = exp(nodePot[np_i]);
    }
  }

  // clean up
  THTensor_(free)(ns);
  THTensor_(free)(ww);
  return 0;
}

static int gm_energies_(crfMakeEdgePotentials)(lua_State *L) {
  // get args
  THTensor *xe = (THTensor *)luaT_checkudata(L, 1, torch_Tensor);
  THTensor *em = (THTensor *)luaT_checkudata(L, 2, torch_Tensor);
  THTensor *ww = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 3, torch_Tensor));
  THTensor *ee = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 4, torch_Tensor));
  THTensor *ns = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 5, torch_Tensor));
  THTensor *ep = (THTensor *)luaT_checkudata(L, 6, torch_Tensor);

  // dims
  long nEdges = ep->size[0];
  long nEdgeFeatures = xe->size[0];

  // zero output
  THTensor_(zero)(ep);

  // raw pointers
  real *Xedge = THTensor_(data)(xe);
  real *edgeMap = THTensor_(data)(em);
  real *nStates = THTensor_(data)(ns);
  real *w = THTensor_(data)(ww);
  real *edgeEnds = THTensor_(data)(ee);
  real *edgePot = THTensor_(data)(ep);

  // generate edge potentials
#pragma omp parallel for
  for (long e = 0; e < nEdges; e++) {
    long n1 = edgeEnds[e*2+0]-1;
    long n2 = edgeEnds[e*2+1]-1;
    for (long s1 = 0; s1 < nStates[n1]; s1++) {
      for (long s2 = 0; s2 < nStates[n2]; s2++) {
        long ep_i = e*ep->stride[0]+s1*ep->stride[1]+s2*ep->stride[2];
        for (long f = 0; f < nEdgeFeatures; f++) {
          long map = edgeMap[e*em->stride[0]+s1*em->stride[1]+s2*em->stride[2]+f*em->stride[3]];
          if (map > 0) {
            real prod = w[map-1]*Xedge[f*xe->stride[0]+e*xe->stride[1]];
            edgePot[ep_i] += prod;
          }
        }
        edgePot[ep_i] = exp(edgePot[ep_i]);
      }
    }
  }

  // clean up
  THTensor_(free)(ww);
  THTensor_(free)(ee);
  THTensor_(free)(ns);
  return 0;
}

static const struct luaL_Reg gm_energies_(methods__) [] = {
  {"crfGradWrtNodes", gm_energies_(crfGradWrtNodes)},
  {"crfGradWrtEdges", gm_energies_(crfGradWrtEdges)},
  {"crfMakeNodePotentials", gm_energies_(crfMakeNodePotentials)},
  {"crfMakeEdgePotentials", gm_energies_(crfMakeEdgePotentials)},
  {NULL, NULL}
};

static void gm_energies_(Init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, gm_energies_(methods__), "gm");
  lua_pop(L,1);
}

#endif
