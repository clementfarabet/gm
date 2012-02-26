#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/gm_energies.c"
#else

static int gm_energies_(crfGradWrtNodes)(lua_State *L) {
  // get args
  const void *id = torch_(Tensor_id);
  THTensor *xn = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 1, id));
  THTensor *nm = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 2, id));
  THTensor *ww = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 3, id));
  THTensor *ns = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 4, id));
  THTensor *yy = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 5, id));
  THTensor *nb = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 6, id));
  THTensor *gd = (THTensor *)luaT_checkudata(L, 7, id);

  // dims
  long nNodes = nm->size[0];
  long nNodeFeatures= xn->size[0];

  // raw pointers
  real *Xnode = THTensor_(data)(xn);
  real *nodeMap = THTensor_(data)(nm);
  real *nodeBel = THTensor_(data)(nb);
  real *nStates = THTensor_(data)(ns);
  real *w = THTensor_(data)(ww);
  real *Y = THTensor_(data)(yy);
  real *grad = THTensor_(data)(gd);

  // compute gradients wrt nodes
  for (long n = 0; n < nNodes; n++) {
    long label = (long)Y[n]-1;
    for (long s = 0; s < nStates[n]; s++) {
      for (long f = 0; f < nNodeFeatures; f++) {
        long map = nodeMap[n*nm->stride[0]+s*nm->stride[1]+f];
        if (map > 0) {
          real obs = (s == label) ? 1 : 0;
          grad[map-1] += Xnode[f*xn->stride[0]+n] * (nodeBel[n*nb->stride[0]+s] - obs);
        }
      }
    }
  }

  // clean up
  THTensor_(free)(xn);
  THTensor_(free)(nm);
  THTensor_(free)(ww);
  THTensor_(free)(ns);
  THTensor_(free)(yy);
  THTensor_(free)(nb);
  return 0;
}

static int gm_energies_(crfGradWrtEdges)(lua_State *L) {
  // get args
  const void *id = torch_(Tensor_id);
  THTensor *xe = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 1, id));
  THTensor *em = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 2, id));
  THTensor *ww = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 3, id));
  THTensor *ee = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 4, id));
  THTensor *ns = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 5, id));
  THTensor *yy = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 6, id));
  THTensor *eb = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 7, id));
  THTensor *gd = (THTensor *)luaT_checkudata(L, 8, id);

  // dims
  long nEdges = em->size[0];
  long nEdgeFeatures = xe->size[0];

  // raw pointers
  real *Xedge = THTensor_(data)(xe);
  real *edgeMap = THTensor_(data)(em);
  real *edgeBel = THTensor_(data)(eb);
  real *nStates = THTensor_(data)(ns);
  real *w = THTensor_(data)(ww);
  real *edgeEnds = THTensor_(data)(ee);
  real *Y = THTensor_(data)(yy);
  real *grad = THTensor_(data)(gd);

  // compute gradients wrt edges
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
            grad[map-1] += Xedge[f*xe->stride[0]+e] * (edgeBel[e*eb->stride[0]+s1*eb->stride[1]+s2] - obs);
          }

        }
      }
    }
  }

  // clean up
  THTensor_(free)(xe);
  THTensor_(free)(em);
  THTensor_(free)(ww);
  THTensor_(free)(ee);
  THTensor_(free)(ns);
  THTensor_(free)(yy);
  THTensor_(free)(eb);
  return 0;
}

static int gm_energies_(crfMakeNodePotentials)(lua_State *L) {
  // get args
  const void *id = torch_(Tensor_id);
  THTensor *xn = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 1, id));
  THTensor *nm = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 2, id));
  THTensor *ww = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 3, id));
  THTensor *ns = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 4, id));
  THTensor *np = (THTensor *)luaT_checkudata(L, 5, id);

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
  for (long n = 0; n < nNodes; n++) {
    for (long s = 0; s < nStates[n]; s++) {
      long np_i = n*np->stride[0]+s*np->stride[1];
      for (long f = 0; f < nNodeFeatures; f++) {
        long map = nodeMap[n*nm->stride[0]+s*nm->stride[1]+f];
        if (map > 0) {
          nodePot[np_i] += w[map-1]*Xnode[n+nNodes*f];
        }
      }
      nodePot[np_i] = exp(nodePot[np_i]);
    }
  }

  // clean up
  THTensor_(free)(xn);
  THTensor_(free)(nm);
  THTensor_(free)(ns);
  THTensor_(free)(ww);
  return 0;
}

static int gm_energies_(crfMakeEdgePotentials)(lua_State *L) {
  // get args
  const void *id = torch_(Tensor_id);
  THTensor *xe = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 1, id));
  THTensor *em = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 2, id));
  THTensor *ww = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 3, id));
  THTensor *ee = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 4, id));
  THTensor *ns = THTensor_(newContiguous)((THTensor *)luaT_checkudata(L, 5, id));
  THTensor *ep = (THTensor *)luaT_checkudata(L, 6, id);

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
  for (long e = 0; e < nEdges; e++) {
    long n1 = edgeEnds[e*2+0]-1;
    long n2 = edgeEnds[e*2+1]-1;
    for (long s1 = 0; s1 < nStates[n1]; s1++) {
      for (long s2 = 0; s2 < nStates[n2]; s2++) {
        long ep_i = e*ep->stride[0]+s1*ep->stride[1]+s2*ep->stride[2];
        for (long f = 0; f < nEdgeFeatures; f++) {
          long map = edgeMap[e*em->stride[0]+s1*em->stride[1]+s2*em->stride[2]+f*em->stride[3]];
          if (map > 0) {
            real prod = w[map-1]*Xedge[e+nEdges*f];
            edgePot[ep_i] += prod;
          }
        }
        edgePot[ep_i] = exp(edgePot[ep_i]);
      }
    }
  }

  // clean up
  THTensor_(free)(xe);
  THTensor_(free)(em);
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
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, gm_energies_(methods__), "gm");
  lua_pop(L,1);
}

#endif
