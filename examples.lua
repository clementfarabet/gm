----------------------------------------------------------------------
--
-- Copyright (c) 2012 Clement Farabet
-- 
-- Permission is hereby granted, free of charge, to any person obtaining
-- a copy of this software and associated documentation files (the
-- "Software"), to deal in the Software without restriction, including
-- without limitation the rights to use, copy, modify, merge, publish,
-- distribute, sublicense, and/or sell copies of the Software, and to
-- permit persons to whom the Software is furnished to do so, subject to
-- the following conditions:
-- 
-- The above copyright notice and this permission notice shall be
-- included in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
-- EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
-- MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
-- NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
-- LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
-- OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
-- WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
-- 
----------------------------------------------------------------------
-- description:
--     gm.examples - a list of example/test functions
--
-- history: 
--     February 2012 - initial draft - Clement Farabet
----------------------------------------------------------------------

-- that table contains all the examples
gm.examples = {}

-- shortcuts
local zeros = torch.zeros
local ones = torch.ones
local eye = torch.eye
local Tensor = torch.Tensor
local sort = torch.sort
local log = torch.log
local exp = torch.exp

-- messages
local warning = function(msg)
   print(sys.COLORS.red .. msg .. sys.COLORS.none)
end

----------------------------------------------------------------------
-- Simple example, doing decoding and inference
--
function gm.examples.simple()
   -- define graph structure
   local nNodes = 10
   local adjacency = ones(nNodes,nNodes) - eye(nNodes)
   local nEdges = nNodes^2 - nNodes

   -- unary potentials
   local nStates = 2
   local nodePot = Tensor{{1,3}, {9,1}, {1,3}, {9,1}, {1,3},
                          {1,3}, {9,1}, {1,3}, {9,1}, {1,1}}

   -- joint potentials
   local edgePot = Tensor(nEdges,nStates,nStates)
   local basic = Tensor{{2,1}, {1,2}}
   for e = 1,nEdges do
      edgePot[e] = basic
   end

   -- create graph
   local g = gm.graph{adjacency=adjacency, nStates=nStates, 
                      nodePot=nodePot, edgePot=edgePot, verbose=true}

   -- exact inference
   local exact = g:decode('exact')
   print()
   print('<gm.testme> exact optimal config:')
   print(exact)

   local nodeBel,edgeBel,logZ = g:infer('exact')
   print('<gm.testme> node beliefs:')
   print(nodeBel)
   --print('<gm.testme> edge beliefs:')
   --print(edgeBel)
   print('<gm.testme> log(Z):')
   print(logZ)

   -- bp inference
   local bp = g:decode('bp',10)
   print()
   print('<gm.testme> optimal config with belief propagation:')
   print(bp)

   local nodeBel,edgeBel,logZ = g:infer('bp',10)
   print('<gm.testme> node beliefs:')
   print(nodeBel)
   --print('<gm.testme> edge beliefs:')
   --print(edgeBel)
   print('<gm.testme> log(Z):')
   print(logZ)

   -- done
   return g
end

----------------------------------------------------------------------
-- Example of how to train a CRF
--
function gm.examples.trainCRF()
   -- training data
   local y = torch.Tensor(1059,28):apply(function() return torch.bernoulli(0.5)+1 end)
   local nInstances = y:size(1)
   local nNodes = y:size(2)

   -- define graph structure
   local nStates = y:max()
   local adj = zeros(nNodes,nNodes)
   for i = 1,nNodes-1 do
      adj[i][i+1] = 1
   end
   adj = adj + adj:t()
   local graph = gm.graph{adjacency=adj, nStates=nStates, verbose=false}
   local nEdges = graph.nEdges
   local maxStates = nStates

   -- bias features
   local nFeatures = 1
   local Xnode = ones(nInstances,nFeatures,nNodes)
   local Xedge = ones(nInstances,nFeatures,nEdges)

   -- map
   local nodeMap = zeros(nNodes,maxStates,nFeatures)
   nodeMap:select(2,1):fill(1)
   local edgeMap = zeros(nEdges,maxStates,maxStates,nFeatures)
   edgeMap:select(2,1):select(2,1):fill(2)
   edgeMap:select(2,2):select(2,1):fill(3)
   edgeMap:select(2,1):select(2,2):fill(4)

   -- initialize weights
   local nParams = math.max(nodeMap:max(), edgeMap:max())
   local w = zeros(nParams)

   -- function to minimize
   local func = function(w)
      return gm.energies.crf.nll(graph,w,Xnode,Xedge,y,nodeMap,edgeMap,'bp',10)
   end

   -- optimize
   w = optim.lbfgs(func, w, {verbose=true, maxIter=1})

   -- make potentials
   gm.energies.crf.makePotentials(graph,w,Xnode[1],Xedge[1],nodeMap,edgeMap)
   local optimal = graph:decode('bp',10)
   print(optimal)
end
