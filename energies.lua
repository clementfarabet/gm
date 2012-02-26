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
--     gm.energies - a list of functions to compute the energy of
--                    a graph, and the gradient ot that energy wrt
--                    its internal parameters
--
-- history: 
--     February 2012 - initial draft - Clement Farabet
----------------------------------------------------------------------

-- that table contains all the gradient functions
gm.energies = {}
gm.energies.crf = {}
gm.energies.mrf = {}

-- shortcuts
local zeros = torch.zeros
local ones = torch.ones
local eye = torch.eye
local sort = torch.sort
local log = torch.log
local exp = torch.exp

-- messages
local warning = function(msg)
   print(sys.COLORS.red .. msg .. sys.COLORS.none)
end

----------------------------------------------------------------------
-- Negative log-likelihood of a CRF
--
function gm.energies.crf.nll(graph,w,Xnode,Xedge,Y,nodeMap,edgeMap,inferMethod,maxIter)
   -- check sizes
   if Xnode:nDimension() == 2 then -- single example
      Xnode = Xnode:reshape(1,Xnode:size(1),Xnode:size(2))
      Xedge = Xedge:reshape(1,Xedge:size(1),Xedge:size(2))
      Y = Y:reshape(1,Y:size(1))
   end

   -- locals
   local Tensor = torch.Tensor
   local nInstances = Y:size(1)
   local nNodes = graph.nNodes
   local maxStates = nodeMap:size(2)
   local nNodeFeatures = Xnode:size(2)
   local nEdgeFeatures = Xedge:size(2)
   local nEdges = graph.nEdges
   local nStates = graph.nStates
   local edgeEnds = graph.edgeEnds

   -- init
   local nll = 0
   local grad = zeros(w:size())

   -- verbose
   if graph.verbose then
      print('<gm.energies.crf.nll> computing negative log-likelihood')
   end

   -- compute E=nll and dE/dw
   for i = 1,nInstances do
      -- make potentials
      gm.energies.crf.makePotentials(graph,w,Xnode[i],Xedge[i],nodeMap,edgeMap)

      -- perform inference
      local nodeBel,edgeBel,logZ = graph:infer(inferMethod,maxIter)

      -- update nll
      nll = nll - graph:getLogPotentialForConfig(Y[i]) + logZ

      -- compute gradients wrt nodes
      grad.gm.crfGradWrtNodes(Xnode[i],nodeMap,w,nStates,Y[i],nodeBel,grad)

      -- compute gradients wrt edges
      grad.gm.crfGradWrtEdges(Xedge[i],edgeMap,w,edgeEnds,nStates,Y[i],edgeBel,grad)
   end

   -- return nll and grad
   return nll,grad
end

----------------------------------------------------------------------
-- Make potentials for a CRF
--
function gm.energies.crf.makePotentials(graph,w,Xnode,Xedge,nodeMap,edgeMap)
   -- locals
   local Tensor = torch.Tensor
   local nNodes = graph.nNodes
   local maxStates = nodeMap:size(2)
   local nNodeFeatures = Xnode:size(1)
   local nEdgeFeatures = Xedge:size(1)
   local nEdges = graph.nEdges
   local nStates = graph.nStates
   local edgeEnds = graph.edgeEnds

   -- verbose
   if graph.verbose then
      print('<gm.energies.crf.makePotentials> making potentials from parameters')
   end

   -- generate node potentials
   local nodePot = graph.nodePot or Tensor()
   nodePot:resize(nNodes,maxStates)
   nodePot.gm.crfMakeNodePotentials(Xnode,nodeMap,w,nStates,nodePot)

   -- generate edge potentials
   local edgePot = graph.edgePot or Tensor()
   edgePot:resize(nEdges,maxStates,maxStates)
   nodePot.gm.crfMakeEdgePotentials(Xedge,edgeMap,w,edgeEnds,nStates,edgePot)

   -- store potentials
   graph:setPotentials(nodePot,edgePot)
end
