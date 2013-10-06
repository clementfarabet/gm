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
function gm.energies.crf.nll(graph,w,nodeMap,edgeMap,inferMethod,maxIter,Y,Xnode,Xedge)
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
      gm.energies.crf.makePotentials(graph,w,nodeMap,edgeMap,Xnode[i],Xedge[i])

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
-- Negative log-likelihood of an MRF
--
function gm.energies.mrf.nll(graph,w,nodeMap,edgeMap,inferMethod,maxIter,Y)
   -- locals
   local Tensor = torch.Tensor
   local nNodes = graph.nNodes
   local nEdges = graph.nEdges
   local nStates = graph.nStates
   local edgeEnds = graph.edgeEnds
   local maxStates = nodeMap:size(2)
   local nInstances = Y:size(1)

   -- verbose
   if graph.verbose then
      print('<gm.energies.mrf.nll> computing negative log-likelihood')
   end

   -- compute sufficient statistics
   local suffStats = g.w:clone():zero()
   for i = 1,nInstances do
      local y = Y[i]
      for n = 1,nNodes do
         local idx = nodeMap[n][y[n]]
         if idx > 0 then
            suffStats[idx] = suffStats[idx] + 1
         end
      end
      for e = 1,nEdges do
         local n1 = edgeEnds[e][1]
         local n2 = edgeEnds[e][2]
         local idx = edgeMap[e][y[n1]][y[n2]]
         if idx > 0 then
            suffStats[idx] = suffStats[idx] + 1
         end
      end
   end

   -- make potentials
   gm.energies.mrf.makePotentials(graph,w,nodeMap,edgeMap)

   -- perform inference
   local nodeBel,edgeBel,logZ = graph:infer(inferMethod,maxIter)

   -- update nll
   local nll = -w*suffStats + logZ*nInstances

   -- compute gradients wrt nodes
   local grad = suffStats:clone():mul(-1)
   for n = 1,nNodes do
      for s = 1,nStates[n] do
         local idx = nodeMap[n][s]
         if idx > 0 then
            grad[idx] = grad[idx] + nInstances * nodeBel[n][s]
         end
      end
   end

   -- compute gradients wrt edges
   for e = 1,nEdges do
      local n1 = edgeEnds[e][1]
      local n2 = edgeEnds[e][2]
        for s1 = 1,nStates[n1] do
            for s2 = 1,nStates[n2] do
               local idx = edgeMap[e][s1][s2]
               if idx > 0 then
                  grad[idx] = grad[idx] + nInstances * edgeBel[e][s1][s2]
               end
            end
        end
    end

   -- return nll and grad
   return nll,grad
end

----------------------------------------------------------------------
-- Make potentials for a CRF
--
function gm.energies.crf.makePotentials(graph,w,nodeMap,edgeMap,Xnode,Xedge)
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

----------------------------------------------------------------------
-- Make potentials for an MRF
--
function gm.energies.mrf.makePotentials(graph,w,nodeMap,edgeMap)
   -- locals
   local Tensor = torch.Tensor
   local exp = math.exp
   local nNodes = graph.nNodes
   local maxStates = nodeMap:size(2)
   local nEdges = graph.nEdges
   local nStates = graph.nStates
   local edgeEnds = graph.edgeEnds

   -- verbose
   if graph.verbose then
      print('<gm.energies.mrf.makePotentials> making potentials from parameters')
   end

   -- generate node potentials
   local nodePot = graph.nodePot or Tensor()
   nodePot:resize(nNodes,maxStates)
   nodePot:zero()
   for n = 1,nNodes do
      for s = 1,nStates[n] do
         local idx = nodeMap[n][s]
         if idx == 0 then
            nodePot[n][s] = 1
         else
            nodePot[n][s] = exp(w[idx])
         end
      end
   end

   -- generate edge potentials
   local edgePot = graph.edgePot or Tensor()
   edgePot:resize(nEdges,maxStates,maxStates)
   edgePot:zero()
   for e = 1,nEdges do
      local n1 = edgeEnds[e][1]
      local n2 = edgeEnds[e][2]
      for s1 = 1,nStates[n1] do
         for s2 = 1,nStates[n2] do
            local idx = edgeMap[e][s1][s2]
            if idx == 0 then
               edgePot[e][s1][s2] = 1
            else
               edgePot[e][s1][s2] = exp(w[idx])
            end
         end
      end
   end

   -- store potentials
   graph:setPotentials(nodePot,edgePot)
end
