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
local Tensor = torch.Tensor
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

   -- compute E=nll and dE/dw
   for i = 1,nInstances do
      -- make potentials
      gm.energies.crf.makePotentials(graph,w,Xnode[i],Xedge[i],nodeMap,edgeMap)

      -- perform inference
      local nodeBel,edgeBel,logZ = graph:infer(inferMethod,maxIter)

      -- update nll
      nll = nll - graph:getLogPotentialForConfig(Y[i]) + logZ

      -- compute grad
      for n = 1,nNodes do
         for s = 1,nStates[n] do
            for f = 1,nNodeFeatures do
               if nodeMap[n][s][f] > 0 then
                  local obs = 0
                  if s == Y[i][n] then
                     obs = 1
                  end
                  grad[nodeMap[n][s][f]] = grad[nodeMap[n][s][f]] + Xnode[i][f][n]*(nodeBel[n][s] - obs)
               end
            end
         end
      end
      for e = 1,nEdges do
         local n1 = edgeEnds[e][1]
         local n2 = edgeEnds[e][2]
         for s1 = 1,nStates[n1] do
            for s2 = 1,nStates[n2] do
               for f = 1,nEdgeFeatures do
                  if edgeMap[e][s1][s2][f] > 0 then
                     local obs = 0
                     if s1 == Y[i][n1] and s2 == Y[i][n2] then
                        obs = 1
                     end
                     grad[edgeMap[e][s1][s2][f]] = grad[edgeMap[e][s1][s2][f]] + Xedge[i][f][e]*(edgeBel[e][s1][s2] - obs)
                  end
               end
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
function gm.energies.crf.makePotentials(graph,w,Xnode,Xedge,nodeMap,edgeMap)
   -- locals
   local nNodes = graph.nNodes
   local maxStates = nodeMap:size(2)
   local nNodeFeatures = Xnode:size(1)
   local nEdgeFeatures = Xedge:size(1)
   local nEdges = graph.nEdges
   local nStates = graph.nStates
   local edgeEnds = graph.edgeEnds

   -- generate node potentials
   local nodePot = zeros(nNodes,maxStates)
   for n = 1,nNodes do
      for s = 1,nStates[n] do
         for f = 1,nNodeFeatures do
            if nodeMap[n][s][f] > 0 then
               nodePot[n][s] = nodePot[n][s] + w[nodeMap[n][s][f]]*Xnode[f][n]
            end
         end
         nodePot[n][s] = exp(nodePot[n][s])
      end
   end

   -- generate edge potentials
   local edgePot = zeros(nEdges,maxStates,maxStates)
   for e = 1,nEdges do
      local n1 = edgeEnds[e][1]
      local n2 = edgeEnds[e][2]
      for s1 = 1,nStates[n1] do
         for s2 = 1,nStates[n2] do
            for f = 1,nEdgeFeatures do
               if edgeMap[e][s1][s2][f] > 0 then
                  edgePot[e][s1][s2] = edgePot[e][s1][s2] + w[edgeMap[e][s1][s2][f]]*Xedge[f][e]
               end
            end
            edgePot[e][s1][s2] = exp(edgePot[e][s1][s2])
         end
      end
   end

   -- store potentials
   graph:setPotentials(nodePot,edgePot)
end
