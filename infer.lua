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
--     gm.infer - a list of functions to perform inference
--
-- history: 
--     February 2012 - initial draft - Clement Farabet
----------------------------------------------------------------------

-- that table contains all the infer
gm.infer = {}

-- shortcuts
local zeros = torch.zeros
local ones = torch.ones
local eye = torch.eye
local sort = torch.sort
local log = torch.log
local eps = 1e-15

-- messages
local warning = function(msg)
   print(sys.COLORS.red .. msg .. sys.COLORS.none)
end

----------------------------------------------------------------------
-- exact inference: only adapted to super small graphs
--
function gm.infer.exact(graph)
   -- check args
   if not graph.nodePot or not graph.edgePot then
      xlua.error('missing nodePot/edgePot, please call graph:setFactors(...)','infer')
   end

   -- verbose
   if graph.verbose then
      print('<gm.infer.exact> doing exact inference')
   end

   -- local vars
   local Tensor = torch.Tensor
   local nNodes = graph.nNodes
   local maxStates = graph.nodePot:size(2)
   local nEdges = graph.nEdges
   local nStates = graph.nStates
   local edgeEnds = graph.edgeEnds

   -- init
   local config = ones(nNodes)
   local nodeBel = zeros(nNodes,maxStates)
   local edgeBel = zeros(nEdges,maxStates,maxStates)
   local Z = 0

   -- exact inference
   while true do
      -- get potential for configuration
      local pot = graph:getPotentialForConfig(config)

      -- update nodeBel
      for n = 1,nNodes do
         nodeBel[n][config[n]] = nodeBel[n][config[n]] + pot
      end

      -- update edgeBel
      for e = 1,nEdges do
         local n1 = edgeEnds[e][1]
         local n2 = edgeEnds[e][2]
         edgeBel[e][config[n1]][config[n2]] = edgeBel[e][config[n1]][config[n2]] + pot
      end

      -- update Z
      Z = Z + pot

      -- next config
      local idx
      for i = 1,nNodes do
         idx = i
         config[i] = config[i] + 1
         if config[i] <= nStates[i] then
            break
         else
            config[i] = 1
         end
      end

      -- stop when tried all configurations
      if idx == nNodes and config[config:size(1)] == 1 then
         break
      end
   end

   -- normalize
   nodeBel:div(Z)
   edgeBel:div(Z)

   -- return marginal beliefs, pairwise beliefs, and negative of free energy
   return nodeBel, edgeBel, log(Z)
end

----------------------------------------------------------------------
-- belief propagation: if the given graph is loopy, and maxIter is
-- set > 1, then loopy belief propagation is done
--
function gm.infer.bp(graph,maxIter)
   -- check args
   if not graph.nodePot or not graph.edgePot then
      xlua.error('missing nodePot/edgePot, please call graph:setFactors(...)','decode')
   end
   maxIter = maxIter or 1

   -- verbose
   if graph.verbose then
      print('<gm.infer.bp> inference with belief-propagation')
   end

   -- local vars
   local Tensor = torch.Tensor
   local nNodes = graph.nNodes
   local maxStates = graph.nodePot:size(2)
   local nEdges = graph.nEdges
   local nStates = graph.nStates
   local edgeEnds = graph.edgeEnds
   local V = graph.V
   local E = graph.E
   local nodePot = graph.nodePot
   local edgePot = graph.edgePot

   -- init
   local product = ones(nNodes,maxStates)
   local nodeBel = ones(nNodes,maxStates)
   local edgeBel = zeros(nEdges,maxStates,maxStates)
   local nodeBel_old = nodeBel:clone()
   local msg = zeros(nEdges*2,maxStates)
   local msg_old = zeros(nEdges*2,maxStates)

   -- propagate state normalizations
   msg.gm.bpInitMessages(edgeEnds,nStates,msg)

   -- do loopy belief propagation (if maxIter = 1, it's regular bp)
   local idx
   for i = 1,maxIter do
      idx = i
      -- pass messages, for all nodes (false = sum of products)
      msg.gm.bpComputeMessages(nodePot,edgePot,edgeEnds,nStates,E,V,msg,false)

      -- check convergence
      if (msg-msg_old):abs():sum() < 1e-4 then break end
      msg_old:copy(msg)
   end
   if graph.verbose then
      if idx == maxIter then
         warning('<gm.infer.bp> reached max iterations ('..maxIter..') before convergence')
      else
         print('<gm.infer.bp> decoded graph in '..idx..' iterations')
      end
   end

   -- compute marginal node beliefs
   msg.gm.bpComputeNodeBeliefs(nodePot,nodeBel,edgeEnds,nStates,E,V,product,msg)

   -- compute marginal edge beliefs
   msg.gm.bpComputeEdgeBeliefs(edgePot,edgeBel,nodeBel,edgeEnds,nStates,E,V,msg)

   -- compute negative free energy
   local logZ = msg.gm.bpComputeLogZ(nodePot,edgePot,nodeBel,edgeBel,edgeEnds,nStates,E,V)

   -- return marginal beliefs, pairwise beliefs, and negative of free energy
   return nodeBel, edgeBel, logZ
end
