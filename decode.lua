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
--     gm.decode - a list of functions to decode optimal states
--
-- history: 
--     February 2012 - initial draft - Clement Farabet
----------------------------------------------------------------------

-- that table contains all the decode
gm.decode = {}

-- shortcuts
local zeros = torch.zeros
local ones = torch.ones
local eye = torch.eye
local sort = torch.sort

-- messages
local warning = function(msg)
   print(sys.COLORS.red .. msg .. sys.COLORS.none)
end

----------------------------------------------------------------------
-- exact decoding: only adapted to super small graphs
--
function gm.decode.exact(graph)
   -- check args
   if not graph.nodePot or not graph.edgePot then
      xlua.error('missing nodePot/edgePot, please call graph:setFactors(...)','decode')
   end

   -- verbose
   if graph.verbose then
      print('<gm.decode.bp> decoding using exhaustive search')
   end

   -- local vars
   local Tensor = torch.Tensor
   local nNodes = graph.nNodes
   local maxStates = graph.nodePot:size(2)
   local nEdges = graph.nEdges
   local nStates = graph.nStates

   -- init
   local config = ones(nNodes)
   local optimalconfig = ones(nNodes)

   -- decode, exactly
   local maxpot = -1
   while true do
      -- get potential for configuration
      local pot = graph:getPotentialForConfig(config)

      -- compare configurations
      if pot > maxpot then
         maxpot = pot
         optimalconfig:copy(config)
      end

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

   -- store and return optimal config
   graph.optimal = optimalconfig
   return optimalconfig
end

----------------------------------------------------------------------
-- belief propagation: if the given graph is loopy, and maxIter is
-- set > 1, then loopy belief propagation is done
--
function gm.decode.bp(graph,maxIter)
   -- check args
   if not graph.nodePot or not graph.edgePot then
      xlua.error('missing nodePot/edgePot, please call graph:setFactors(...)','decode')
   end
   maxIter = maxIter or graph.maxIter or 1

   -- verbose
   if graph.verbose then
      print('<gm.decode.bp> decoding using belief-propagation')
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
   local nodeBel_old = nodeBel:clone()
   local msg = zeros(nEdges*2,maxStates)
   local msg_old = zeros(nEdges*2,maxStates)

   -- propagate state normalizations
   for e = 1,nEdges do
      local n1 = edgeEnds[e][1]
      local n2 = edgeEnds[e][2]
      msg[{ e,{1,nStates[n2]} }] = 1/nStates[n2] -- n1 ==> n2
      msg[{ e+nEdges,{1,nStates[n1]} }] = 1/nStates[n1] -- n2 ==> n1
   end

   -- do loopy belief propagation (if maxIter = 1, it's regular bp)
   local idx
   for i = 1,maxIter do
      idx = i
      -- pass messages, for all nodes (true = max of products)
      msg.gm.bpComputeMessages(nodePot,edgePot,edgeEnds,nStates,E,V,msg,true)

      -- check convergence
      if (msg-msg_old):abs():sum() < 1e-4 then break end
      msg_old:copy(msg)
   end
   if graph.verbose then
      if idx == maxIter then
         warning('<gm.decode.bp> reached max iterations ('..maxIter..') before convergence')
      else
         print('<gm.decode.bp> decoded graph in '..idx..' iterations')
      end
   end

   -- compute marginal node beliefs
   msg.gm.bpComputeNodeBeliefs(nodePot,nodeBel,edgeEnds,nStates,E,V,product,msg)

   -- get argmax of nodeBel: that's the optimal config
   local pot, optimalconfig = nodeBel:max(2)
   optimalconfig = optimalconfig:squeeze(2)

   -- store and return optimal config
   graph.optimal = optimalconfig
   return optimalconfig,nodeBel
end
